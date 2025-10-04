# input: image (Image.Image) + description (str) + model name (str) -> output: coordination (tuple)
# All in one file
import json
from pathlib import Path
import time, random
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import io, re, base64, copy, os, cv2, numpy as np
from typing import Tuple, List, Any, Dict
from PIL import Image, ImageDraw, ImageFont
from types import SimpleNamespace
from vllm import LLM, SamplingParams
from typing import Tuple, Optional, Union

# constant
_X_OPTIONS = {"left", "center", "right"}
_Y_OPTIONS = {"top", "center", "bottom"}
_MIN_SIDE   = 28         # cropped width / height must be ≥ 28 px

MERGE_TOOL_EXAMPLE = '''User: [Image_0 is displayed above] Could you identify the location of the “Close” button in this interface?

Assistant: <think>I’d like to zoom in on the menu icon at the top-left corner to get a clearer view.</think>  
<crop>(Image_0, (10, 20), (110, 100))</crop>

User: [Image_1 is displayed above]  # (cropped image returned)xw

Assistant: <think>In this cropped image, I can not see the “Close” button, I will use the `extract` tool to find the possible area where the “Close” button is located.</think>
<extract>(Image_0, left, top)</extract>

User: [Image_2 is displayed above]  # (possible area returned)

Assistant: <think>In this cropped image, I can see the approximate position of the “Close” button—it sits near the center of the region, slightly toward the lower-right. it’s at (45, 60).</think>  
<answer>(Image_2, (45, 60))</answer>'''

TOOL_PROMPT = f"""You should use three tools to help you analyze the image and find the target coordinate:
1. **crop**: This tool allows you to crop a specific area of the image by specifying the top-left and bottom-right coordinates of the rectangle you want to crop.
2. **extract**: This tool allows you to extract one quarter of the image based on the specified horizontal and vertical positions (left, center, right for x-axis; top, center, bottom for y-axis).
3. **find_color**: This tool allows you to find a specific color in the image by providing the RGB values of the target color.
Example Usage:
<crop>(Image_0, (10, 20), (110, 100))</crop> # Crop a rectangle from Image_0 from (10, 20) to (110, 100)
<extract>(Image_0, left, top)</extract> # Extract the top-left quarter of Image_0
<find_color>(Image_2, (255, 0, 0))</find_color> # Find the red color in Image_2
Before each tool call, please enclose your reasoning within <think>...</think> tags. 
In the end, you should return your final answer using the <answer>...</answer> tag.
In the <answer> tag, you should return the image and the coordinate of the target object in the format (Image_X, (x, y)), where Image_X is the image containing the target object and (x, y) is the coordinate of the target object.
Here is an example of how to find the final target coordinate:
{MERGE_TOOL_EXAMPLE}
Now, let's work on the real task:
[Image_0 is displayed below]"""

# --------- Tools ---------

def crop(
    img: Image.Image,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int]
) -> Tuple[Any, str]:
    """
    crop a rectangular region from an image.
    Args:
        img (Image.Image): The image to crop.
        top_left (Tuple[int, int]): The top-left corner of the cropping rectangle (x1, y1).
        bottom_right (Tuple[int, int]): The bottom-right corner of the cropping rectangle (x2, y2).
    Returns:
        Tuple[bytes, str]:
            cropped image as PNG bytes,
            message -> As text describing the crop operation
            offset -> For calculating the coordinates relative to the original image
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    width, height = img.size

    # Boundary checks
    errors = []
    if x1 < 0:
        errors.append(f"x1 ({x1}) < 0")
    if y1 < 0:
        errors.append(f"y1 ({y1}) < 0")
    if x2 > width:
        errors.append(f"x2 ({x2}) > image width ({width})")
    if y2 > height:
        errors.append(f"y2 ({y2}) > image height ({height})")
    if x2 <= x1:
        errors.append(f"x2 ({x2}) ≤ x1 ({x1})")
    if y2 <= y1:
        errors.append(f"y2 ({y2}) ≤ y1 ({y1})")

    if errors:
        detail = "; ".join(errors)
        msg = (
            "<|Tool_Error|>: "
            f"Invalid crop coordinates: {detail}. "
            f"Image size: width={width}, height={height}; "
            f"Requested: top_left=({x1}, {y1}), bottom_right=({x2}, {y2})."
        )
        return None, msg, None

    # Ensure minimum dimensions
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < 28 or crop_h < 28:
        return None, (
            "<|Tool_Error|>: "
            f"Crop size too small: width={crop_w}, height={crop_h}. "
            "Both crop_w and crop_h must be at least 28 pixels."
        ), None

    # Adjust for edge case of width or height exactly 3
    if crop_w == 3:
        if x2 < width:
            x2 += 1
        elif x1 > 0:
            x1 -= 1
    if crop_h == 3:
        if y2 < height:
            y2 += 1
        elif y1 > 0:
            y1 -= 1

    # Perform crop
    cropped = img.crop((x1, y1, x2, y2))
    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG")
    data = buffer.getvalue()

    # # Save backup
    # os.makedirs("backup", exist_ok=True)
    # filename = f"backup/output_(({x1},{y1}),({x2},{y2})).png"
    # cropped.save(filename, format="PNG")

    # Descriptive message
    new_w, new_h = cropped.size
    message = f"Cropped a region of size {new_w}×{new_h} pixels."
    return data, message, top_left

def extract(
    img_input: Union[str, Image.Image],
    x_pos: str,
    y_pos: str
) -> Tuple[Optional[bytes], str, Optional[Tuple[int, int]]]:
    """
    Extract one-quarter of an image (½ width × ½ height).

    Returns:
        data (bytes | None)        : PNG bytes of the cropped image, or None on error
        message (str)              : success / error message
        offset  (Tuple[int,int] | None): (x0, y0) of the crop in the original image
    """
    # 1. load image ------------------------------------------------------
    if isinstance(img_input, Image.Image):
        img = img_input
    elif isinstance(img_input, str):
        if not os.path.isfile(img_input):
            return None, f"File not found: {img_input}", None
        try:
            img = Image.open(img_input)
        except Exception as e:
            return None, f"Could not open image: {e}", None
    else:
        return None, "Invalid img_input: must be file path or PIL.Image.Image", None

    # 2. validate positions ---------------------------------------------
    x_pos, y_pos = x_pos.lower(), y_pos.lower()
    if x_pos not in _X_OPTIONS or y_pos not in _Y_OPTIONS:
        return None, (
            f"x_pos must be one of {_X_OPTIONS}, "
            f"y_pos must be one of {_Y_OPTIONS}."
        ), None

    W, H = img.size
    half_w, half_h = W // 2, H // 2              # target crop size

    # --- NEW: minimum-size check ---------------------------------------
    if half_w < _MIN_SIDE or half_h < _MIN_SIDE:
        return None, (
            "<|Tool_Error|>: "
            f"Crop size too small: width={half_w}, height={half_h}. "
            f"Both crop_w and crop_h must be at least {_MIN_SIDE} pixels."
        ), None

    # 3. compute offset --------------------------------------------------
    x0 = 0 if x_pos == "left"   else (W - half_w) // 2 if x_pos == "center" else W - half_w
    y0 = 0 if y_pos == "top"    else (H - half_h) // 2 if y_pos == "center" else H - half_h

    # 4. crop & serialize -----------------------------------------------
    crop_box = (x0, y0, x0 + half_w, y0 + half_h)
    cropped  = img.crop(crop_box)

    with io.BytesIO() as buf:
        cropped.save(buf, format="PNG")
        data = buf.getvalue()

    message = f"Cropped a region of size {half_w}×{half_h}."
    return data, message, (x0, y0)

# ΔE（CIE76）------------------------------------------------------------
def _delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    return float(np.linalg.norm(lab1.astype(np.float32) - lab2.astype(np.float32)))

# main function ---------------------------------------------------------------
def find_color(
    img_input : Union[str, Image.Image],
    target_rgb: Tuple[int, int, int],
) -> Tuple[Optional[bytes], str, Optional[Tuple[int, int]]]:
    """
    1. use 10*10 sliding window to find the best match for target_rgb in the image;
    2. take the center of the best match as the center of a 200×200 window;
    3. return the cropped 200×200 window as PNG bytes.
    """
    # ---------- read image ---------- #
    if isinstance(img_input, Image.Image):
        # PIL → ndarray (BGR)
        img_bgr = cv2.cvtColor(np.asarray(img_input), cv2.COLOR_RGB2BGR)
    elif isinstance(img_input, str):
        if not os.path.isfile(img_input):
            return None, f"File not found: {img_input}", None
        img_bgr = cv2.imread(img_input)
        if img_bgr is None:
            return None, f"Could not open image: {img_input}", None
    else:
        return None, "Invalid img_input: must be file path or PIL.Image.Image", None

    h, w = img_bgr.shape[:2]

    # ---------- target window size ---------- #
    ws = 200   
    if min(h, w) < ws:       
        ws = min(h, w)
    half = ws // 2

    # ---------- prepare target color ---------- #
    tgt_lab = cv2.cvtColor(
        np.uint8([[target_rgb[::-1]]]),   # BGR
        cv2.COLOR_BGR2LAB
    )[0, 0]

    # ---------- sliding window search ---------- #
    best = {"delta_e": 1e9}
    sp, stride = 10, 10
    for y in range(0, h - sp + 1, stride):
        for x in range(0, w - sp + 1, stride):
            patch = img_bgr[y:y+sp, x:x+sp]
            mean_lab = cv2.cvtColor(
                patch.mean(axis=(0,1), dtype=np.float32).reshape(1,1,3).astype(np.uint8),
                cv2.COLOR_BGR2LAB
            )[0, 0]
            de = _delta_e_cie76(mean_lab, tgt_lab)
            if de < best["delta_e"]:
                best.update({"delta_e": de, "center": (x + sp//2, y + sp//2)})

    # ---------- locate best match ---------- #
    cx, cy = best["center"]
    ws, half = 200, 100
    x0 = max(0, min(cx - half, w - ws))
    y0 = max(0, min(cy - half, h - ws))
    window = img_bgr[y0:y0+ws, x0:x0+ws]

    # ---------- PNG serialization ---------- #
    success, buf = cv2.imencode(".png", window)
    if not success:
        return None, "Failed to encode PNG.", None
    data = buf.tobytes()

    msg = f"Cropped a region of size {ws}×{ws} matching target RGB {target_rgb}."
    return data, msg, (x0, y0)

# --------- Parse and call tools ---------

def call_crop(tool_cmd: str, images: List[Image.Image]):
    """
    1. Convert Crop usage: 
    (Image_0, (10, 20), (110, 100)) ->
    crop(
        image: Image.Image,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int]
    )
    2. Perform the crop operation on the specified image.
    Args:
        tool_cmd (str): The command string containing image name, id_num, and coordinates.
        images (List[Image.Image]): List of images available for cropping.
    Returns:
        Tuple[bytes, str]:
            cropped image as PNG bytes,
            message -> As text describing the crop operation
            offset -> For calculating the coordinates relative to the original image
            imgae id_num -> The index of the image in the images list
    """
    try:
        # extract image name, id_num, and coordinates from the tool command
        pattern = re.compile(
            r"""\(?\s*
                ([A-Za-z0-9_-]+)
                _(\d+)
                \s*,\s*
                \(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)   # ③④ x1,y1
                \s*,\s*
                \(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)   # ⑤⑥ x2,y2
                \s*\)?
            """,
            re.VERBOSE,
        )
        m = pattern.fullmatch(tool_cmd)
        
        if m:
            image_name, id_num, x1, y1, x2, y2 = m.groups()
            id_num = int(id_num)
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        top_left = (x1, y1)
        bottom_right = (x2, y2)
        img = images[id_num]
        # 4. call the crop tool
        data, message, off_set = crop(img, top_left, bottom_right)
        return data, message, off_set, id_num

    except Exception as e:
        usage = (
            f"<crop>(Image_0,(x1, y1), (x2, y2))</crop>, "
            "The first argument is a string, which record the image name with an ID, e.g. Image_0 "
            "and the second and third arguments are tuples representing the top-left and bottom-right coordinates."
        )
        return None, f"<|Tool_Error|>, correct usage: {usage}", None, None

def call_extract(tool_cmd: str, images: List[Image.Image]) -> bytes:
    """
    1. Convert Extract usage:
    (Image_0, x_pos, y_pos) ->
    extract(
        image: Image.Image,
        x_pos: Literal["left", "center", "right"],
        y_pos: Literal["top", "center", "bottom"]
    )
    2. Perform the extract operation on the specified image.
    Args:
        tool_cmd (str): The command string containing image name, x_pos, and y_pos.
        images (List[Image.Image]): List of images available for extraction.
    Returns:
        Tuple[bytes, str]:
            extracted image as PNG bytes,
            message -> As text describing the extract operation
            offset -> For calculating the coordinates relative to the original image
    """
    try:
        # 1. Extract image name and positions from the tool command
        pattern = re.compile(
            r"""
            \(?\s*
                ([A-Za-z0-9_-]+)          # ① image name  (e.g. Image)
                _(\d+)                    # ② id number   (e.g. 0)
                \s*,\s*
                ["']?(left|center|right)["']?   # ③ x_pos   (allow optional quotes)
                \s*,\s*
                ["']?(top|center|bottom)["']?   # ④ y_pos   (allow optional quotes)
                \s*\)?
            """,
            re.VERBOSE | re.IGNORECASE,
        )
        m = pattern.fullmatch(tool_cmd)
        
        if m:
            image_name, id_num, x_pos, y_pos = m.groups()
            id_num = int(id_num)
        
        img = images[id_num]
        data, message, off_set = extract(img, x_pos, y_pos)
        return data, message, off_set, id_num

    except Exception as e:
        usage = (
            f"<extract>(Image_0, x_pos, y_pos)</extract>, "
            "The first argument is a string, which record the image name with an ID, e.g. Image_0 "
            "and the second and third arguments are strings representing the horizontal and vertical positions."
        )
        return None, f"<|Tool_Error|>, correct usage: {usage}", None, None

def call_color(tool_cmd: str, images: List[Image.Image]):
    """
    1. Convert find_color usage:
    (Image_0, (R, G, B)) ->
    find_color(
        img_input: Image.Image,
        target_rgb: Tuple[int, int, int]
    )
    2. Perform the color-based window extraction on the specified image.

    Args:
        tool_cmd (str): The command string containing image name and RGB triple.
                        e.g. "Image_2, (255, 0, 0)"
        images (List[Image.Image]): List of PIL Image objects available.

    Returns:
        Tuple containing:
            1. bytes | None:
            PNG bytes of the extracted 200×200 window if successful; otherwise None.
            2. str:
            A message describing success (with ΔE/offset) or the error.
            3. Tuple[int, int] | None:
            (x_offset, y_offset) of the window’s top-left corner, or None on failure.
            4. int | None:
            The image index used (id_num), or None on failure.
    """
    try:
        # parse "Image_<id>, (R, G, B)"
        pattern = re.compile(r'''
            ^\(\s*([A-Za-z0-9_-]+)_(\d+)\s*,\s*     # Image_0
            \(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)\s*  # (123,123,12)
            \)$
        ''', re.VERBOSE)
        m = pattern.match(tool_cmd)
        if not m:
            raise ValueError("Invalid syntax")

        _, id_str, r_str, g_str, b_str = m.groups()
        id_num = int(id_str)
        target_rgb = (int(r_str), int(g_str), int(b_str))

        # bounds check
        if id_num < 0 or id_num >= len(images):
            return None, f"<|Tool_Error|> Invalid image index: {id_num}", None, None
        if not all(0 <= c <= 255 for c in target_rgb):
            return None, f"<|Tool_Error|> RGB values must be in [0,255]: {target_rgb}", None, None

        img = images[id_num]
        # call the find_color tool
        data, message, offset = find_color(img, target_rgb)
        print(f"find_color: {message}, offset: {offset}")
        return data, message, offset, id_num

    except Exception as e:
        print(f"Error in call_color: {e}")
        usage = (
            "Usage: <find_color>(Image_<id>, (R, G, B))</find_color>\n"
            "  - <id>: integer index into the provided images list\n"
            "  - R, G, B: integers in [0,255]\n"
            "Example: <find_color>(Image_2, (255, 0, 0))</find_color>"
        )
        return None, f"<|Tool_Error|> {e}. {usage}", None, None

def call_tool(category: str ,tool_cmd: str, images: List[Image.Image]) -> Any:
    """
    Call different tools
    Returns:
        Tuple[bytes, str]:
            image as PNG bytes,
            message -> As text describing the crop operation
            offset -> For calculating the coordinates relative to the original image
            id_num -> The index of the image in the images list
    """
    if category == "crop":
        return call_crop(tool_cmd, images)
    elif category == "find_color":
        return call_color(tool_cmd, images)
    elif category == "extract":
        return call_extract(tool_cmd, images)
    else:
        return None, f"<|Tool_Error|>, Unsupported tool category: {category}", None, None

def parse(text: str, strip: bool = True) -> Any:
    """
    Parse the given XML string and return an object with attributes corresponding
    to all allowed tags in the schema.
    
    For each field defined:
        - If it is a simple field (e.g. 'reasoning'), the output object will have
        an attribute 'reasoning' set to the text content (or None if missing).
        - If it is defined with alternatives (e.g. ("code", "answer")), the output
        object will have attributes for *each* allowed tag name. For example,
        if the schema is ['reasoning', ('code', 'answer')], then both
        `result.code` and `result.answer` are always accessible. If a tag is not
        found in the XML, its corresponding attribute is set to None.
    """
    results: Dict[str, Optional[str]] = {}
    for alt in ["crop", "answer", "find_color", "extract"]:
        # Regex pattern to capture the content between the tags.
        pattern = rf"<{alt}>\s*(.*?)\s*</{alt}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            results[alt] = match.group(1).strip() if strip else match.group(1)
        else:
            results[alt] = None
    return SimpleNamespace(**results) # Convert dict to object for attribute access
    
# --------- env response ---------

def env_response(messages: List[dict[str, Union[str, List[dict]]]], 
                    images: List[Image.Image] , images_offset: List[tuple], **kwargs: Any) -> Dict[str, Any]:
    try:
        # Determine which tool to call based on the last assistant message
        parsed = parse(messages[-1]["content"][0]["text"])
        if hasattr(parsed, 'crop') and parsed.crop is not None:
            category = 'crop'
            tool_cmd = parsed.crop
        elif hasattr(parsed, 'find_color') and parsed.find_color is not None:
            category = 'find_color'
            tool_cmd = parsed.find_color
        elif hasattr(parsed, 'extract') and parsed.extract is not None:
            category = 'extract'
            tool_cmd = parsed.extract
        else: # No valid tool command found
            tool_feedback = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<|Format_Error|>: No valid tool command found in the last message."}
                ]
            }
            messages.append(tool_feedback)
            return
        crop_bytes, info_message, off_set, id_num = call_tool(category, tool_cmd, images)
        if crop_bytes:
            crop_b64 = base64.b64encode(crop_bytes).decode('utf-8')
            cropped_img = Image.open(io.BytesIO(crop_bytes)).convert("RGB")
            images.append(cropped_img) # add to images list for further processing
            
            # Calculate and add offset
            dx = off_set[0]
            dy = off_set[1]
            x_off_set = images_offset[id_num][0] + dx
            y_off_set = images_offset[id_num][1] + dy
            off_set = (x_off_set, y_off_set)
            images_offset.append(off_set)
            
            info_message = (
                f"[Image_{len(images)-1} is displayed above, offset: {off_set}]\n{info_message}\n"
            )
            multimodal_message = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{crop_b64}"}
                    },
                    {
                        "type": "text",
                        "text": info_message
                    }
                ]
            tool_feedback =  {"role": "user", "content": multimodal_message}
            messages.append(tool_feedback)
            return
            
        else:
            tool_feedback = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Error: {info_message}"}
                ]
            }
            messages.append(tool_feedback)
            return

    except Exception as e:
        tool_feedback = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Unexpected error in env_response: {e}"}
            ]
        }
        messages.append(tool_feedback)
        return
    
# -------- Util --------
def _get_step_count(messages: List[Dict[str, str]]) -> int:
    """Count the number of tool uses in the message history, excluding few-shot examples."""
    step_count = 0
    
    # Skip messages that are part of few-shot examples
    # We need to determine where the actual conversation starts
    # System message + few-shot examples + user query = start of actual conversation
    conversation_start = 1  # Start after system message
    
    # Only count tool uses from the actual conversation
    for message in messages[conversation_start:]:
        if message.get("role") == "assistant":
            step_count += 1
    return step_count

def is_completed(messages: List[dict[str, Union[str, List[dict]]]], **kwargs: Any) -> bool:
    try:
        # Check if we've hit max steps by counting tool uses in the message history
        step_count = _get_step_count(messages)
        if step_count > 5:
            return True
        
        parsed = parse(messages[-1]["content"][0]["text"])
        # Check if we got a valid answer field (not just None from failed parsing)
        return hasattr(parsed, 'answer') and parsed.answer is not None
    except Exception:
        return False

def _prepare_multimodal_chat_template(prompt: str, image:Image.Image) -> List[dict]:
    '''
    Prepare the multimodal chat template for vLLM inference.
    This function takes a list of prompts and a list of images, and returns a list of dictionaries
    that can be used as input to the vLLM model.
    '''
    multimodal_inputs = []
    # for prompt, image in zip(prompts, images):
        # initial_prompts = CROP_SYSTEM_PROMPT.format(
        # tool_descriptions=CROP_TOOL_DESCRIPTION+EXTRACT_TOOL_DESCRIPTION+FIND_COLOR_TOOL_DESCRIPTION,
        # tool_example=CROP_TOOL_EXAMPLE+ EXTRACT_TOOL_EXAMPLE + FIND_COLOR_TOOL_EXAMPLE
        # ) + f"\nNow Let's work on the real case:\n[Image_0 is displayed below]\nplease help me to identify the coordinate of the following element: \n{prompt}"
    initial_prompts = TOOL_PROMPT + f"please help me to identify the coordinate of the following element: \n{prompt}"
    if image is not None:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        initial_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": initial_prompts},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ]
    else:
        initial_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": initial_prompts},
                    ]
                }
            ]
    multimodal_inputs.append(initial_message)
    return multimodal_inputs

def get_last_answer(trajectory: List[Dict[str, str]]):
    """Extract the last answer from a trajectory."""
    for msg in reversed(trajectory):
        if msg['role'] == 'assistant':
            parsed = parse(msg['content'][0]['text'])
            if hasattr(parsed, 'answer') and parsed.answer is not None:
                return parsed.answer
    return None

# -------- generate & step --------

def step(states: List[Dict[str, Any]], llm: LLM, sampling_params: SamplingParams) -> List[Dict[str, Any]]:
    
    live_indices = [i for i, s in enumerate(states) if not s["completed"]]
    messages_to_step = [states[i]["messages"] for i in live_indices]
    llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=True) # type: ignore

    def update_state(j, llm_response):
        """
        Update three things in the state:
        1. messages: append the assistant response
        2. all_prompts: include the prompt token ids and the assistant response text from all turns
        3. images: append the image from the tools if it exists
        """
        # sleep for 0-1 seconds to avoid rate limiting
        time.sleep(1 * random.random())
        state = deepcopy(states[j])
        
        # Avoid image padding in the response
        # OtherWise there is some chance that the ERROR: 
        # num_image_tokens = image_grid_thw[index].prod() // merge_length IndexError: will happen
        clean_text = llm_response.outputs[0].text.replace('<|image_pad|>', '')
        state["messages"].append({"role": "assistant", "content": [{'type': 'text', 'text': clean_text}]})
    
        # Finish or execute the tools
        current_id_length = len(llm_response.prompt_token_ids) + len(llm_response.outputs[0].token_ids)
        if is_completed(state["messages"]) or current_id_length > sampling_params.max_tokens - 1:
            state["completed"] = True
            state['all_prompts'] = llm_response.prompt + clean_text + '<|im_end|>' # update all_prompts
        else:
            env_response(state["messages"], state["images"], state["images_offset"]) # call tools and add environment response

        return j, state

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda args: update_state(*args),
            [(j, llm_responses[i]) for i, j in enumerate(live_indices)]
        ))

    for j, state in results:
        states[j] = state

    return states

def generate(prompts: List[List[Dict[str, Any]]],
                llm: LLM):
    """
    Generate responses for multiple prompts using the LLM and sampling parameters.
    Args:
        prompts: List of prompts, each a list of message dicts
        llm: LLM instance for generating responses
        sampling_params: Sampling parameters for the generation
        **kwargs: Additional arguments (not used here)
    Returns:
        A dictionary containing:
        - all_prompts: List of all prompts generated by the LLM
        - images: List of images generated by the tools, if any
    """
    custom_sp = SamplingParams(
        n=1,
        max_tokens=19263,
        temperature=0,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0
    )

    def bs64_image(messages) -> str:
        # image_entry = next(item for item in messages[0]["content"] if item["type"] == "image_url")
        image_entry = next(item for item in messages[0]["content"] if item["type"] == "image_url")
        data_uri = image_entry["image_url"]["url"]
        bs64_str = data_uri.split(",", 1)[1]
        image_bytes = base64.b64decode(bs64_str)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return img
    
    # initialize state variables
    all_completed = False
    states = []
    for m in prompts:
        # print(f'm:{m}')
        img = bs64_image(m[0])
        state = {
            "messages": m[0],
            "all_prompts": "",
            "completed": False,
            "images": [img],
            "images_offset": [(0,0)],  # Store additional image info if needed
        }
        states.append(state)
    
    # main loop
    while not all_completed:
        states = step(states, llm, custom_sp)
        all_completed = all(state["completed"] for state in states)
        
    all_prompts = [s["all_prompts"] for s in states]
    all_images = [s["images"] for s in states] # list[list[Image.Image]] 
    all_messages = [s["messages"] for s in states]
    all_images_offset = [s["images_offset"] for s in states] # list[list[Tuple[int, int]]] 
    
    output = {
        "all_prompts": all_prompts,
        "images": all_images,
        "all_messages": all_messages,
        "images_offset": all_images_offset,
    }
    return output

# -------- Post process --------

def extract_coordination(
    completion: List[Any],
    images_offset,
):
    """
    Reward function that checks if the predicted point lies within the ground-truth bounding box.
    """
    
    # for completion, images_offset in zip(completions, all_images_offset):
    #     print(f"completion:{completion}")
    raw = str(get_last_answer(completion)).strip()
    raw = f'<answer>{raw}</answer>'
    
    try:
        pattern = re.compile(
            r"""
                <answer>
                \s*\(\s*
                Image_(\d+)
                \s*,\s*
                \(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)  # ②③ x, y
                \s*\)\s*
                </answer>
            """,
            re.VERBOSE
        )
        m = pattern.search(raw)
        if m:
            id_num, x, y = m.groups()
            id_num = int(id_num)
            x, y   = int(x), int(y)
            x = x + images_offset[id_num][0]
            y = y + images_offset[id_num][1]
            coordination = (x, y)
        else:
            coordination = 'none'
            
    except Exception:
        coordination = 'none'
    
    return coordination

from typing import Union, Tuple, List
import math

Number = Union[int, float]
Point = Tuple[Number, Number]
PolyLike = Union[List[Number], List[Point]]

def point_in_bbox(pt: Union[str, Point, None], bbox_xywh_or_poly: PolyLike) -> bool:
    if pt in ("none", None):
        return False
    try:
        x, y = float(pt[0]), float(pt[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
    except Exception:
        return False

    arr = bbox_xywh_or_poly
    if not isinstance(arr, (list, tuple)) or len(arr) < 4:
        return False

    if len(arr) == 4 and all(isinstance(v, (int, float)) for v in arr):
        bx, by, bw, bh = map(float, arr)
        if bw < 0 or bh < 0:
            x1, y1, x2, y2 = bx, by, bw, bh
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            return (xmin <= x <= xmax) and (ymin <= y <= ymax)
        return (bx <= x <= bx + bw) and (by <= y <= by + bh)

    poly: List[Tuple[float, float]]
    if all(isinstance(v, (int, float)) for v in arr):
        if len(arr) % 2 != 0:
            return False
        it = iter(arr)
        poly = [(float(a), float(b)) for a, b in zip(it, it)]
    else:
        try:
            poly = [(float(p[0]), float(p[1])) for p in arr]
        except Exception:
            return False

    if len(poly) < 3:
        return False

    def point_on_segment(px, py, ax, ay, bx, by, eps=1e-9):
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if abs(cross) > eps:
            return False
        dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
        if dot < -eps:
            return False
        sq_len = (bx - ax) ** 2 + (by - ay) ** 2
        if dot - sq_len > eps:
            return False
        return True

    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if point_on_segment(x, y, x1, y1, x2, y2):
            return True

    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        if (y1 > y) != (y2 > y):
            x_inter = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
            if x_inter == x:
                return True
            if x_inter > x:
                inside = not inside

    return inside


def _try_load_font() -> Optional[ImageFont.ImageFont]:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception:
            return None

from typing import Union, Tuple, List, Sequence
from pathlib import Path
from PIL import Image, ImageDraw

Number = Union[int, float]
Point = Tuple[Number, Number]

def _normalize_poly(bbox_xywh_or_poly: Sequence) -> List[Tuple[float, float]]:
    if not isinstance(bbox_xywh_or_poly, (list, tuple)) or len(bbox_xywh_or_poly) < 4:
        raise ValueError("bbox_xywh_or_poly shape is invalid.")
    arr = bbox_xywh_or_poly

    if len(arr) == 4 and all(isinstance(v, (int, float)) for v in arr):
        x1, y1, a, b = map(float, arr)
        if a >= 0 and b >= 0:
            x2, y2 = x1 + a, y1 + b
        else:
            x2, y2 = a, b
        xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
        ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
        return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

    if all(isinstance(v, (int, float)) for v in arr):
        if len(arr) % 2 != 0:
            raise ValueError("Must have even number of coordinates for polygon points.")
        it = iter(arr)
        return [(float(x), float(y)) for x, y in zip(it, it)]

    try:
        poly = [(float(p[0]), float(p[1])) for p in arr]
    except Exception as e:
        raise ValueError(f"cannot parse polygon points: {e}")
    if len(poly) < 3:
        raise ValueError("At least 3 points are required to form a polygon.")
    return poly

def _poly_bbox(poly: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, ymin, xmax, ymax

def draw_viz(
    image: Image.Image,
    pred: Union[str, Tuple[float, float], List[float], None],
    bbox_xywh: List[float],
    save_path: Union[str, Path],
    show_text: bool = True,
):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        poly = _normalize_poly(bbox_xywh)
    except Exception as e:
        print(f"[draw_viz] failed to normalize bbox: {bbox_xywh}. Error: {e}")
        poly = None

    if poly:
        pts = poly + [poly[0]]
        draw.line(pts, fill="green", width=3)
        bx, by, bx2, by2 = _poly_bbox(poly)
    else:
        bx = by = 0.0
        bx2 = by2 = 0.0

    if pred not in ("none", None):
        try:
            px = float(pred[0])
            py = float(pred[1])
            r = 5
            draw.ellipse([px - r, py - r, px + r, py + r], fill="red", outline="red")
        except Exception as e:
            print(f"[draw_viz] failed to draw pred: {pred}. Error: {e}")

    if show_text:
        txt = f"pred={pred}"
        tx = bx
        ty = max(0.0, by - 18.0)
        try:
            font = _try_load_font()
        except Exception:
            font = None
        try:
            bg_pad = 2
            approx_w = len(txt) * 6 + 2 * bg_pad
            approx_h = 12 + 2 * bg_pad
            draw.rectangle([tx - bg_pad, ty - bg_pad, tx + approx_w, ty + approx_h], fill=(0, 0, 0, 100))
        except Exception:
            pass
        try:
            draw.text((tx, ty), txt, font=font, fill="yellow")
        except Exception:
            pass

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(p))
    img.close()


def build_llm(model_name: str):
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        max_model_len=30000,
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
    )
    return llm


def _normalize_env_result(env_result):
    per_sample = []
    if isinstance(env_result, dict) and "all_messages" in env_result:
        all_msgs = env_result["all_messages"]
        offsets = env_result.get("images_offset")
        if isinstance(offsets, list):
            for i in range(len(all_msgs)):
                per_sample.append((all_msgs[i], offsets[i] if i < len(offsets) else None))
        else:
            for i in range(len(all_msgs)):
                per_sample.append((all_msgs[i], offsets))
    elif isinstance(env_result, list):
        for er in env_result:
            per_sample.append((er.get("all_messages"), er.get("images_offset")))
    else:
        raise ValueError("Unexpected env_result format from generate().")
    return per_sample


def main_batch(prompts: List[str], images: List[Image.Image], llm) -> List[Union[str, Tuple[float, float]]]:
    assert len(prompts) == len(images), "Prompts and images must have the same length."

    batched_inputs = []
    for p, img in zip(prompts, images):
        batched_inputs.append(_prepare_multimodal_chat_template(p, img))

    env_result = generate(prompts=batched_inputs, llm=llm)
    per_sample = _normalize_env_result(env_result)

    coords = []
    for (msgs_i, img_offset_i) in per_sample:
        coord = extract_coordination(msgs_i, img_offset_i)
        coords.append(coord)
    return coords


import json
import statistics
from pathlib import Path
from typing import Optional, List, Tuple, Union

# ======= Per-category helpers =======
_CLS_JSON_DEFAULT = "OSWorld-G/benchmark/classification_result.json"
_DISPLAY_ORDER = [
    ("text_matching", "Text Matching"),
    ("element_recognition", "Element Recognition"),
    ("layout_understanding", "Layout Understanding"),
    ("fine_grained_manipulation", "Fine-grained Manipulation"),
    ("refusal", "Refusal"),
]

def _load_id2cat(
    classification_json: Optional[str] = None,
    resolve: str = "all",   # "priority" | "first" | "last" | "all"
    verbose: bool = True,
) -> dict:
    """
    Load classification_result.json and build an ID→category mapping.

    - If an ID appears in multiple categories, resolve the conflict by:
        * "priority" (default): choose by predefined priority order
        * "first": keep the first seen category
        * "last":  keep the last seen category
        * "all":   keep a tuple of all categories (caller must handle tuple)
    - Returns: {id: category} (or {id: tuple(categories)} if resolve="all")

    Expected JSON structure:
      {"classified": {
          "text_matching": [...],
          "element_recognition": [...],
          "layout_understanding": [...],
          "fine_grained_manipulation": [...],
          "refusal": [...]
      }}
    """
    priority_list = [
        "text_matching",
        "element_recognition",
        "layout_understanding",
        "fine_grained_manipulation",
        "refusal",
    ]
    prio = {k: i for i, k in enumerate(priority_list)}

    path = classification_json or _CLS_JSON_DEFAULT
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    cls = obj.get("classified", {})
    id2cats = {}

    for cat_key, items in cls.items():
        if not items:
            continue
        for it in items:
            sid = it.get("id")
            if not sid:
                continue
            id2cats.setdefault(sid, set()).add(cat_key)

    id2cat_final: dict = {}
    for sid, cats in id2cats.items():
        if not cats:
            continue

        if resolve == "all":
            id2cat_final[sid] = tuple(sorted(cats, key=lambda k: prio.get(k, 1_000_000)))
            continue

        if len(cats) == 1:
            chosen = next(iter(cats))
        else:
            if resolve == "first":
                chosen = sorted(cats, key=lambda k: prio.get(k, 1_000_000))[0]
            elif resolve == "last":
                chosen = sorted(cats, key=lambda k: prio.get(k, -1), reverse=True)[0]
            else:  # "priority"
                chosen = sorted(cats, key=lambda k: prio.get(k, 1_000_000))[0]

            if verbose:
                others = [c for c in cats if c != chosen]

        id2cat_final[sid] = chosen

    return id2cat_final


def _init_cat_stats() -> dict:
    # {cat_key: {"total":0, "hits":0, "none":0}}
    return {k: {"total": 0, "hits": 0, "none": 0} for k, _ in _DISPLAY_ORDER}

def _update_cat_stats(cat_stats: dict, cat_key, hit: bool, is_none: bool):
    """
    Update per-category counters.
    - cat_key can be a single category (str) or multi-label (list/tuple/set of str).
    - Unknown categories will be initialized on the fly.
    """
    if not cat_key:
        return

    # Normalize to iterable of category keys
    if isinstance(cat_key, (list, tuple, set)):
        keys = [k for k in cat_key if isinstance(k, str)]
        if not keys:
            return
    elif isinstance(cat_key, str):
        keys = [cat_key]
    else:
        return

    for k in keys:
        if k not in cat_stats:
            cat_stats[k] = {"total": 0, "hits": 0, "none": 0}
        cat_stats[k]["total"] += 1
        if hit:
            cat_stats[k]["hits"] += 1
        if is_none:
            cat_stats[k]["none"] += 1


def _format_cat_progress(cat_stats: dict) -> str:
    """
    Return a one-line string with per-category accuracy and overall (macro) average.
    Only categories with total>0 are used for averaging.
    """
    parts = []
    accs = []
    for key, disp in _DISPLAY_ORDER:
        st = cat_stats.get(key, {"total": 0, "hits": 0})
        tot = st.get("total", 0)
        hit = st.get("hits", 0)
        acc = (hit / tot) if tot else 0.0
        if tot:
            accs.append(acc)
        parts.append(f"{disp}: {acc:.4f} ({hit}/{tot})")
    avg = statistics.mean(accs) if accs else 0.0
    parts.append(f"Average: {avg:.4f}")
    return " | ".join(parts)

# ======= Main evaluation with realtime per-category stats =======
def evaluate_dataset_in_batches(
    dataset_json: Path,
    images_dir_opt: Optional[Path],
    output_dir: Path,
    model_name: str,
    save_viz: bool = False,
    batch_size: int = 16,
    classification_json: Optional[str] = None,  # NEW: path to classification_result.json
):
    with open(dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_dir = images_dir_opt if images_dir_opt else dataset_json.parent / "images"

    try:
        id2cat = _load_id2cat(classification_json)
    except Exception as e:
        print(f"[WARN] {e}")
        id2cat = {}
    cat_stats = _init_cat_stats()

    total = hits = misses = none_cnt = 0
    entries = []
    for i, item in enumerate(data):
        try:
            image_file = images_dir / item["image_path"]
            prompt = item["instruction"]
            bbox_xywh = item["box_coordinates"]
            image = Image.open(str(image_file)).convert("RGB")
            entries.append({
                "idx": i,
                "id": item.get("id"),
                "prompt": prompt,
                "image": image,
                "bbox": bbox_xywh,
                "image_path": item["image_path"],
            })
        except Exception as e:
            total += 1
            misses += 1
            print(f"[WARN] item {i} ({item.get('id')}) Fail to load image: {e}")

    if not entries:
        print("No valid entries to evaluate.")
        return

    llm = build_llm(model_name)

    for start in range(0, len(entries), batch_size):
        batch = entries[start:start + batch_size]
        batch_prompts = [x["prompt"] for x in batch]
        batch_images  = [x["image"]  for x in batch]

        batch_results = main_batch(batch_prompts, batch_images, llm)

        for x, result in zip(batch, batch_results):
            total += 1
            is_none = (result == "none")
            if is_none:
                none_cnt += 1

            hit = False
            if not is_none:
                hit = point_in_bbox(result, x["bbox"])

            if hit:
                hits += 1
            else:
                misses += 1

            # per-category stats
            cat_key = id2cat.get(x["id"])
            _update_cat_stats(cat_stats, cat_key, hit=hit, is_none=is_none)

            if save_viz:
                save_name = f"{Path(x['image_path']).stem}__{'hit' if hit else 'miss'}.png"
                draw_viz(x["image"], result, x["bbox"], output_dir / save_name)

        for img in batch_images:
            try: img.close()
            except: pass

        done = min(start + batch_size, len(entries))
        if (done % 1 == 0) or (done == len(entries)):
            acc = hits / total if total else 0.0
            cat_line = _format_cat_progress(cat_stats)
            print(f"[{done}/{len(entries)}] hits={hits}, misses={misses}, none={none_cnt}, acc={acc:.4f}")
            print(f"  ↳ Per-category: {cat_line}")

    acc = hits / total if total else 0.0
    print("=" * 60)
    print(f"Done. total={total}, hits={hits}, misses={misses}, none={none_cnt}, acc={acc:.4f}")
    print("Per-category final:", _format_cat_progress(cat_stats))
    print("=" * 60)

def evaluate_single_sample(
    prompt: str,
    image_path: Path,
    model_name: str,
):
    image = Image.open(str(image_path)).convert("RGB")
    llm = build_llm(model_name)
    result = main_batch([prompt], [image], llm)[0]
    print(f"The coordination is: {result}")

    if result != "none":
        try:
            x, y = float(result[0]), float(result[1])
            draw = ImageDraw.Draw(image)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red", outline="red")
            Path("tests").mkdir(parents=True, exist_ok=True)
            image.save("tests/output_image.png")
            print(f"Image saved with the point drawn at {result}.")
        except Exception:
            print("Failed to draw the point on the image.")
    else:
        print("No valid coordination found.")

    image.close()


# =========================
# CLI
# =========================

def build_argparser():
    import argparse
    parser = argparse.ArgumentParser(description="Batch & Single evaluation for GUI pointing.")

    parser.add_argument("--model", required=True, type=str, help="Model name or path")
    parser.add_argument("--save_viz", action="store_true", help="Save visualization images (only for batch eval)")
    parser.add_argument("--output_dir", type=str, default="viz_out", help="Directory to save visualization images")

    parser.add_argument("--dataset_json", type=str, help="Dataset JSON file for batch evaluation")
    parser.add_argument("--images_dir", type=str, help="Directory containing images (if not specified, assumes 'images' subdir next to dataset_json)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")

    parser.add_argument("--prompt", type=str, help="Single sample prompt")
    parser.add_argument("--image_path", type=str, help="Single sample image path")

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.dataset_json:
        ds_path = Path(args.dataset_json)
        images_dir = Path(args.images_dir) if args.images_dir else None
        evaluate_dataset_in_batches(
            dataset_json=ds_path,
            images_dir_opt=images_dir,
            output_dir=output_dir,
            model_name=args.model,
            save_viz=bool(args.save_viz),
            batch_size=int(args.batch_size),
        )
    else:
        if not args.prompt or not args.image_path:
            raise SystemExit("Single sample evaluation requires --prompt and --image_path.")
        evaluate_single_sample(
            prompt=args.prompt,
            image_path=Path(args.image_path),
            model_name=args.model,
        )

if __name__ == "__main__":
    main()

