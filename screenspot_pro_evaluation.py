import re
import io
import ast
import json
import copy
import base64
import argparse
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from spotlight.tools_envs.tool_env import ToolEnv
from vllm import LLM, SamplingParams
from spotlight.parser import XMLParser
from datasets import Dataset, load_from_disk, load_dataset
from huggingface_hub import snapshot_download
from typing import List, Dict, Any, Tuple

SYSTEM_PROMPT = """Your goal is to accurately provide a coordinate point based on the user’s description and the initial image they supplied. 
You may use the crop, extract and find_color tool to help you analyze and hone in on the target coordinate by placing the tool call inside 
<crop>...</crop>, <extract>...</extract> and <find_color>...</find_color> tags; 
each time you call the tool, I will return the result to you. 
In the end, you must place your selected coordinate inside <answer>...</answer> tags.

The tool functions are used like this: 

{tool_descriptions}

and here is an example of their usage: 

{tool_example}

Please note:
1. You may call the tool multiple times if needed.
2. Do NOT include parameter names (e.g., image_id, top_left, bottom_right) in your tool calls.
For example, use <crop>(Image_0, (10, 20), (110, 100))</crop> instead of <crop>image_id="Image_0" top_left="(10, 20)" bottom_right="(110, 100)"></crop>.
3. In the end you must provide the coordinate in the format <answer>(Image_id, (x, y))</answer>, where the (x, y) values are relative to the image corresponding to Image_id, not to the original image.
4. For each tool call, you must provide your thought in <think>...</think> tags."""

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

EXTRACT_TOOL_EXAMPLE = '''User: [Image_0 is displayed above] Could you identify the location of the “Close” button in this interface?

Assistant: <think>I’d like to zoom in on the menu icon at the top-left corner to get a clearer view.</think>  
<extract>(Image_0, left, top)</extract>

User: [Image_1 is displayed above]  # (possible area returned)

Assistant: <think>In this cropped image, I can see the approximate position of the “Close” button—it sits near the center of the region, slightly toward the lower-right. it’s at (45, 60).</think>  
<answer>(Image_1, (45, 60))</answer>'''

EXTRACT_TOOL_PROMPT = f"""You should use the **extract** tool to help you analyze the image and find the target coordinate:
This tool allows you to extract one quarter of the image based on the specified horizontal and vertical positions (left, center, right for x-axis; top, center, bottom for y-axis).
Before each tool call, please enclose your reasoning within <think>...</think> tags.
In the end, you should return your final answer using the <answer>...</answer> tag.
In the <answer> tag, you should return the image and the coordinate of the target object in the format (Image_X, (x, y)), where Image_X is the image containing the target object and (x, y) is the coordinate of the target object.
Here is an example of how to find the final target coordinate:
{EXTRACT_TOOL_EXAMPLE}
Now, let's work on the real task:
[Image_0 is displayed below]"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Bin12345/qwen2_5vl_ui-tars_stage4_ckpt_50", help="Path to the pretrained model")
    parser.add_argument('--dataset_name', type=str, default="likaixin/ScreenSpot-Pro", help="Dataset path")
    return parser.parse_args()

PLACEHOLDER = "<IMAGE>"
B64_PATTERN = re.compile(r"^data:image\/\w+;base64,(.+)", re.I)

def sanitize_dialogs(dialogs: List[List[dict]], placeholder: str = PLACEHOLDER):
    safe = copy.deepcopy(dialogs)
    for dialog in safe:
        for msg in dialog:
            content = msg.get("content")
            if isinstance(content, list):
                for piece in content:
                    if piece.get("type") == "image_url":
                        piece["image_url"] = placeholder
    return safe

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

def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')

def extract_point_answer(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    point_pattern = r'(\d+\.?\d*(?:\s*[,;\s]\s*|\s+)\d+\.?\d*)'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        point_match = re.search(point_pattern, content_answer, re.DOTALL)
        if point_match:
            point_str = point_match.group(1)
            point = [float(x) for x in re.findall(r'\d+\.?\d*', point_str)]
            if len(point) >= 2:
                point = point[:2]
            else:
                point = [0, 0]
            return point
    return [0, 0]

def extract_coordinates(result: list[str]):
    text = result[0].strip()

    # if <answer> tag is present, extract the content inside it
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    content = answer_match.group(1) if answer_match else text

    # 1) try to match Image_id + coordinates (x, y) format
    img_coord_match = re.search(
        r'Image_(\d+)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
        content
    )
    if img_coord_match:
        img_id = int(img_coord_match.group(1))
        x = int(img_coord_match.group(2))
        y = int(img_coord_match.group(3))
        return img_id, x, y

    # 2) if no Image_id, try to match coordinates (x, y) format
    point_match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', content)
    if point_match:
        x, y = map(int, point_match.groups())
        return None, x, y

    # 3) if no coordinates, try to match bounding box format
    box_match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', content)
    if box_match:
        x1, y1, x2, y2 = map(int, box_match.groups())
        return None, (x1 + x2)//2, (y1 + y2)//2

    return None, None, None

def load_processed_dataset(data_path: str) -> Dataset:
    """
    Dataset format:
    {
        "image": List[bytes],
        "width": List[int],
        "height": List[int],
        "question": List[str],
        "answer": List[str]
    }
    """
    print(f"Loading processed data from {data_path}...")
    dataset = load_from_disk(data_path)
    print(f"Finish loading the dataset: {len(dataset)}")
    return dataset

def _prepare_multimodal_chat_template(prompts: List[str], images: List[Image.Image]) -> List[dict]:
    '''
    Prepare the multimodal chat template for vLLM inference.
    This function takes a list of prompts and a list of images, and returns a list of dictionaries
    that can be used as input to the vLLM model.
    '''
    multimodal_inputs = []
    for prompt, image in zip(prompts, images):
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

def vg_reward_func(
    parser: XMLParser,
    completions: List[Any],
    answer: List[Tuple[int, int, int, int]],
    all_images, 
    all_images_offset,
) -> List[float | None]:
    """
    Reward function that checks if the predicted point lies within the ground-truth bounding box.
    """
    rewards: List[float | None] = []
    
    for completion, box, images, images_offset in zip(completions, answer, all_images, all_images_offset):
        raw = str(get_last_answer(parser,completion)).strip()
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
            
            if isinstance(box, str):
                try:
                    box = tuple(ast.literal_eval(box))
                except Exception:
                    nums2 = re.findall(r"-?\d+", box)
                    box = tuple(map(int, nums2))
            x1, y1, x2, y2 = box
            
            reward = 1.0 if (x1 <= x <= x2 and y1 <= y <= y2) else 0.0

        except Exception:
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards

def get_last_answer(parser, trajectory: List[Dict[str, str]]) -> str | None:
    """Extract the last answer from a trajectory."""
    for msg in reversed(trajectory):
        if msg['role'] == 'assistant':
            if parser is None:
                raise ValueError("Parser is not set")
            parsed = parser.parse(msg['content'][0]['text'])
            if hasattr(parsed, 'answer') and parsed.answer is not None:
                return parsed.answer
    return None

def parse_crop_bbox_from_text(text: str):
    m_img = re.search(
        r"\[Image_(\d+)[^\]]*offset[:：]?\s*\(\s*(\d+),\s*(\d+)\s*\)",
        text
    )
    if not m_img:
        return None
    dx, dy = int(m_img.group(2)), int(m_img.group(3))
    
    # 4. extract bounding box of the cropped region
    m_size = re.search(r"Cropped a region of size\s*(\d+)×(\d+)", text)
    if not m_size:
        raise ValueError("Cannot parse crop size or coords.")
    w, h = map(int, m_size.groups())
    x1, y1, x2, y2 = 0, 0, w, h
    real_tl = (x1 + dx, y1 + dy)
    real_br = (x2 + dx, y2 + dy)
    return (real_tl[0], real_tl[1], real_br[0], real_br[1]) if m_size else None

class OSS_LLM:
    def __init__(self, args):
        self.args = args
        self.model = args.model_name
        self.tokenizer = args.model_name
        self.oss_llm = None
        self.oss_llm_init()
    
    def oss_llm_init(self):
        if self.oss_llm is None:
            self.oss_llm = LLM(
                model=self.model,
                tokenizer=self.model,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.95,
                enforce_eager=True,
                max_model_len=30000,
                disable_custom_all_reduce=True,
                enable_prefix_caching=False,
                trust_remote_code=True,
            )
            
    def oss_llm_completion(self, messages, stop=None):
        sampling_params = SamplingParams(
                    n=1,
                    max_tokens=19263,
                    temperature=0,
                    top_p=1.0,
                    frequency_penalty=0,
                    presence_penalty=0
                )  
        sampling_params.stop = stop
        request_output = self.oss_llm.chat(messages, sampling_params)
        return request_output

    def _ask_llm(self, image_bytes: bytes, text: str) -> tuple[int,int]:
        b64: str = encode_image(image_bytes)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": text},
                ]
            }
        ]
        result = self.oss_llm_completion(messages)
        return result 

def save_case_analysis(batch_num, case_num, original_img, cropped_imgs, final_click, crop_bboxs, gt_bbox, save_dir):
    case_dir = save_dir / f"batch_{batch_num}_case_{case_num}"
    case_dir.mkdir(parents=True, exist_ok=True)

    original_img.save(case_dir / "original.png")

    for index, cropped_img in enumerate(cropped_imgs):
        cropped_img.save(case_dir / f"cropped_{index}.png")

    marked_img = original_img.copy()
    draw = ImageDraw.Draw(marked_img)
    
    for crop_bbox in crop_bboxs:
        print(f'crop_bbox: {crop_bbox}')
        draw.rectangle(crop_bbox, outline="blue", width=3)
        
    print(f'gt_bbox: {gt_bbox}, final_click: {final_click}')
    draw.rectangle(gt_bbox, outline="green", width=3)
    if final_click:
        draw.ellipse((final_click[0]-5, final_click[1]-5, final_click[0]+5, final_click[1]+5), fill="red")
    marked_img.save(case_dir / "marked.png")

def _derive_application(case: Dict[str, Any]) -> str:
    app = case.get("application")
    if isinstance(app, str) and app:
        return app
    img_fn = case.get("img_filename", "")
    cls = img_fn.split("/")[0] if "/" in img_fn else img_fn
    if not cls:
        return "unknown"
    return cls.rsplit("_", 1)[0] if "_" in cls else cls


def _print_per_app(prefix: str, total_map: Dict[str, int], correct_map: Dict[str, int]) -> None:
    if not total_map:
        print(f"{prefix}: (no data)")
        return
    parts = []
    for app in sorted(total_map.keys()):
        tot = total_map.get(app, 0)
        cor = correct_map.get(app, 0)
        acc = (cor / tot) if tot else 0.0
        parts.append(f"{app}: {acc:.2%} ({cor}/{tot})")
    print(f"{prefix}: " + " | ".join(parts))

def main(multiturn_tools: bool = True):
    args = parse_args()
    dataset_name = args.dataset_name if hasattr(args, 'dataset_name') else None
    try:
        # dataset = load_processed_dataset(dataset_name)
        dataset = load_dataset(
            "json",
            data_files=f"hf://datasets/{dataset_name}/annotations/*.json",
            split="train",
        )
        repo_dir = snapshot_download(
            repo_id="likaixin/ScreenSpot-Pro",
            repo_type="dataset",
            allow_patterns=["images/*/*.png"],
            local_dir="./ScreenSpot-Pro",
            local_dir_use_symlinks=False,
        )
        print(f"The size of the dataset: {len(dataset)}")
        print(f'dataset: {dataset}')
        print(f'dataset example: {dataset[0]}')
    except Exception as e:
        print(f"Fail to load the dataset: {e}")
        return

    tool_env = ToolEnv(
        system_prompt=SYSTEM_PROMPT,
        few_shot=[],
        tools=[crop],
        max_steps=5
    )

    model_name = args.model_name if hasattr(args, 'model_name') else "model_results"
    model_name = model_name.split("/")[-1]

    results_dir = Path(model_name)
    results_dir.mkdir(parents=True, exist_ok=True)

    tester = OSS_LLM(args)

    if multiturn_tools:
        llm = tester.oss_llm
        sampling_params = SamplingParams(
            n=1,
            max_tokens=19263,
            temperature=0,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0
        )

    parser = XMLParser(fields=["reasoning", ("tool", "answer")])

    batch_size = 64
    total_correct = 0
    batch_correct_list = []
    application_total   = {}  # {app: seen_count}
    application_correct = {}  # {app: correct_count}

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = dataset.select(range(start, end))
        
        prompts = [case["instruction"] for case in batch]
        answers = [case["bbox"] for case in batch]
        image_folder_paths = [case['img_filename'] for case in batch]
        
        images = [None] * len(image_folder_paths)
        for i in range(len(image_folder_paths)):
            images[i] = f'./ScreenSpot-Pro/images/{image_folder_paths[i]}' 
        images = [Image.open(image).convert("RGB") for image in images]

        multimodal_inputs = _prepare_multimodal_chat_template(prompts, images)
        env_result = tool_env.generate(
            prompts=multimodal_inputs,
            llm=llm,
            sampling_params=sampling_params,
        )
        completions = env_result["all_messages"]
        all_images = env_result['images'] # calculate log_pb
        all_images_offset = env_result["images_offset"]
        
        rewards = vg_reward_func(
            parser = parser,
            completions  = completions,
            answer       = answers,
            all_images  = all_images,
            all_images_offset = all_images_offset
        )

        good_cnt = rewards.count(1)
        total_correct += good_cnt
        batch_correct_list.append(good_cnt)
        
        print(f"Batch {start//batch_size:4d}: kept {good_cnt}/{len(batch)}")
        
        batch_app_total: Dict[str, int] = {}
        batch_app_correct: Dict[str, int] = {}

        for idx, reward in enumerate(rewards):
            app = _derive_application(batch[idx])

            if app not in application_total:
                application_total[app] = 0
                application_correct[app] = 0
            if app not in batch_app_total:
                batch_app_total[app] = 0
                batch_app_correct[app] = 0

            application_total[app] += 1
            batch_app_total[app] += 1
            if reward == 1:
                application_correct[app] += 1
                batch_app_correct[app] += 1

        seen_so_far = (start + len(batch))
        overall_acc_so_far = total_correct / seen_so_far if seen_so_far else 0.0
        print(f"[Progress] Overall so far: {overall_acc_so_far:.2%} ({total_correct}/{seen_so_far})")
        _print_per_app("[Cumulative per-application]", application_total, application_correct)
        _print_per_app("[This-batch  per-application]", batch_app_total, batch_app_correct)
        
        # Save analysis for correct and wrong cases
        for case_type, predicate in [("correct", lambda r: r == 1),
                                    ("wrong",   lambda r: r != 1)]:
            for idx, reward in enumerate(rewards):
                if predicate(reward):
                    msgs = env_result["all_messages"][idx]
                    image_offset = env_result["images_offset"][idx]
                    
                    sanitize_message = sanitize_dialogs([msgs])[0]
                    print(f'len(msgs): {len(msgs)}')
                    print(f"sanitize_message: {json.dumps(sanitize_message, indent=2, ensure_ascii=False)}")
                    cropped_images = [
                        Image.open(io.BytesIO(base64.b64decode(item["image_url"]["url"]
                                    .split("base64,")[1]))).convert("RGB")
                        for msg in msgs[1:] if msg.get("role") == "user"
                        for item in msg.get("content", [])
                        if item.get("type") == "image_url"
                    ]

                    crop_bboxs = [
                        parse_crop_bbox_from_text(item["text"])
                        for msg in msgs[1:] if msg.get("role") == "user"
                        for item in msg.get("content", [])
                        if item.get("type") == "text"
                        if parse_crop_bbox_from_text(item["text"]) is not None
                    ]

                    raw = str(get_last_answer(parser, msgs)).strip()
                    img_id, x, y = extract_coordinates([raw])
                    if img_id:
                        dx, dy  = image_offset[img_id]
                        final_click = (x + dx, y + dy) if x is not None and y is not None else None
                    else:
                        final_click = None
                    if isinstance(final_click, str):
                        try:
                            final_click = tuple(ast.literal_eval(final_click))
                        except Exception:
                            nums = re.findall(r"-?\\d+", final_click)
                            final_click = tuple(map(int, nums))
                            
                    # Convert gt_bbox to tuple if it's a string
                    box = answers[idx]
                    if isinstance(box, str):
                        try:
                            box = tuple(ast.literal_eval(box))
                        except Exception:
                            nums2 = re.findall(r"-?\d+", box)
                            box = tuple(map(int, nums2))

                    save_case_analysis(
                        batch_num=start//batch_size,
                        case_num=f"{case_type}_{idx}",
                        original_img=images[idx],
                        cropped_imgs=cropped_images,
                        final_click=final_click,
                        crop_bboxs=crop_bboxs,
                        gt_bbox=box,
                        save_dir=results_dir
                    )
                    print(f"Saved {case_type} case {idx} of batch {start//batch_size}")
                    break

    plt.figure(figsize=(10, 6))
    plt.plot(batch_correct_list, marker='o')
    plt.title('Correct Counts per Batch')
    plt.xlabel('Batch Number')
    plt.ylabel('Correct Count')
    plt.grid(True)
    plt.savefig(results_dir / "batch_accuracy.png")
    plt.close()

    print(f"\n✅ Total correct: {total_correct} / {len(dataset)} ({total_correct / len(dataset):.2%})")

if __name__ == "__main__":
    main()
