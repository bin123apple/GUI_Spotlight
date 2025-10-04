import io
import re
import ast
import base64
import inspect

from PIL import Image
from transformers import PreTrainedModel
from typing import Callable, Dict, Union, Any, List
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
from spotlight.tools import crop, extract, find_color
from spotlight.tools_envs.multiturn_env import MultiTurnEnv
from spotlight.parser import XMLParser
from spotlight.reward.tool_rubric import ToolRubric

def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()
    
    # Extract examples if present
    examples = []
    return_description = ""
    for part in doc_parts:
        if part.startswith("Examples:"):
            examples = [line.strip() for line in part.split("\n")[1:] if line.strip()]
        elif part.startswith("Returns:"):
            return_description = part.split("\n")[1].strip()

    return_type = str(sig.return_annotation.__name__ if sig.return_annotation != inspect.Parameter.empty else "any")

    # print(f"return_description: {return_description} ({return_type})")
    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name)+1:].strip()
        
        args[name] = {
            "type": str(param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any"),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default
    
    return {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": return_description + f" ({return_type})",
        "examples": examples
    }

def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]
        
        desc.append("\nArguments:")
        for arg_name, arg_info in schema['args'].items():
            default = f" (default: {arg_info['default']})" if 'default' in arg_info else ""
            desc.append(f"  - {arg_name}: {arg_info['description']}{default}")
        
        if schema['examples']:
            desc.append("\nExamples:")
            for example in schema['examples']:
                desc.append(f"  {example}")
        
        if schema['returns']:
            desc.append(f"\nReturns: {schema['returns']}")
        
        descriptions.append("\n".join(desc))
    
    return "\n\n".join(descriptions)

class ToolEnv(MultiTurnEnv):
    def __init__(self,
                 tools: List[Callable] = [],
                 few_shot: List[Dict[str, str]] = [],
                 llm_fields: List[str | tuple[str, str]] = [("crop", "answer", "find_color", "extract")],
                 env_fields: List[str | tuple[str, str]] = ["result"],
                 sampling_args={
                     "stop": ["</crop>", "</answer>", "</extract>", "</find_color>"],
                     "include_stop_str_in_output": True
                 },
                 mask_env_response: bool = True,
                 max_steps: int = 10, **kwargs):
        # Infer schemas from tool functions
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}
        
        super().__init__(
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            max_steps=max_steps,
            sampling_args=sampling_args,
            **kwargs
        )
        self.max_steps = max_steps
        self.llm_parser = XMLParser(fields=llm_fields)
        self.env_parser = XMLParser(fields=env_fields)
        self.rubric = ToolRubric(tools=tools, parser=self.llm_parser, env_parser=self.env_parser)

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()

    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of tool uses in the message history, excluding few-shot examples."""
        step_count = 0
        
        # Skip messages that are part of few-shot examples
        # We need to determine where the actual conversation starts
        # System message + few-shot examples + user query = start of actual conversation
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)
        
        # Only count tool uses from the actual conversation
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                step_count += 1
        return step_count
    
    def is_completed(self, messages: List[dict[str, Union[str, List[dict]]]], **kwargs: Any) -> bool:
        try:
            # Check if we've hit max steps by counting tool uses in the message history
            step_count = self._get_step_count(messages)
            if step_count >= self.max_steps:
                return True
            
            parsed = self.llm_parser.parse(messages[-1]["content"][0]["text"])
            # Check if we got a valid answer field (not just None from failed parsing)
            return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception:
            return False

    def call_crop(self, tool_cmd: str, images: List[Image.Image]):
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

    def call_scan(self, tool_cmd: str, images: List[Image.Image]) -> bytes:
        raise NotImplementedError("call_scan is not implemented yet.")
        
    def call_extract(self, tool_cmd: str, images: List[Image.Image]) -> bytes:
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

    def call_color(
        self,
        tool_cmd: str,
        images: List[Image.Image]
    ):
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

    def call_tool(self, category: str ,tool_cmd: str, images: List[Image.Image]) -> Any:
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
            return self.call_crop(tool_cmd, images)
        elif category == "find_color":
            return self.call_color(tool_cmd, images)
        elif category == "extract":
            return self.call_extract(tool_cmd, images)
        else:
            return None, f"<|Tool_Error|>, Unsupported tool category: {category}", None, None


    def env_response(self, messages: List[dict[str, Union[str, List[dict]]]], 
                     images: List[Image.Image] , images_offset: List[tuple], **kwargs: Any) -> Dict[str, Any]:
        try:
            # Find the target element from the first user message
            phrase = "please help me to identify the coordinate of the following element:"
            first_user_msg = next(msg for msg in messages if msg.get("role") == "user")
            content_list = first_user_msg.get("content", [])
            first_user_content_with_text = next(
                (item for item in content_list if item.get("type") == "text"),
                None
            )
            first_user_text = first_user_content_with_text.get("text", "") if first_user_content_with_text else ""
            _, _, tail = first_user_text.partition(phrase)
            element = tail.strip().split('\n')[0] if tail else "target element"
            
            # Determine which tool to call based on the last assistant message
            parsed = self.llm_parser.parse(messages[-1]["content"][0]["text"])
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
            
            crop_bytes, info_message, off_set, id_num = self.call_tool(category, tool_cmd, images)
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