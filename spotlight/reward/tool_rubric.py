import re
import ast
import base64
import textwrap
from io import BytesIO
from PIL import Image
from typing import List, Dict, Callable, Any, Tuple
from spotlight.parser import XMLParser
from spotlight.reward.rubric import Rubric

def parse_crop_bbox_from_text(text: str):
    """
    Extract:
      - id_num
      - top_left: tuple (x1, y1)
      - bottom_right: tuple (x2, y2)
    from  <crop>(Image_0, (10, 20), (110, 100))</crop>
    else, return  (None, None, None)。
    """
    pattern = re.compile(textwrap.dedent(r"""
        <crop>\s*
        \(\s*
        ([A-Za-z0-9_-]+)_(\d+)
        \s*,\s*
        \(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)
        \s*,\s*
        \(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)
        \s*\)\s*
        </crop>                         
    """), re.VERBOSE)

    m = pattern.search(text)
    if not m:
        return None, None, None
    
    image_name, id_str, x1, y1, x2, y2 = m.groups()
    return int(id_str), (int(x1), int(y1)), (int(x2), int(y2))

def extract_first_image(conv_list):
    for msg in conv_list:
        if msg.get("role") == "user":
            for part in msg.get("content", []):
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    prefix = "data:image/png;base64,"
                    if url.startswith(prefix):
                        b64_str = url[len(prefix):]
                        image_data = base64.b64decode(b64_str)
                        return Image.open(BytesIO(image_data))
    return None

def compute_iou(pred_bbox, gt_bbox, img_size):
    """
    pred_bbox, gt_bbox: (x1, y1, x2, y2)
    img_size: (width, height)
    """
    width, height = img_size

    def is_valid(bbox):
        x1, y1, x2, y2 = bbox
        # 1) size check
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return False
        # 2) coordinate check
        if x2 <= x1 or y2 <= y1:
            return False
        # 3) minimum size check
        if (x2 - x1) < 28 or (y2 - y1) < 28:
            return False
        return True

    # return 0.0 if pred_bbox is invalid
    if not is_valid(pred_bbox):
        return 0.0

    # IoU calculation
    xA = max(pred_bbox[0], gt_bbox[0])
    yA = max(pred_bbox[1], gt_bbox[1])
    xB = min(pred_bbox[2], gt_bbox[2])
    yB = min(pred_bbox[3], gt_bbox[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area   = (gt_bbox[2]   - gt_bbox[0]) * (gt_bbox[3]   - gt_bbox[1])
    union = pred_area + gt_area - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union

def average_crop_reward(crop_bboxs, gt_bbox, img_size, weights=None, ):
    """
    crop_bboxs: List[Tuple[int,int,int,int]]
    gt_bbox:   Tuple[int,int,int,int], ground-truth box
    weights:   Optional[List[float]]
    """
    n = len(crop_bboxs)
    if n == 0:
        return 0.0

    coverages = [compute_iou(pred, gt_bbox, img_size) for pred in crop_bboxs]

    if weights is None:
        weights = [1.0 / n] * n
    else:
        s = sum(weights)
        weights = [w / s for w in weights]

    reward = sum(w * c for w, c in zip(weights, coverages))
    return reward

class ToolRubric(Rubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["reasoning", ("tool", "answer")]),
                 env_parser: XMLParser = XMLParser(fields=["result"]),
                 tools: List[Callable] = []):
        self.parser = parser
        self.env_parser = env_parser
        self.tools = {tool.__name__: tool for tool in tools}
        self.reward_funcs = [
            self.correct_answer_reward_func,
            self.correct_crop_func,
            self.correct_extract_func,
            self.correct_find_color,
            self.parser.get_format_reward_func(),
        ]
        self.reward_weights = [
            0.3,
            0.3,
            0.03,
            0.27,
            0.1,
        ]

    def vg_reward_func(
        self,
        completions: List[Any],
        answer: List[Tuple[int, int, int, int]],
        task: List[str],
        all_images, 
        all_images_offset,
    ) -> List[float | None]:
        """
        Reward function that checks if the predicted point lies within the ground-truth bounding box.
        """
        rewards: List[float | None] = []
        
        for completion, box, t, images, images_offset in zip(completions, answer, task, 
                                                             all_images, all_images_offset):
            # parser ground-truth to tuple
            if t == "vg":
                # 1. extract the crop bbox from assistant messages
                raw = str(self.get_last_answer(completion)).strip()
                raw = f'<answer>{raw}</answer>'
                
                try:
                    # 2. parse the answer
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
                    
                    # 3. parse ground-truth box
                    if isinstance(box, str):
                        try:
                            box = tuple(ast.literal_eval(box))
                        except Exception:
                            nums2 = re.findall(r"-?\d+", box)
                            box = tuple(map(int, nums2))
                    x1, y1, x2, y2 = box
                    
                    # 4. give reward
                    reward = 1.0 if (x1 <= x <= x2 and y1 <= y <= y2) else 0.0

                except Exception:
                    reward = 0.0
            else:
                reward = None
            
            rewards.append(reward)
        
        return rewards
    
    def correct_answer_reward_func(self, completions, answer, task, 
                                   all_images, all_images_offset, **kwargs) -> List[float | None]:
        """Reward function that checks if the final answer matches the expected answer."""
        rewards = []
        for completion, ans, t, images, images_offset, in zip(completions, answer, task, all_images, all_images_offset):
            reward = None
            if t == "vg":
                try:
                    reward = self.vg_reward_func(
                    completions=[completion],
                    answer=[ans],
                    task=[t],
                    all_images=[images],
                    all_images_offset=[images_offset],
                    )[0]
                except:
                    reward = None
            else:
                reward = None
            rewards.append(reward)
        return rewards

    def correct_crop_func(self, completions, answer, all_images, all_images_offset, **kwargs) -> List[float | None]:
        debug: bool = kwargs.get("debug", False)
        rewards: List[float | None] = []
        if debug:
            print(f'images_offset: {all_images_offset}')
        for completion, box, images, images_offset in zip(completions, answer, all_images, all_images_offset):
            # 1. extract the crop bbox from assistant messages
            crop_bboxs: List[Tuple[int,int,int,int]] = []
            try:
                for msg in completion:
                    if msg.get("role") != "assistant":
                        continue
                    for item in msg.get("content", []):
                        if item.get("type") != "text":
                            continue
                        text = item["text"]
                        id_num, top_left, bottom_right = parse_crop_bbox_from_text(text)
                        if top_left is not None:
                            dx, dy = images_offset[id_num]
                            x1, y1 = top_left
                            x2, y2 = bottom_right
                            
                            # Adjust coordinates based on image offset
                            parsed = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                            crop_bboxs.append(parsed)
                
                    if isinstance(box, str):
                        try:
                            box = tuple(ast.literal_eval(box))
                        except Exception:
                            nums2 = re.findall(r"-?\d+", box)
                            box = tuple(map(int, nums2))

                    # extract image size from completion
                    first_image = extract_first_image(completion)
                    if first_image is None:
                        raise ValueError("No image found in the conversation.")
                    img_size = first_image.size # (width, height)
                    
                    reward = average_crop_reward(crop_bboxs, box, img_size)
                    # print(f"Crop reward: {reward}.")

            except Exception:
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards
    
    def correct_extract_func(
        self,
        completions,              # List[List[MessageDict]]
        answer,                   # List[GT bbox | str]
        all_images,               # List[List[PIL.Image]]
        all_images_offset,        # List[List[Tuple[int,int]]]
        **kwargs,
    ) -> list[float | None]:
        """
        Evaluate whether the assistant’s <extract>(...) calls correctly cover the
        ground-truth bbox for each sample.

        Returns
        -------
        List[float | None]
            One reward per sample: 1.0 if any extract region fully contains the GT
            bbox, otherwise 0.0. (None is never returned here, kept for API parity.)
        """
        import re, ast
        from typing import Optional, Tuple, List

        debug: bool = kwargs.get("debug", False)

        # ------------------------------------------------------------------
        # === 1. inner tools ===
        # ------------------------------------------------------------------
        _EXTRACT_RE = re.compile(
            r"<extract>\s*\(\s*['\"]?Image_(\d+)['\"]?\s*,\s*['\"]?"
            r"(left|center|right)['\"]?\s*,\s*['\"]?(top|center|bottom)['\"]?\s*\)\s*</extract>",
            re.I,
        )

        def _parse_extract(text: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
            m = _EXTRACT_RE.search(text)
            if not m:
                return None, None, None
            return int(m.group(1)), m.group(2).lower(), m.group(3).lower()

        def _quadrant_bbox(w: int, h: int, x_pos: str, y_pos: str) -> Tuple[int, int, int, int]:
            half_w, half_h = w // 2, h // 2
            x0 = 0 if x_pos == "left" else (w - half_w) // 2 if x_pos == "center" else w - half_w
            y0 = 0 if y_pos == "top"  else (h - half_h) // 2 if y_pos == "center" else h - half_h
            return (x0, y0, x0 + half_w, y0 + half_h)

        def _contains(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]) -> bool:
            ox1, oy1, ox2, oy2 = outer
            ix1, iy1, ix2, iy2 = inner
            return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2

        def _extract_reward(
            bboxes: List[Tuple[int, int, int, int]],
            gt: Tuple[int, int, int, int]
        ) -> float:
            if not bboxes:
                return 0.0

            hits = sum(1 for b in bboxes if _contains(b, gt))
            return hits / len(bboxes)

        rewards: List[float] = []
        if debug:
            print(f'images_offset: {all_images_offset}')
        for completion, gt_box, images, offsets in zip(
            completions, answer, all_images, all_images_offset
        ):
            try:
                extract_bboxes: list[Tuple[int, int, int, int]] = []

                for idx, msg in enumerate(completion):
                    if msg.get("role") != "assistant":
                        continue
                    for part in msg.get("content", []):
                        if part.get("type") != "text":
                            continue
                        img_id, x_pos, y_pos = _parse_extract(part["text"])
                        if x_pos is None:
                            continue
                        if debug:
                                print(f"images: {images}")
                        w, h = images[img_id].size
                        bbox = _quadrant_bbox(w, h, x_pos, y_pos)
                        if debug:
                            print(f"Image_{img_id} size: ({w}, {h}), Extract bbox (pre-offset): {bbox}, offset: {offsets}")
                        dx, dy = offsets[img_id]
                        x1, y1, x2, y2 = bbox
                        extract_bboxes.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy))
                        
                        if debug:
                            print(f"[DEBUG] Extract real_bbox: {(x1 + dx, y1 + dy, x2 + dx, y2 + dy)}")

                            for nxt in completion[idx+1:]:
                                if nxt.get("role") == "user":
                                    user_text = "\n".join(
                                        c["text"] for c in nxt.get("content", []) if c.get("type")=="text"
                                    )
                                    print(f"[DEBUG] Corresponding user message:\n{user_text}")
                                    m_img = re.search(
                                        r"\[Image_(\d+)[^\]]*offset[:：]?\s*\(\s*(\d+),\s*(\d+)\s*\)",
                                        user_text
                                    )
                                    dx_debug, dy_debug = int(m_img.group(2)), int(m_img.group(3))
                                    m_size = re.search(r"Cropped a region of size\s*(\d+)×(\d+)", user_text)
                                    w, h = map(int, m_size.groups())
                                    x1_debug, y1_debug, x2_debug, y2_debug = 0, 0, w, h
                                    if (x1_debug + dx_debug, y1_debug + dy_debug, x2_debug + dx_debug, y2_debug + dy_debug) != (x1 + dx, y1 + dy, x2 + dx, y2 + dy):
                                        print(f"[ERROR] Offset mismatch: expected ({x1_debug + dx}, {y1_debug + dy}, {x2_debug + dx}, {y2_debug + dy}), got ({x1 + dx}, {y1 + dy}, {x2 + dx}, {y2 + dy})")
                                    else:
                                        print(f"[DEBUG] Offset matches: ({x1 + dx}, {y1 + dy}, {x2 + dx}, {y2 + dy}) == ({x1_debug + dx_debug}, {y1_debug + dy_debug}, {x2_debug + dx_debug}, {y2_debug + dy_debug})")
                                    break
                        

                # 2-2. 解析 ground-truth
                if isinstance(gt_box, str):
                    try:
                        gt_box = tuple(ast.literal_eval(gt_box))
                    except Exception:
                        nums = [int(n) for n in re.findall(r"-?\d+", gt_box)]
                        gt_box = tuple(nums)

                # 2-3. 计算奖励
                reward = _extract_reward(extract_bboxes, gt_box)

                if debug:
                    print(f"GT: {gt_box} | Extracts: {extract_bboxes} | Reward={reward}")

            except Exception as e:
                if debug:
                    print(f"[correct_extract_func] Error: {e}")
                reward = 0.0

            rewards.append(reward)

        return rewards

    def correct_find_color(
        self,
        completions: List[List[dict]],
        answer: List[Tuple[int, ...]],
        all_images: List[List],
        all_images_offset: List[List[Tuple[int, int]]],
        **kwargs
    ) -> List[float]:
        """
        Reward = 1 if ground-truth coordinate/box is inside the region returned by find_color.
        """
        debug: bool = kwargs.get("debug", False)
        rewards: List[float] = []
        
        for completion, box, images, images_offset in zip(completions, answer, all_images, all_images_offset):
            if debug:
                print(f"[DEBUG] images_offset: {images_offset}")
            reward = 0.0
            
            try:
                # 1. locate the assistant message with a find_color tool call
                for idx, msg in enumerate(completion):
                    if msg.get("role") != "assistant":
                        continue
                    # check for closing tag
                    if not any(item.get("type") == "text" and "</find_color>" in item["text"] 
                            for item in msg.get("content", [])):
                        continue
                    
                    # 2. find the following user message (tool output)
                    user_msg = None
                    for nxt in completion[idx+1:]:
                        if nxt.get("role") == "user":
                            user_msg = nxt
                            break
                    if user_msg is None:
                        raise ValueError("No user message after find_color call.")
                    
                    # 3. extract image id and offset from user message text
                    combined_text = "\n".join(
                        c["text"] for c in user_msg.get("content", []) if c.get("type") == "text"
                    )
                    if debug:
                        print(f"[DEBUG] Find find_color command, the User message text is: {combined_text}")
                    m_img = re.search(
                        r"\[Image_(\d+)[^\]]*offset[:：]?\s*\(\s*(\d+),\s*(\d+)\s*\)",
                        combined_text
                    )
                    if not m_img:
                        raise ValueError("Cannot parse Image ID and offset.")
                    dx, dy = int(m_img.group(2)), int(m_img.group(3))
                    if debug:
                        img_id = int(m_img.group(1))
                        offset_from_input = images_offset[img_id]
                        if (dx, dy) != offset_from_input:
                            print(f"[ERROR] Offset mismatch: expected {offset_from_input}, got ({dx}, {dy})")
                        else:
                            print(f"[DEBUG] Offset matches: ({dx}, {dy})")
                    
                    # 4. extract bounding box of the cropped region
                    m_size = re.search(r"Cropped a region of size\s*(\d+)×(\d+)", combined_text)
                    if not m_size:
                        raise ValueError("Cannot parse crop size or coords.")
                    w, h = map(int, m_size.groups())
                    x1, y1, x2, y2 = 0, 0, w, h
                    real_tl = (x1 + dx, y1 + dy)
                    real_br = (x2 + dx, y2 + dy)
                    
                    # 6. normalize ground-truth
                    if isinstance(box, str):
                        try:
                            box = tuple(ast.literal_eval(box))
                        except Exception:
                            box = tuple(map(int, re.findall(r"-?\d+", box)))
                    
                    # 7. check inclusion
                    if len(box) == 2:
                        x_gt, y_gt = box
                        if real_tl[0] <= x_gt <= real_br[0] and real_tl[1] <= y_gt <= real_br[1]:
                            reward = 1.0
                    else:
                        x1_gt, y1_gt, x2_gt, y2_gt = box
                        if (real_tl[0] <= x1_gt <= real_br[0] and
                            real_tl[1] <= y1_gt <= real_br[1] and
                            real_tl[0] <= x2_gt <= real_br[0] and
                            real_tl[1] <= y2_gt <= real_br[1]):
                            reward = 1.0
                    
                    break  # done for this example
                    
            except Exception as e:
                if debug:
                    print(f"[ERROR] correct_find_color: {e}")
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards
    
    def tool_execution_reward_func(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """
        def check_execution(trajectory):
            tool_attempts = 0
            successful_executions = 0
            
            # Find assistant messages with tools and their responses
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg['content'][0]["text"])
                    if hasattr(parsed, 'crop') and parsed.crop is not None:
                        # Found a properly formatted tool message
                        if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                            tool_attempts += 1
                            multiplier = 1.0 
                            response = str(parsed.crop)
                            if (("sympy" in response) or ("numpy" in response)) and len(response) > 100:
                                multiplier = 1.5
                            else:
                                multiplier = 1.0
                                
                            # Extract tool response text
                            tool_response = None
                            for elem in trajectory[i + 1]['content']:
                                # make sure the content is a dict and has 'text' key
                                if isinstance(elem, dict) and "text" in elem:
                                    tool_response = elem["text"]
                                    break
                                
                            if '<|Tool_Error|>' not in tool_response:
                                successful_executions += 1 * multiplier
            
            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return (successful_executions / tool_attempts)
        
        return [check_execution(c) for c in completions]
    
    def get_named_tool_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that checks tool execution success for a specific tool.

        Uses XMLParser to identify proper tool calls.
        """
        def tool_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
            """
            Reward function that checks execution success for the {tool_name} tool.
            
            Check Whether the tool was executed successfully.
            For example, crop tool -> image should be included in the next message.
            """
            import json
            
            def check_tool_execution(trajectory: List[Dict[str, str]]) -> float:
                tool_attempts = 0
                successful_executions = 0
                
                # Find assistant messages with the specific tool and their responses
                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        # Use parser to check for tool tag
                        parsed = self.parser.parse(msg['content'][0]["text"])
                        if hasattr(parsed, 'crop') and parsed.crop is not None:
                            try:
                                command = parsed.crop
                                if isinstance(command, str):
                                    # extract the function name from the command
                                    func_name = command.split('(', 1)[0].strip()
                                    if func_name == 'crop':
                                        # Found a properly formatted tool message for the specific tool
                                        if i + 1 < len(trajectory) and trajectory[i+1]['role'] == 'user':
                                            tool_attempts += 1

                                            next_msg = trajectory[i+1]
                                            content = next_msg.get('content', [])

                                            # if the content is a string, and the 'image_url' is in the content, count it as successful execution
                                            if isinstance(content, list) and any(
                                                isinstance(item, dict) and item.get("type") == "image_url"
                                                for item in content
                                            ):
                                                successful_executions += 1
                            except json.JSONDecodeError:
                                pass
                
                # Calculate reward
                if tool_attempts == 0:
                    return 0.0
                return (successful_executions / tool_attempts)
            
            return [check_tool_execution(c) for c in completions]
        
        # Create a function with the dynamic name based on tool_name
        tool_reward_func.__name__ = f"{tool_name}_reward_func"
        return tool_reward_func
    
    def get_named_tool_count_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """
        def tool_count_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            def count_tool_executions(trajectory: List[Dict[str, str]]) -> float:
                successful_executions = 0.0
                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        parsed = self.parser.parse(msg['content'])
                        if hasattr(parsed, 'tool') and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if isinstance(command, dict) and command.get("name") == tool_name:
                                    # Found a properly formatted tool message for the specific tool
                                    if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                                        parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                                        if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                            successful_executions += 1
                            except json.JSONDecodeError:
                                pass
                return successful_executions
            
            return [count_tool_executions(c) for c in completions]
        
        tool_count_reward_func.__name__ = f"{tool_name}_count_reward_func"
        return tool_count_reward_func

    def get_named_tool_attempt_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """
        def tool_attempt_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            def count_tool_executions(trajectory: List[Dict[str, str]]) -> float:
                attempted_executions = 0.0
                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        parsed = self.parser.parse(msg['content'])
                        if hasattr(parsed, 'tool') and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if isinstance(command, dict) and command.get("name") == tool_name:
                                    attempted_executions += 1
                            except json.JSONDecodeError:
                                pass
                return attempted_executions
            
            return [count_tool_executions(c) for c in completions]
            
        tool_attempt_reward_func.__name__ = f"{tool_name}_attempt_reward_func"
        return tool_attempt_reward_func