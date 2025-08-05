import re
from typing import List, Dict, Any, Union, Tuple, Optional, Callable
from types import SimpleNamespace

class XMLParser:
    def __init__(self, fields: List[Union[str, Tuple[str, ...]]]):
        """
        Initialize the parser with field definitions.
        
        Each field may be:
          - a string (e.g. "reasoning"): the XML tag is fixed.
          - a tuple of alternatives (e.g. ("code", "answer")): the first element is
            the canonical name used for formatting, and all elements are allowed tags
            when parsing.
            
        The schema is assumed to have no duplicate names.
        """
        self._fields: List[Tuple[str, List[str]]] = []  # List of (canonical, [alternatives])
        seen = set()
        for field in fields:
            if isinstance(field, str):
                canonical = field
                alternatives = [field]
            elif isinstance(field, tuple):
                if not field:
                    raise ValueError("Field tuple cannot be empty.")
                canonical = field[0]
                if not all(isinstance(alt, str) for alt in field):
                    raise TypeError("All alternatives in a tuple must be strings.")
                alternatives = list(field)
            else:
                raise TypeError("Each field must be a string or a tuple of strings.")
            if canonical in seen:
                raise ValueError(f"Duplicate field name: {canonical}")
            seen.add(canonical)
            self._fields.append((canonical, alternatives))
    
    def get_xml_reward_func(self) -> Callable:
        """
        Return a reward function that checks for proper XML tag usage.
        
        The returned function evaluates if messages in trajectories properly use 
        the expected XML tags defined in this parser's fields configuration.
        For example: 
        <crop>...</crop>
        <answer>...</answer>
        They should appear in pairs.
        """
        def xml_reward_func(completions, **kwargs) -> List[float]:
            """Reward function that checks for proper XML tag usage in completions."""
            def count_xml(trajectory) -> float:
                # Get all messages from the model
                model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
                if not model_messages:
                    return 0.0
                
                # Calculate XML tag usage scores for each message
                xml_scores = []
                for msg in model_messages:
                    content = msg['content'][0]["text"]
                    score = 0
                    total_checks = 0
                    
                    # For each canonical field with its alternatives
                    for canonical, alternatives in self._fields:
                        # Track if at least one alternative was used for this field
                        field_used = False
                        
                        # Each alt can only be used once
                        for alt in alternatives:
                            # If this alternative is used, check it has proper tags
                            if content.count(f"<{alt}>") > 0 or content.count(f"</{alt}>") > 0:
                                field_used = True
                                score += 1 - abs(content.count(f"<{alt}>") - 1)
                                score += 1 - abs(content.count(f"</{alt}>") - 1)
                                total_checks += 2
                        
                        # If no alternatives for this field were used, we don't add to total_checks
                        # because we're not requiring any specific field to be present
                    
                    # Calculate normalized score for this message
                    if total_checks > 0:
                        xml_scores.append(score / total_checks)
                    else:
                        # If no tags used at all, give a zero score
                        xml_scores.append(0.0)
                
                # Return average XML score across all messages
                if not xml_scores:
                    return 0.0
                return (sum(xml_scores) / len(xml_scores))
            
            # Apply the XML check to each completion trajectory
            return [count_xml(c) for c in completions]

        return xml_reward_func

    def get_format_reward_func(self) -> Callable:
        """
        Return a reward function that applies a unified three-stage reward process to all assistant tags (tool calls and final answer), then applies end-of-trajectory discount:

        For each assistant message containing any of the tags (<think>, <crop>, <find_color>, <extract>, <answer>):

        Stage 1 (format structure):
        - Must contain <think>...</think> and one of the target tags (<crop>, <find_color>, <extract>, <answer>)
        - The <think> tag must occur before any target tag
        - No content allowed between </think> and the target tag's opening
        - Reward: +0.2

        Stage 2 (pattern match):
        - The content of the target tag (tool or answer) must match its specific regex:
            crop:   <crop>(Image_id, (x1, y1), (x2, y2))</crop>
            extract:<extract>(Image_id, positionX, positionY)</extract>
            find_color:<find_color>(Image_id, (r, g, b))</find_color>
            answer: <answer>(image_id,(x,y))</answer>
        - Reward: +0.3

        Stage 3 (result correctness):
        - For tool tags (crop, extract, find_color): the next user message must exist and contain no <Tool_ERROR>
        - For answer tag: this assistant message must be the last assistant message in the trajectory
        - Reward: +0.5

        After summing raw rewards over assistant steps, multiply by gamma**N
        (N = number of assistant steps), gamma = exp(-1/2.5).
        """
        import math
        import re

        # Precompile regex patterns for each tag
        patterns = {
            'crop': re.compile(r'^<crop>\(Image_\d+,\s*\(\s*\d+\s*,\s*\d+\s*\),\s*\(\s*\d+\s*,\s*\d+\s*\)\)</crop>$'),
            'extract': re.compile(r'^<extract>\(Image_\d+,\s*"?(?:left|center|right)"?,\s*"?(?:top|center|bottom)"?\)</extract>$'),
            'find_color': re.compile(r'^<find_color>\(Image_\d+,\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)\)</find_color>$'),
            'answer': re.compile(r'^<answer>\(\s*Image_\d+\s*,\s*\(\s*\d+\s*,\s*\d+\s*\)\s*\)</answer>$')
        }

        def format_reward_func(completions, **kwargs) -> List[float]:
            # gamma = math.exp(-1/2.5)
            gamma = 0.75
            def check_format(trajectory):
                # print(f"Checking trajectory")
                raw_reward = 0.0
                assistant_positions = [i for i, msg in enumerate(trajectory) if msg.get('role') == 'assistant']

                for idx, pos in enumerate(assistant_positions):
                    content = trajectory[pos]['content'][0]['text'].strip()
                    step_reward = 0.0

                    # Must have a <think> tag
                    think_start = content.find('<think>')
                    think_end = content.find('</think>')
                    if think_start == -1 or think_end == -1 or think_start > think_end:
                        raw_reward += -0.3
                        continue
                    
                    # Penalty for overly long think & tool content: proportional to excess length
                    if len(content) > 512:
                        excess = len(content) - 512
                        # proportional penalty: scale by ratio of excess to max_think_len
                        penalty = min((excess / 512) * 0.3, 0.3)
                        raw_reward -= penalty
                    
                    # Find earliest target tag start
                    tag_positions = []
                    for tag in patterns:
                        p = content.find(f'<{tag}>')
                        if p != -1:
                            tag_positions.append(p)
                    if not tag_positions:
                        continue
                    first_tag_pos = min(tag_positions)
                    # Ensure <think> comes before any tag
                    if first_tag_pos < think_start:
                        continue

                    # Check each tag for the three-stage process
                    for tag, pat in patterns.items():
                        open_tag = f'<{tag}>'
                        close_tag = f'</{tag}>'
                        start = content.find(open_tag)
                        end = content.find(close_tag)
                        # Must find a complete tag after the think_end
                        if start != -1 and end != -1 and start > think_end:
                            # Stage 1: no content between </think> and <tag>
                            between = content[think_end + len('</think>'):start]
                            if between.strip() != '':
                                break  # fail stage 1
                            step_reward += 0.2
                            # Extract full tag substring
                            full_tag = content[start:end + len(close_tag)]
                            # Stage 2: pattern match
                            if not pat.match(full_tag):
                                break  # fail stage 2
                            step_reward += 0.3
                            # Stage 3: correctness
                            if tag == 'answer':
                                # Must be the last assistant message
                                if pos == assistant_positions[-1]:
                                    step_reward += 1
                            else:
                                next_idx = pos + 1
                                if next_idx < len(trajectory) and trajectory[next_idx].get('role') == 'user':
                                    user_text = trajectory[next_idx]['content'][0].get('text', '')
                                    if '<|Format_Error|>' not in user_text:
                                        step_reward += 0.5
                            break  # only first applicable tag

                    raw_reward += step_reward
                    # print(f"Step {idx} (pos {pos}): raw_reward={raw_reward}, step_reward={step_reward}")
                # Apply end-of-trajectory discount
                N = len(assistant_positions)
                return raw_reward * (gamma ** N)
                # return raw_reward / N

            return [check_format(traj) for traj in completions]

        return format_reward_func

    def get_fields(self) -> List[str]:
        """Return a list of the canonical field names (in order)."""
        return [canonical for canonical, _ in self._fields]
    
    def format(self, **kwargs) -> str:
        """
        Format the provided keyword arguments into an XML string.
        
        For fields with alternatives (tuple), the canonical name (the first element)
        is used as the XML tag. The method looks for a provided value using any of the
        allowed names (preferring the canonical if present).
        
        Example usage:
            parser = XMLParser(['reasoning', ('code', 'answer')])
            formatted_str = parser.format(reasoning="...", code="...")
        """
        parts = []
        for canonical, alternatives in self._fields:
            value = None
            # Look for a provided value using any of the acceptable keys,
            # preferring the canonical name if it exists.
            if canonical in kwargs:
                value = kwargs[canonical]
            else:
                for alt in alternatives:
                    if alt in kwargs:
                        value = kwargs[alt]
                        break
            if value is None:
                raise ValueError(f"Missing value for field '{canonical}' (allowed: {alternatives})")
            # Use the canonical name as the tag for formatting.
            parts.append(f"<{canonical}>\n{value}\n</{canonical}>")
        return "\n".join(parts)
    
    def parse(self, text: str, strip: bool = True) -> Any:
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
        for canonical, alternatives in self._fields:
            # For each allowed alternative tag, search independently.
            for alt in alternatives:
                # Regex pattern to capture the content between the tags.
                pattern = rf"<{alt}>\s*(.*?)\s*</{alt}>"
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    results[alt] = match.group(1).strip() if strip else match.group(1)
                else:
                    results[alt] = None
        return SimpleNamespace(**results) # Convert dict to object for attribute access