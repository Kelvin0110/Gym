import re
from typing import Optional


def box_parser(output_str: str) -> Optional[str]:
    """Parse content from \\boxed{...} format commonly used in mathematical responses.
    
    Args:
        output_str: String that may contain boxed content
        
    Returns:
        The content inside the boxed format, or None if not found
    """
    if not isinstance(output_str, str):
        output_str = str(output_str)

    if output_str is None:
        return None
    try:
        match = re.search(r"\\boxed\{(.*?)\}", output_str)
        parsed_option = match.group(1) if match else None
        return parsed_option
    except Exception as e:
        print(f"Regex error: {e}")
        return None


def _extract_last_assistant_text(body: GoogleSearchVerifyRequest) -> str:
    last_message = body.response.output[-1]
    if last_message.type == "message" and last_message.role == "assistant":
        return last_message.content
    else:
        return None