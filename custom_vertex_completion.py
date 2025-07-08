"""
Custom Vertex AI completion module that overrides scikit-llm's get_completion_chat_gemini
to disable thinking by setting thinking_budget=0.
"""

from skllm.utils import retry
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
import skllm.llm.vertex.completion as vertex_completion
import skllm.llm.vertex.mixin as vertex_mixin


client = genai.Client()


@retry(max_retries=3)
def get_completion_chat_gemini_no_thinking(model: str, context: str, text: str):
    """
    Modified version of get_completion_chat_gemini that disables thinking.
    
    Args:
        model: The Gemini model name
        context: System instruction/context
        text: The input text to process
        
    Returns:
        str: The model's response text
    """
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=GenerateContentConfig(
            system_instruction=[context] if context else None,
            temperature=0.0,
            thinking_config=ThinkingConfig(thinking_budget=0)
        )
    )
    return response.text


def apply_no_thinking_patch():
    """
    Apply monkey patch to replace the original get_completion_chat_gemini function
    with our version that disables thinking.
    """
    # Replace the function in the completion module
    vertex_completion.get_completion_chat_gemini = get_completion_chat_gemini_no_thinking
    
    # Also replace it in the mixin module since it imports the function
    vertex_mixin.get_completion_chat_gemini = get_completion_chat_gemini_no_thinking
