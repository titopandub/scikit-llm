from skllm.utils import retry
from vertexai.language_models import ChatModel, TextGenerationModel
from vertexai.generative_models import GenerativeModel, GenerationConfig


# Import Google GenAI for enhanced Gemini support
try:
    from google import genai
    from google.genai.types import GenerateContentConfig, ThinkingConfig
    _GENAI_AVAILABLE = True
    _genai_client = None  # Initialize later with proper config
except ImportError:
    _GENAI_AVAILABLE = False
    _genai_client = None


def _get_genai_client():
    """Get or create the GenAI client with proper Vertex AI configuration."""
    global _genai_client
    if _genai_client is None and _GENAI_AVAILABLE:
        try:
            # Try to initialize with Vertex AI configuration
            import vertexai
            from google.auth import default
            
            # Get default credentials and project
            credentials, project = default()
            
            # Initialize with Vertex AI
            _genai_client = genai.Client(
                vertexai=True,
                project=project,
                location="us-central1"  # Default location
            )
        except Exception:
            # Fallback: try without explicit config (will use environment variables)
            try:
                _genai_client = genai.Client()
            except Exception:
                # If all fails, disable GenAI
                global _GENAI_AVAILABLE
                _GENAI_AVAILABLE = False
                _genai_client = None
    return _genai_client


@retry(max_retries=3)
def get_completion(model: str, text: str):
    if model.startswith("text-"):
        model_instance = TextGenerationModel.from_pretrained(model)
    else:
        model_instance = TextGenerationModel.get_tuned_model(model)
    response = model_instance.predict(text, temperature=0.0)
    return response.text


@retry(max_retries=3)
def get_completion_chat_mode(model: str, context: str, text: str):
    model_instance = ChatModel.from_pretrained(model)
    chat = model_instance.start_chat(context=context)
    response = chat.send_message(text, temperature=0.0)
    return response.text


@retry(max_retries=3)
def get_completion_chat_gemini(model: str, context: str, text: str):
    """
    Get completion from Gemini model using Vertex AI.
    
    Args:
        model: The Gemini model name
        context: System instruction/context
        text: The input text to process
        
    Returns:
        str: The model's response text
    """
    model_instance = GenerativeModel(model, system_instruction=context)
    response = model_instance.generate_content(
        text, generation_config=GenerationConfig(temperature=0.0)
    )
    return response.text


@retry(max_retries=3)
def get_completion_chat_gemini_enhanced(model: str, context: str, text: str, thinking_budget: int = 0):
    """
    Enhanced Gemini completion using Google GenAI library with thinking control.
    
    Args:
        model: The Gemini model name
        context: System instruction/context
        text: The input text to process
        thinking_budget: Budget for thinking tokens (0 disables thinking)
        
    Returns:
        str: The model's response text
    """
    if not _GENAI_AVAILABLE:
        # Fallback to standard implementation if GenAI library is not available
        return get_completion_chat_gemini(model, context, text)
    
    # Get the properly initialized GenAI client
    client = _get_genai_client()
    if client is None:
        # Fallback if client initialization failed
        return get_completion_chat_gemini(model, context, text)
    
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=GenerateContentConfig(
            system_instruction=[context] if context else None,
            temperature=0.0,
            thinking_config=ThinkingConfig(thinking_budget=thinking_budget)
        )
    )
    return response.text
