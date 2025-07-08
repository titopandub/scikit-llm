# Enhanced Vertex AI Support for scikit-llm

This document describes the enhanced Vertex AI functionality that adds support for the Google GenAI library and thinking control for Gemini models, integrated directly into the standard Vertex AI classifiers.

## Overview

The enhanced Vertex AI support provides:

1. **Google GenAI Library Integration**: Uses the newer `google-genai` library alongside the existing Vertex AI library
2. **Thinking Control**: Ability to control the "thinking" behavior of Gemini models via `thinking_budget` parameter
3. **Backward Compatibility**: All existing functionality remains unchanged
4. **Seamless Integration**: Thinking control is built directly into the standard classifiers

## New Features

### Enhanced Completion Functions

- `get_completion_chat_gemini_enhanced()`: Enhanced Gemini completion with thinking control
- `get_completion_chat_gemini_no_thinking()`: Gemini completion with thinking explicitly disabled

### Enhanced Standard Classes

The standard Vertex AI classifiers now support thinking control:

- `ZeroShotVertexClassifier`: Now supports `thinking_budget` parameter for Gemini models
- `MultiLabelZeroShotVertexClassifier`: Now supports `thinking_budget` parameter for Gemini models

## Installation

The enhanced functionality requires the `google-genai` library, which is now included in the dependencies:

```bash
pip install scikit-llm
```

Or for development:

```bash
pip install -e .
```

## Usage

### Basic Usage with Thinking Disabled (Default)

```python
from skllm.models.vertex.classification.zero_shot import ZeroShotVertexClassifier

# Create classifier with thinking disabled (default)
classifier = ZeroShotVertexClassifier(
    model="gemini-1.5-flash",
    thinking_budget=0  # Disable thinking for faster responses
)

# Fit and predict as usual
classifier.fit(None, ["positive", "negative", "neutral"])
predictions = classifier.predict(["I love this!", "This is terrible"])
```

### Usage with Thinking Enabled

```python
from skllm.models.vertex.classification.zero_shot import ZeroShotVertexClassifier

# Create classifier with thinking enabled
classifier = ZeroShotVertexClassifier(
    model="gemini-1.5-flash",
    thinking_budget=1000  # Allow thinking for complex reasoning
)

classifier.fit(None, ["positive", "negative", "neutral"])
predictions = classifier.predict(["I love this!", "This is terrible"])
```

### Multi-Label Classification

```python
from skllm.models.vertex.classification.zero_shot import MultiLabelZeroShotVertexClassifier

classifier = MultiLabelZeroShotVertexClassifier(
    model="gemini-1.5-flash",
    thinking_budget=0,  # Disable thinking
    max_labels=2
)

classifier.fit(None, ["positive", "negative", "emotional", "factual"])
predictions = classifier.predict(["I love this amazing product!"])
```

### Backward Compatibility with Older Models

```python
from skllm.models.vertex.classification.zero_shot import ZeroShotVertexClassifier

# Using older models - thinking_budget is ignored for non-Gemini models
classifier = ZeroShotVertexClassifier(
    model="text-bison@002",
    thinking_budget=100  # This will be ignored for non-Gemini models
)

classifier.fit(None, ["positive", "negative", "neutral"])
predictions = classifier.predict(["I love this!", "This is terrible"])
```

### Direct Function Usage

```python
from skllm.llm.vertex.completion import (
    get_completion_chat_gemini_enhanced,
    get_completion_chat_gemini_no_thinking
)

# Enhanced completion with thinking control
response = get_completion_chat_gemini_enhanced(
    model="gemini-1.5-flash",
    context="You are a helpful assistant",
    text="Classify this sentiment: I love this!",
    thinking_budget=0  # Disable thinking
)

# Simplified no-thinking completion
response = get_completion_chat_gemini_no_thinking(
    model="gemini-1.5-flash",
    context="You are a helpful assistant", 
    text="Classify this sentiment: I love this!"
)
```

## Parameters

### thinking_budget

- **Type**: `int`
- **Default**: `0`
- **Description**: Controls the thinking behavior of Gemini models
  - `0`: Disables thinking (faster, more deterministic responses)
  - `> 0`: Enables thinking with the specified token budget (slower, potentially more thoughtful responses)
- **Note**: Only applies to Gemini models (models starting with "gemini-"). Ignored for other models.

## Backward Compatibility

All existing Vertex AI functionality remains unchanged:

- Existing code using `ZeroShotVertexClassifier` and `MultiLabelZeroShotVertexClassifier` continues to work exactly as before
- The `thinking_budget` parameter is optional and defaults to `0` (thinking disabled)
- Non-Gemini models ignore the `thinking_budget` parameter completely

## Fallback Behavior

If the `google-genai` library is not available, the enhanced functions automatically fall back to the standard Vertex AI implementation, ensuring compatibility across different environments.

## Examples

See `examples/vertex_enhanced_example.py` for a complete working example demonstrating all the new functionality.

## Migration from Custom Implementation

If you were using a custom monkey-patch approach (like the `custom_vertex_completion.py`), you can now use the built-in enhanced functionality:

### Before (Custom Monkey Patch)
```python
from custom_vertex_completion import apply_no_thinking_patch
apply_no_thinking_patch()

# Use regular classifier
from skllm.models.vertex.classification.zero_shot import ZeroShotVertexClassifier
classifier = ZeroShotVertexClassifier(model="gemini-1.5-flash")
```

### After (Built-in Enhanced Functionality)
```python
from skllm.models.vertex.classification.zero_shot import ZeroShotVertexClassifier

classifier = ZeroShotVertexClassifier(
    model="gemini-1.5-flash",
    thinking_budget=0  # Equivalent to the monkey patch
)
```

## Key Benefits

1. **No Separate Classes**: Thinking control is integrated directly into the standard classifiers
2. **Seamless Migration**: Existing code continues to work without changes
3. **Clean API**: Simple `thinking_budget` parameter controls the behavior
4. **Automatic Fallback**: Works even when google-genai library is not available
5. **Model-Aware**: Only applies thinking control to Gemini models

## Contributing

When contributing to the enhanced Vertex AI functionality:

1. Ensure backward compatibility with existing code
2. Add appropriate tests for new functionality
3. Update documentation as needed
4. Follow the existing code style and patterns

## Dependencies

- `google-genai>=0.3.0,<1.0.0`: For enhanced Gemini support
- `google-cloud-aiplatform[pipelines]>=1.27.0,<2.0.0`: For standard Vertex AI support (existing)
