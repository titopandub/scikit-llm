"""
Example demonstrating the enhanced Vertex AI functionality with thinking control.

This example shows how to use Vertex classifiers with thinking control
integrated directly into the standard classes.
"""

from skllm.models.vertex.classification.zero_shot import (
    ZeroShotVertexClassifier,
    MultiLabelZeroShotVertexClassifier
)

# Example data
texts = [
    "I love this product! It's amazing.",
    "This is terrible, I hate it.",
    "It's okay, nothing special.",
    "Best purchase ever! Highly recommend.",
    "Worst experience of my life."
]

labels = ["positive", "negative", "neutral"]

def main():
    print("Vertex AI Classifier with Thinking Control Example")
    print("=" * 50)
    
    # Example 1: Standard classifier with Gemini model and thinking disabled
    print("\n1. ZeroShotVertexClassifier with Gemini (thinking disabled)")
    classifier = ZeroShotVertexClassifier(
        model="gemini-1.5-flash",
        thinking_budget=0  # Disable thinking for faster, more deterministic responses
    )
    
    # Fit the classifier with the labels
    classifier.fit(None, labels)
    
    # Predict sentiment
    predictions = classifier.predict(texts)
    
    print("Predictions:")
    for text, pred in zip(texts, predictions):
        print(f"  '{text}' -> {pred}")
    
    # Example 2: Standard classifier with thinking enabled
    print("\n2. ZeroShotVertexClassifier with thinking enabled")
    classifier_with_thinking = ZeroShotVertexClassifier(
        model="gemini-1.5-flash",
        thinking_budget=1000  # Allow thinking for more complex reasoning
    )
    
    classifier_with_thinking.fit(None, labels)
    predictions_with_thinking = classifier_with_thinking.predict(texts)
    
    print("Predictions with thinking:")
    for text, pred in zip(texts, predictions_with_thinking):
        print(f"  '{text}' -> {pred}")
    
    # Example 3: Multi-label classifier with thinking control
    print("\n3. MultiLabelZeroShotVertexClassifier with thinking control")
    multi_labels = ["positive", "negative", "neutral", "emotional", "factual"]
    
    multi_classifier = MultiLabelZeroShotVertexClassifier(
        model="gemini-1.5-flash",
        thinking_budget=0,  # Disable thinking
        max_labels=2
    )
    
    multi_classifier.fit(None, multi_labels)
    multi_predictions = multi_classifier.predict(texts)
    
    print("Multi-label predictions:")
    for text, pred in zip(texts, multi_predictions):
        print(f"  '{text}' -> {pred}")
    
    # Example 4: Backward compatibility - using older models without thinking control
    print("\n4. Backward compatibility with text-bison model")
    legacy_classifier = ZeroShotVertexClassifier(
        model="text-bison@002",  # Older model, thinking_budget will be ignored
        thinking_budget=100  # This will be ignored for non-Gemini models
    )
    
    legacy_classifier.fit(None, labels)
    legacy_predictions = legacy_classifier.predict(texts[:2])  # Just test with 2 examples
    
    print("Legacy model predictions:")
    for text, pred in zip(texts[:2], legacy_predictions):
        print(f"  '{text}' -> {pred}")

if __name__ == "__main__":
    main()
