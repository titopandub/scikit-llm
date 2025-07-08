from skllm.llm.vertex.mixin import VertexClassifierMixin as _VertexClassifierMixin
from skllm.models._base.classifier import (
    BaseZeroShotClassifier as _BaseZeroShotClassifier,
    SingleLabelMixin as _SingleLabelMixin,
    MultiLabelMixin as _MultiLabelMixin,
)
from typing import Optional


class ZeroShotVertexClassifier(
    _BaseZeroShotClassifier, _SingleLabelMixin, _VertexClassifierMixin
):
    def __init__(
        self,
        model: str = "text-bison@002",
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        thinking_budget: int = 0,
        **kwargs,
    ):
        """
        Zero-shot text classifier using Vertex AI models.

        Parameters
        ----------
        model : str, optional
            model to use, by default "text-bison@002"
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
        prompt_template : Optional[str], optional
            custom prompt template to use, by default None
        thinking_budget : int, optional
            budget for thinking tokens (Gemini models only), 0 disables thinking, by default 0
        """
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            thinking_budget=thinking_budget,
            **kwargs,
        )


class MultiLabelZeroShotVertexClassifier(
    _BaseZeroShotClassifier, _MultiLabelMixin, _VertexClassifierMixin
):
    def __init__(
        self,
        model: str = "text-bison@002",
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        max_labels: Optional[int] = 5,
        thinking_budget: int = 0,
        **kwargs,
    ):
        """
        Multi-label zero-shot text classifier using Vertex AI models.

        Parameters
        ----------
        model : str, optional
            model to use, by default "text-bison@002"
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
        prompt_template : Optional[str], optional
            custom prompt template to use, by default None
        max_labels : Optional[int], optional
            maximum labels per sample, by default 5
        thinking_budget : int, optional
            budget for thinking tokens (Gemini models only), 0 disables thinking, by default 0
        """
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            max_labels=max_labels,
            thinking_budget=thinking_budget,
            **kwargs,
        )
