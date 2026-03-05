"""See `TransformersClassifier` documentation."""

from typing import Any, Callable

from transformers import AutoImageProcessor, AutoModelForImageClassification

from ..base import BaseClassifier


class TransformersClassifier(BaseClassifier):
    """
    Pretrained classifier model loaded from the [HuggingFace model
    hub](https://huggingface.co/models?pipeline_tag=image-classification).
    """

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        head_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_name: Model name from the `HuggingFace hub
                <https://huggingface.co/models?pipeline_tag=image-classification>`_.
                If the model name starts with ``timm/``, use
                `TimmClassifier` instead.
            n_classes: Number of output classes.
            head_name: Name of the classification head. If
                ``None``, the default head is used.
        """
        if model_name.startswith("timm/"):
            raise ValueError(
                "If the model name starts with `timm/`, use "
                "`lightning_wrappers.timm.TimmClassifier` instead."
            )
        model = AutoModelForImageClassification.from_pretrained(model_name)
        super().__init__(
            model=model,
            n_classes=n_classes,
            head_name=head_name,
            **kwargs,
        )
        self.save_hyperparameters()

    def _get_transform(
        self, *args: Any, **kwargs: Any
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """
        Wrap the HuggingFace `AutoImageProcessor` for this model.

        The ``*args`` and ``**kwargs`` are forwarded to
        `AutoImageProcessor.from_pretrained`. To pass kwargs to
        the image processor itself, nest them under ``kwargs``::

            model = TransformersClassifier("microsoft/resnet-50", ...)
            transform = model.get_transform(
                token=True,  # → from_pretrained
                kwargs={"do_resize": 384},  # → processor
            )

        See also:
            `AutoImageProcessor documentation
            <https://huggingface.co/docs/transformers/v5.2.0/en/model_doc/auto#transformers.AutoImageProcessor>`_
        """
        hftr = AutoImageProcessor.from_pretrained(
            self.hparams.model_name,  # type: ignore
            *args,
            **kwargs,
        )

        def _transform(batch: dict[str, Any]) -> dict[str, Any]:
            return {
                k: (
                    hftr(
                        [img.convert("RGB") for img in v],
                        return_tensors="pt",
                    )["pixel_values"]
                    # TODO: pass image_key from DS ↓
                    if k in ["img", "image", "jpg", "png"]
                    else v
                )
                for k, v in batch.items()
            }

        return _transform
