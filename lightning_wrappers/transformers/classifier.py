"""See `HuggingFaceClassifier` documentation."""

from typing import Any, Callable

from anyio.functools import lru_cache
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
            model_name (str): Model name as in the [HuggingFace model
                hub](https://huggingface.co/models?pipeline_tag=image-classification).
                If the model name starts with `timm/`, use
                `lightning_wrappers.timm.TimmClassifier` instead.
            n_classes (int): Number of output classes.
            head_name (str | None, optional): Name of the classification head.
                If None, the default head is used.
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

    @lru_cache(maxsize=1)
    def get_transform(
        self, *args: Any, **kwargs: Any
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """
        Wraps the HuggingFace `AutoImageProcessor` corresponding to the current
        model architecture.

        Note:
            The `*args` and `**kwargs` are passed to the
            `AutoImageProcessor.from_pretrained` method. In particular, if you
            want to pass `kwargs` to the image processor itself, set `kwargs` in
            the `kwargs`:

            ```python
            model = TransformersClassifier("microsoft/resnet-50", ...)
            # AutoImageProcessor.from_pretrained("microsoft/resnet-50")
            # constructs a ConvNextImageProcessor
            transform = model.get_transform(
                token=True,  # → AutoImageProcessor.from_pretrained
                kwargs={"do_resize": 384},  → ConvNextImageProcessor.__init__
            )
            ```

        See also:
            [`AutoImageProcessor` documentation](https://huggingface.co/docs/transformers/v5.2.0/en/model_doc/auto#transformers.AutoImageProcessor)
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
