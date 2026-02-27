"""See `HuggingFaceClassifier` documentation."""

from typing import Any, Callable

import lightning as pl
from anyio.functools import lru_cache
from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

from ..utils import replace_head


class TransformersClassifier(pl.LightningModule):
    """
    Pretrained classifier model loaded from the [HuggingFace model
    hub](https://huggingface.co/models?pipeline_tag=image-classification).
    """

    model: nn.Module

    def __init__(
        self,
        model_name: str,
        n_classes: int | None = None,
        head_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_name (str): Model name as in the [HuggingFace model
                hub](https://huggingface.co/models?pipeline_tag=image-classification).
                If the model name starts with `timm/`, use
                `lightning_wrappers.timm.TimmClassifier` instead.
            n_classes (int | None, optional): Number of output classes. If left
                to `None`, the default number of output classes of the
                pretrained model is used, and the classification head is not
                replaced.
            head_name (str | None, optional): Name of the classification head.
                If None, the default head is used.
        """
        self.save_hyperparameters()
        if model_name.startswith("timm/"):
            raise ValueError(
                "If the model name starts with `timm/`, use "
                "`lightning_wrappers.timm.TimmClassifier` instead."
            )
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name
        )
        if head_name is not None and n_classes is not None:
            self.model = replace_head(self.model, head_name, n_classes)

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
