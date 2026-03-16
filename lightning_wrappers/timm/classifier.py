"""
PyTorch Lightning LightningModule for timm image classifiers.

See also:
    - [`timm` documentation](https://huggingface.co/docs/timm/index)
    - [`timm` model zoo](https://huggingface.co/timm/models)
"""

from typing import Any, Callable

import timm
import torch
from torchvision.transforms import v2

from ..base import BaseClassifier


class TimmClassifier(BaseClassifier):
    """
    Lightning module wrapping a timm image classifier.
    """

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_name: Name of the timm model architecture.
            n_classes: Number of output classes.
            pretrained: Whether to load pretrained weights.
            **kwargs: Extra arguments forwarded to
                `BaseClassifier`.
        """
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=n_classes,
        )
        super().__init__(model=model, n_classes=n_classes, **kwargs)
        self.save_hyperparameters()

    def _get_transform(self, *args: Any, **kwargs: Any) -> Callable:
        """
        Return the preprocessing transform for this model.

        Falls back to standard ImageNet preprocessing (resize to
        224×224, normalize with ImageNet statistics) if the
        model's pretrained config cannot be resolved.
        """
        try:
            data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
            transform = timm.data.create_transform(**data_cfg)
            return transform  # type: ignore
        except Exception:
            transform = v2.Compose(
                [
                    v2.Resize(256),
                    v2.CenterCrop(224),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            return transform  # type: ignore
