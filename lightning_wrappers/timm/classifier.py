"""
PyTorch Lightning LightningModule for timm image classifiers.

See also:
    - [`timm` documentation](https://huggingface.co/docs/timm/index)
    - [`timm` model zoo](https://huggingface.co/timm/models)
"""

from functools import lru_cache
from typing import Any, Callable

import timm
from torchvision import transforms

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
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=n_classes,
        )
        super().__init__(
            model=model,
            n_classes=n_classes,
            head_name=None,
            **kwargs,
        )
        self.save_hyperparameters()

    @lru_cache(maxsize=1)
    def get_transform(self) -> Callable | transforms.Compose:
        """
        Get the transformation function for the model.

        Returns:
            Transformation function. If the model's pretrained config cannot be
            resolved, falls back to standard ImageNet preprocessing (resize to
            224×224, normalize with ImageNet statistics).
        """
        try:
            data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
            transform = timm.data.create_transform(**data_cfg)
            return transform
        except Exception:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
