"""Simple Lightning wrapper around a plain PyTorch ``nn.Module``."""

from typing import Any, Callable

import torch
import torch.nn as nn
from torchvision.transforms import v2

from ..base import BaseClassifier


class TorchClassifier(BaseClassifier):
    """
    Simply wraps a PyTorch ``nn.Module``. Unlike `TimmClassifier` or
    `TorchvisionClassifier`, this class does not load a model from a zoo. The
    user is responsible for providing the model instance and, optionally, a
    preprocessing transform.
    """

    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        transform: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: The PyTorch model to use for classification.
            n_classes: Number of output classes.
            transform: Optional preprocessing transform. If
                ``None``, inputs must already be tensors of shape
                `(B, C, H, W)`.
            **kwargs: Extra arguments forwarded to
                `BaseClassifier`.
        """
        super().__init__(model=model, n_classes=n_classes, **kwargs)
        self.save_hyperparameters(ignore=["model", "transform"])
        self._transform = transform

    def _get_transform(self) -> Callable:
        """Return the preprocessing transform."""
        return self._transform or v2.Compose(
            [
                v2.Resize(400),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
