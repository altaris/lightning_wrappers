"""
PyTorch Lightning LightningModule for timm image classifiers.

See also:
    - [`timm` documentation](https://huggingface.co/docs/timm/index)
    - [`timm` model zoo](https://huggingface.co/timm/models)
"""

from functools import lru_cache
from typing import Any, Callable

import lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from PIL import Image
from torchmetrics import Accuracy
from torchvision import transforms


class TimmClassifier(pl.LightningModule):
    """
    Lightning module wrapping a timm image classifier.

    Args:
        model_name: Name of the timm model to use ((e.g., `resnet18`,
            `timm/convnext_nano.r384_in12k`). See the [`timm` model
            zoo](https://huggingface.co/timm/models).
        num_classes: Number of output classes.
        pretrained: Whether to use a pretrained model.
        lr: Learning rate for optimizer.
    """

    model: nn.Module
    num_classes: int
    lr: float
    top1: Accuracy
    top5: Accuracy

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.num_classes = num_classes
        self.lr = lr
        task = "multiclass" if num_classes > 2 else "binary"
        self.top1 = Accuracy(task=task, num_classes=num_classes, top_k=1)  # type: ignore
        self.top5 = Accuracy(task=task, num_classes=num_classes, top_k=5)  # type: ignore

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer (AdamW)."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def forward(
        self, x: torch.Tensor | Image.Image | list[Image.Image]
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape `(B, C, H, W)` or `(C, H, W)`, a single PIL
                Image, or a list of PIL Images. In the latter case, all images
                are assumed to have the same size.

        Returns:
            Logits of shape `(B, num_classes)`. If a `(C, H, W)` tensor or a
            single image was passed, `B = 1`.
        """
        # TODO: implement stricter checks
        if isinstance(x, list):
            tr = self.get_transform()
            x = torch.stack([tr(img) for img in x])
        elif isinstance(x, Image.Image):
            tr = self.get_transform()
            x = tr(x)
            assert isinstance(x, torch.Tensor)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        return self.model(x)  # type: ignore

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

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], *_: Any, **__: Any
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = nnf.cross_entropy(logits, y)
        self.log_dict(
            {
                "test/loss": loss,
                "test/top1": self.top1(logits, y),
                "test/top5": self.top5(logits, y),
            },
            prog_bar=True,
            sync_dist=True,
        )

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], *_: Any, **__: Any
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = nnf.cross_entropy(logits, y)
        self.log_dict(
            {
                "train/loss": loss,
                "train/top1": self.top1(logits, y),
                "train/top5": self.top5(logits, y),
            },
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], *_: Any, **__: Any
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = nnf.cross_entropy(logits, y)
        self.log_dict(
            {
                "val/loss": loss,
                "val/top1": self.top1(logits, y),
                "val/top5": self.top5(logits, y),
            },
            prog_bar=True,
            sync_dist=True,
        )
