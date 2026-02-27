from typing import Any, Callable

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from PIL import Image
from torchmetrics import Accuracy

from ..utils import replace_head


class BaseClassifier(pl.LightningModule):
    model: nn.Module
    lr: float
    top1: Accuracy
    top5: Accuracy

    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        head_name: str | None = None,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        if head_name is not None:
            self.model = replace_head(self.model, head_name, n_classes)
        task = "multiclass" if n_classes > 2 else "binary"
        self.top1 = Accuracy(task=task, num_classes=n_classes, top_k=1)  # type: ignore
        self.top5 = Accuracy(task=task, num_classes=n_classes, top_k=5)  # type: ignore

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer (AdamW)."""
        lr: float = self.hparams.lr  # type: ignore
        return torch.optim.AdamW(self.parameters(), lr=lr)

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

    def get_transform(self, *args: Any, **kwargs: Any) -> Callable:
        return lambda x: x

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
