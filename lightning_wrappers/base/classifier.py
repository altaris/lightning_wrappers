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
    train_top1: Accuracy
    train_top5: Accuracy | None
    val_top1: Accuracy
    val_top5: Accuracy | None
    test_top1: Accuracy
    test_top5: Accuracy | None

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
        ak: dict[str, Any] = {"task": task, "num_classes": n_classes}
        self.train_top1 = Accuracy(**ak, top_k=1)
        self.val_top1 = Accuracy(**ak, top_k=1)
        self.test_top1 = Accuracy(**ak, top_k=1)
        self.train_top5 = Accuracy(**ak, top_k=5) if n_classes > 5 else None
        self.val_top5 = Accuracy(**ak, top_k=5) if n_classes > 5 else None
        self.test_top5 = Accuracy(**ak, top_k=5) if n_classes > 5 else None

    def _step(
        self,
        prefix: str,
        top1: Accuracy,
        top5: Accuracy | None,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = nnf.cross_entropy(logits, y)
        d: dict[str, torch.Tensor] = {
            f"{prefix}/loss": loss,
            f"{prefix}/top1": top1(logits, y),
        }
        if top5 is not None:
            d[f"{prefix}/top5"] = top5(logits, y)
        self.log_dict(d, prog_bar=True, sync_dist=True)
        return loss

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
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        *_: Any,
        **__: Any,
    ) -> None:
        self._step("test", self.test_top1, self.test_top5, batch)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        *_: Any,
        **__: Any,
    ) -> torch.Tensor:
        return self._step("train", self.train_top1, self.train_top5, batch)

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        *_: Any,
        **__: Any,
    ) -> None:
        self._step("val", self.val_top1, self.val_top5, batch)
