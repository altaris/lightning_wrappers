import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from PIL import Image
from torchmetrics import Accuracy

from ..utils import replace_head


class BaseClassifier(ABC, pl.LightningModule):
    """
    Abstract base class for image classifiers.

    Subclasses must implement `_get_transform` to provide the
    preprocessing pipeline for the model.
    """

    model: nn.Module
    train_top1: Accuracy
    train_top5: Accuracy | None
    val_top1: Accuracy
    val_top5: Accuracy | None
    test_top1: Accuracy
    test_top5: Accuracy | None
    _transform: Callable | None = None

    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        head_name: str | None = None,
        lr: float = 1e-3,
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: The backbone model.
            n_classes: Number of output classes.
            head_name: Dot-separated name of the classification
                head submodule to replace. If ``None``, the head
                is left unchanged.
            lr: Learning rate for the optimizer.
            label_smoothing: Label smoothing factor for the loss.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", *kwargs.keys()])
        for k in kwargs:
            logging.warning("Unknown argument '%s'", k)

        self.model = model
        if head_name is not None:
            self.model = replace_head(self.model, head_name, n_classes)

        task = "multiclass" if n_classes > 2 else "binary"
        ak: dict[str, Any] = {"task": task, "num_classes": n_classes}
        self.test_top1 = Accuracy(**ak, top_k=1)
        self.test_top5 = Accuracy(**ak, top_k=5) if n_classes > 5 else None
        self.train_top1 = Accuracy(**ak, top_k=1)
        self.train_top5 = Accuracy(**ak, top_k=5) if n_classes > 5 else None
        self.val_top1 = Accuracy(**ak, top_k=1)
        self.val_top5 = Accuracy(**ak, top_k=5) if n_classes > 5 else None

    @abstractmethod
    def _get_transform(self, *args: Any, **kwargs: Any) -> Callable:
        """Return the preprocessing transform for this model."""
        ...

    def _step(
        self,
        stage: str,
        top1: Accuracy,
        top5: Accuracy | None,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Shared train/val/test step.

        Args:
            prefix: Metric prefix (``"train"``, ``"val"``, or
                ``"test"``).
            top1: Top-1 accuracy metric object.
            top5: Optional top-5 accuracy metric object.
            batch: A ``(images, labels)`` tuple.
        """
        x, y = batch
        logits = self(x)
        loss = nnf.cross_entropy(
            logits, y, label_smoothing=self.hparams.label_smoothing
        )
        d: dict[str, torch.Tensor] = {f"{stage}/loss": loss}
        if not y.is_floating_point():
            d[f"{stage}/top1"] = top1(logits, y)
            if top5 is not None:
                d[f"{stage}/top5"] = top5(logits, y)
        self.log_dict(d, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer (AdamW)."""
        lr: float = self.hparams.lr  # type: ignore
        return torch.optim.AdamW(self.parameters(), lr=lr)

    def forward(
        self, x: torch.Tensor | Image.Image | list | dict
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: One of:
                - A tensor of shape `(B, C, H, W)` or `(C, H, W)`,
                - A single PIL Image,
                - A list of PIL Images (assumed same size),
                - A dict with an image key (processed by `get_transform`).

        Returns:
            Logits of shape `(B, num_classes)`. If a `(C, H, W)` tensor or a
            single image was passed, `B = 1`.
        """
        if isinstance(x, dict):
            tr = self.get_transform()
            x = tr(x)
            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    f"Transform returned {type(x).__name__}, expected Tensor"
                )
        elif isinstance(x, list):
            if not x:
                raise ValueError("Empty image list")
            if not isinstance(x[0], Image.Image):
                raise TypeError(
                    "List elements must be PIL Images, got"
                    f" {type(x[0]).__name__}"
                )
            tr = self.get_transform()
            x = torch.stack([tr(img) for img in x])
        elif isinstance(x, Image.Image):
            tr = self.get_transform()
            x = tr(x)
            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    f"Transform returned {type(x).__name__}, expected Tensor"
                )
        elif not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Expected Tensor, PIL Image, list, or dict,"
                f" got {type(x).__name__}"
            )
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.ndim}D")
        x = x.to(self.device)
        return self.model(x)

    def get_transform(self, *args: Any, **kwargs: Any) -> Callable:
        """
        Return the cached preprocessing transform.

        On first call, delegates to `_get_transform` and caches
        the result.
        """
        if self._transform is None:
            self._transform = self._get_transform(*args, **kwargs)
        return self._transform

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        *_: Any,
        **__: Any,
    ) -> None:
        """Run a single test step."""
        self._step("test", self.test_top1, self.test_top5, batch)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        *_: Any,
        **__: Any,
    ) -> torch.Tensor:
        """Run a single training step."""
        return self._step("train", self.train_top1, self.train_top5, batch)

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        *_: Any,
        **__: Any,
    ) -> None:
        """Run a single validation step."""
        self._step("val", self.val_top1, self.val_top5, batch)
