"""
PyTorch Lightning LightningModule for timm image classifiers.

See also:
    - [`timm` documentation](https://huggingface.co/docs/timm/index)
    - [`timm` model zoo](https://huggingface.co/timm/models)
"""

from typing import Any, Callable, Literal

import timm
import torch
from timm.optim import create_optimizer_v2, list_optimizers
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
        optimizer: str = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: Literal[
            "cosine", "multistep", "plateau", "poly", "step", "tanh"
        ]
        | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_name: Name of the timm model architecture.
            n_classes: Number of output classes.
            pretrained: Whether to load pretrained weights.
            optimizer: Optimizer to use. See `timm.optim.list_optimizers`
                for supported optimizers.
            optimizer_kwargs: Optional keyword arguments to pass to the
                optimizer constructor. See `timm.optim.create_optimizer_v2`
                for supported arguments. Note that `lr` can be passed directly
                to the constructor.
            scheduler: Optional learning rate scheduler to use. If ``None``, no
                scheduler is used. See `timm.scheduler.create_scheduler_v2`
                for supported schedulers.
            scheduler_kwargs: Optional keyword arguments to pass to the
                scheduler constructor. See `timm.scheduler.create_scheduler_v2`
                for supported arguments. Ignored if `scheduler` is ``None``.
            **kwargs: Extra arguments forwarded to
                `BaseClassifier`.
        """
        if optimizer not in list_optimizers():
            raise ValueError(
                f"Unsupported optimizer '{optimizer}'. Supported optimizers: "
                f"{list_optimizers()}"
            )
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
        from timm.data import create_transform, resolve_data_config

        try:
            data_cfg = resolve_data_config(self.model.pretrained_cfg)
            transform = create_transform(**data_cfg)
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

    def lr_scheduler_step(
        self,
        scheduler: Any,
        metric: float | None = None,
    ) -> None:
        """
        Step the LR scheduler.

        Handles both standard PyTorch ``LRScheduler`` and timm's
        non-standard schedulers (which expect ``epoch`` and
        ``metric`` keyword arguments).
        """
        if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)
        else:
            scheduler.step(epoch=self.current_epoch, metric=metric)

    def configure_optimizers(self) -> Any:
        """Configure AdamW optimizer, optionally with cosine schedule."""
        optimizer = create_optimizer_v2(
            self,
            opt=self.hparams.optimizer,
            lr=self.hparams.lr,
            **(self.hparams.optimizer_kwargs or {}),
        )
        if self.hparams.scheduler:
            from timm.scheduler import create_scheduler_v2

            kw = {
                "num_epochs": (
                    self.trainer.max_epochs
                    if hasattr(self, "trainer")
                    else 300
                ),
                "warmup_epochs": 2,
            }
            kw.update(self.hparams.scheduler_kwargs or {})
            scheduler, _ = create_scheduler_v2(
                optimizer=optimizer, sched=self.hparams.scheduler, **kw
            )
            return [optimizer], [scheduler]
        else:
            return optimizer
