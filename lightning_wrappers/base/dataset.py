"""See `BaseDataModule` documentation."""

import abc
from typing import Any

import lightning as pl
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.transforms import v2

DEFAULT_TRAIN_DATALOADER_KWARGS = {
    "shuffle": True,
    "batch_size": 64,
    "num_workers": 2,
    "pin_memory": True,
}
DEFAULT_VAL_DATALOADER_KWARGS = {
    "batch_size": 64,
    "num_workers": 2,
}
DEFAULT_TEST_DATALOADER_KWARGS = {
    "batch_size": 64,
    "num_workers": 2,
}


class BaseDataModule(pl.LightningDataModule):
    """
    Base Lightning DataModule providing common dataloader kwargs
    handling and dataloader methods.

    Subclasses must set `train_dataset`, `val_dataset`, and
    `test_dataset` in their `setup` method.
    """

    cutmix_alpha: float
    mixup_alpha: float
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    train_dataloader_kwargs: dict[str, Any]
    val_dataloader_kwargs: dict[str, Any]
    test_dataloader_kwargs: dict[str, Any]

    def __init__(
        self,
        train_dataloader_kwargs: dict[str, Any] | None = None,
        val_dataloader_kwargs: dict[str, Any] | None = None,
        test_dataloader_kwargs: dict[str, Any] | None = None,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
    ) -> None:
        """
        Args:
            train_dataloader_kwargs: Overrides for the training
                `DataLoader`. Merged into
                `DEFAULT_TRAIN_DATALOADER_KWARGS`.
            val_dataloader_kwargs: Overrides for the validation
                `DataLoader`. Merged into
                `DEFAULT_VAL_DATALOADER_KWARGS`.
            test_dataloader_kwargs: Overrides for the test
                `DataLoader`. Merged into
                `DEFAULT_TEST_DATALOADER_KWARGS`.
            mixup_alpha: Alpha parameter for `MixUp`
                augmentation on the training dataloader. Set to
                ``0.0`` (default) to disable.
            cutmix_alpha: Alpha parameter for `CutMix`
                augmentation on the training dataloader. Set to
                ``0.0`` (default) to disable. If both
                `mixup_alpha` and `cutmix_alpha` are non-zero,
                one is chosen at random for each batch via
                `RandomChoice`.
        """
        super().__init__()
        self.train_dataloader_kwargs = {
            **DEFAULT_TRAIN_DATALOADER_KWARGS,
            **(train_dataloader_kwargs or {}),
        }
        self.val_dataloader_kwargs = {
            **DEFAULT_VAL_DATALOADER_KWARGS,
            **(val_dataloader_kwargs or {}),
        }
        self.test_dataloader_kwargs = {
            **DEFAULT_TEST_DATALOADER_KWARGS,
            **(test_dataloader_kwargs or {}),
        }
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        """
        Return the number of classes in the dataset.

        Warning:
            Will call `setup("train")`
        """

    def _build_collate_transform(self) -> v2.Transform | None:
        """
        Build a batch-level augmentation transform from
        `mixup_alpha` and `cutmix_alpha`.

        Returns:
            A `MixUp`, `CutMix`, or `RandomChoice` transform,
            or ``None`` if both are disabled.
        """
        nc = self.num_classes
        transforms: list[v2.Transform] = []
        if self.mixup_alpha > 0.0:
            transforms.append(v2.MixUp(alpha=self.mixup_alpha, num_classes=nc))
        if self.cutmix_alpha > 0.0:
            transforms.append(
                v2.CutMix(alpha=self.cutmix_alpha, num_classes=nc)
            )
        if not transforms:
            return None
        if len(transforms) == 1:
            return transforms[0]
        return v2.RandomChoice(transforms)

    def train_dataloader(self) -> DataLoader:
        """Return the training `DataLoader`."""
        kwargs = dict(self.train_dataloader_kwargs)
        if (aug := self._build_collate_transform()) is not None:
            kwargs["collate_fn"] = lambda batch: aug(*default_collate(batch))
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return the validation `DataLoader`."""
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        """Return the test `DataLoader`."""
        return DataLoader(self.test_dataset, **self.test_dataloader_kwargs)
