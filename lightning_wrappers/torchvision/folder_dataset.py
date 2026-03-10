"""See `ImageFolderDataModule` documentation."""

from pathlib import Path
from typing import Any, Callable

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from ..base import (
    DEFAULT_TEST_DATALOADER_KWARGS,
    DEFAULT_TRAIN_DATALOADER_KWARGS,
    DEFAULT_VAL_DATALOADER_KWARGS,
    BaseDataset,
)


class ImageFolderDataModule(BaseDataset):
    """
    A Lightning DataModule for datasets in ImageFolder layout with
    pre-existing splits::

        root/
        ├── train/
        │   ├── class_a/
        │   │   ├── 001.jpg
        │   │   └── ...
        │   └── class_b/
        │       └── ...
        ├── val/
        │   └── ...
        └── test/
            └── ...

    Each split directory is loaded as a
    `torchvision.datasets.ImageFolder`.

    Example::

        dm = ImageFolderDataModule("/data/imagenet")
    """

    train_dataset: ImageFolder
    val_dataset: ImageFolder
    test_dataset: ImageFolder
    train_dataloader_kwargs: dict[str, Any]
    val_dataloader_kwargs: dict[str, Any]
    test_dataloader_kwargs: dict[str, Any]

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        train_dir: str = "train",
        val_dir: str = "val",
        test_dir: str = "test",
        train_dataloader_kwargs: dict[str, Any] | None = None,
        val_dataloader_kwargs: dict[str, Any] | None = None,
        test_dataloader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            root: Path to the dataset root directory containing
                the split subdirectories.
            transform: Transform applied to images.
            target_transform: Transform applied to targets.
            train_dir: Name of the training split subdirectory.
            val_dir: Name of the validation split subdirectory.
            test_dir: Name of the test split subdirectory.
            train_dataloader_kwargs: Overrides for the training
                `DataLoader`. Merged into
                `DEFAULT_TRAIN_DATALOADER_KWARGS`.
            val_dataloader_kwargs: Overrides for the validation
                `DataLoader`. Merged into
                `DEFAULT_VAL_DATALOADER_KWARGS`.
            test_dataloader_kwargs: Overrides for the test
                `DataLoader`. Merged into
                `DEFAULT_TEST_DATALOADER_KWARGS`.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["transform", "target_transform"])

        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

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

    def setup(self, stage: str | None = None) -> None:
        """
        Create train/val/test datasets from the split
        subdirectories.

        The ``stage`` argument is accepted for Lightning
        compatibility but ignored — all splits are always created.
        """
        kwargs: dict[str, Any] = {
            "transform": self.transform,
            "target_transform": self.target_transform,
        }
        self.train_dataset = ImageFolder(self.root / self.train_dir, **kwargs)
        self.val_dataset = ImageFolder(self.root / self.val_dir, **kwargs)
        self.test_dataset = ImageFolder(self.root / self.test_dir, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Return the training `DataLoader`."""
        return DataLoader(self.train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return the validation `DataLoader`."""
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        """Return the test `DataLoader`."""
        return DataLoader(self.test_dataset, **self.test_dataloader_kwargs)
