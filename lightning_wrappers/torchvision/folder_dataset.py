"""See `ImageFolderDataModule` documentation."""

from pathlib import Path
from typing import Any, Callable

from torchvision.datasets import ImageFolder

from ..base import BaseDataModule


class ImageFolderDataModule(BaseDataModule):
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

    def __init__(
        self,
        root: str | Path,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        test_transform: Callable | None = None,
        train_dir: str = "train",
        val_dir: str = "val",
        test_dir: str = "test",
        train_dataloader_kwargs: dict[str, Any] | None = None,
        val_dataloader_kwargs: dict[str, Any] | None = None,
        test_dataloader_kwargs: dict[str, Any] | None = None,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
    ) -> None:
        """
        Args:
            root: Path to the dataset root directory containing
                the split subdirectories.
            train_transform: Transform applied to training
                images.
            val_transform: Transform applied to validation
                images.
            test_transform: Transform applied to test images.
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
            mixup_alpha: Alpha parameter for `MixUp`
                augmentation. ``0.0`` disables it.
            cutmix_alpha: Alpha parameter for `CutMix`
                augmentation. ``0.0`` disables it.
        """
        super().__init__(
            train_dataloader_kwargs=train_dataloader_kwargs,
            val_dataloader_kwargs=val_dataloader_kwargs,
            test_dataloader_kwargs=test_dataloader_kwargs,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
        )

        self.root = Path(root)
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        self.setup("train")
        return len(self.train_dataset.classes)

    def setup(self, stage: str | None = None) -> None:
        """
        Create train/val/test datasets from the split
        subdirectories.

        The ``stage`` argument is accepted for Lightning
        compatibility but ignored — all splits are always created.
        """
        self.train_dataset = ImageFolder(
            self.root / self.train_dir,
            transform=self.train_transform,
        )
        self.val_dataset = ImageFolder(
            self.root / self.val_dir,
            transform=self.val_transform,
        )
        self.test_dataset = ImageFolder(
            self.root / self.test_dir,
            transform=self.test_transform,
        )
