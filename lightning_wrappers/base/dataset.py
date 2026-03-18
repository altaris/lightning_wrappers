"""See `BaseDataset` documentation."""

import abc
from typing import Any

import lightning as pl
from torch.utils.data import DataLoader, Dataset

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


class BaseDataset(pl.LightningDataModule):
    """
    Base Lightning DataModule providing common dataloader kwargs
    handling and dataloader methods.

    Subclasses must set `train_dataset`, `val_dataset`, and
    `test_dataset` in their `setup` method.
    """

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

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        """
        Return the number of classes in the dataset.

        Warning:
            Will call `setup("train")`
        """

    def train_dataloader(self) -> DataLoader:
        """Return the training `DataLoader`."""
        return DataLoader(self.train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return the validation `DataLoader`."""
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        """Return the test `DataLoader`."""
        return DataLoader(self.test_dataset, **self.test_dataloader_kwargs)
