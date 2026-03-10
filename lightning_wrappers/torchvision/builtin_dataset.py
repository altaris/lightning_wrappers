"""See `BuiltinDataModule` documentation."""

import inspect
from typing import Any, Callable

import torch
import torchvision.datasets
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

from ..base import (
    DEFAULT_TEST_DATALOADER_KWARGS,
    DEFAULT_TRAIN_DATALOADER_KWARGS,
    DEFAULT_VAL_DATALOADER_KWARGS,
    BaseDataset,
)


def _resolve_dataset_cls(
    dataset_cls: str | type[VisionDataset],
) -> type[VisionDataset]:
    """
    Resolve a dataset class from a string name or pass through a
    class directly.

    Args:
        dataset_cls: A `torchvision.datasets` class or its name
            as a string (e.g. ``"Caltech101"``).
    """
    if isinstance(dataset_cls, str):
        cls = getattr(torchvision.datasets, dataset_cls, None)
        if cls is None or not (
            isinstance(cls, type) and issubclass(cls, VisionDataset)
        ):
            raise ValueError(f"Unknown torchvision dataset: {dataset_cls!r}")
        return cls  # type: ignore
    return dataset_cls


class BuiltinDataModule(BaseDataset):
    """
    A Lightning DataModule wrapping a `torchvision.datasets` class.

    The dataset can be specified as a class or by name::

        dm = TorchvisionDataModule("CIFAR10")
        dm = TorchvisionDataModule(CIFAR10)

    The split mechanism is auto-detected based on the dataset class
    constructor signature:

    - **`train` parameter** (e.g. `CIFAR10`, `MNIST`): the training
      set is created with ``train=True`` and the test set with
      ``train=False``.
    - **`split` parameter** (e.g. `ImageNet`): the training set is
      created with ``split="train"`` and the test set with
      ``split="test"``.
    - **No split parameter** (e.g. `Caltech101`): the full dataset
      is split randomly according to `val_ratio` and `test_ratio`.

    In all cases, a validation set is carved out of the training
    data using `val_ratio`.

    See also:
        `torchvision.datasets
        <https://docs.pytorch.org/vision/stable/datasets.html>`_
    """

    dataset_cls: Any
    dataset_kwargs: dict[str, Any]
    seed: int
    test_dataset: Dataset
    test_ratio: float
    train_dataset: Dataset
    val_dataset: Dataset
    val_ratio: float
    train_dataloader_kwargs: dict[str, Any]
    val_dataloader_kwargs: dict[str, Any]
    test_dataloader_kwargs: dict[str, Any]

    def __init__(
        self,
        dataset_cls: str | type[VisionDataset],
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 0,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        train_dataloader_kwargs: dict[str, Any] | None = None,
        val_dataloader_kwargs: dict[str, Any] | None = None,
        test_dataloader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            dataset_cls: A `torchvision.datasets` class or its
                name as a string (e.g. ``"Caltech101"``).
            val_ratio: Fraction of training data used for
                validation.
            test_ratio: Fraction of data used for testing. Only
                used when the dataset has no built-in split
                mechanism.
            seed: Random seed for reproducible splits.
            transform: Transform applied to images. Defaults to
                resize to 500x500 and convert to float32.
            target_transform: Transform applied to targets.
            dataset_kwargs: Extra keyword arguments forwarded
                to the dataset constructor (e.g.
                ``{"root": "/data"}``). These are merged into
                the defaults (``transform``,
                ``target_transform``, ``root``, ``download``).
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
        self.save_hyperparameters(
            ignore=["dataset_cls", "transform", "target_transform"]
        )

        self.dataset_cls = _resolve_dataset_cls(dataset_cls)
        self.dataset_kwargs = {
            "transform": (
                transform
                or v2.Compose(
                    [
                        v2.ToImage(),
                        v2.Resize((500, 500)),
                        v2.ToDtype(torch.float32, scale=True),
                    ]
                )
            ),
            "target_transform": target_transform,
            "root": "~/.torchvision/datasets",
            "download": True,
        }
        self.dataset_kwargs.update(dataset_kwargs or {})

        self.val_ratio, self.test_ratio = val_ratio, test_ratio

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

        self.seed = seed

    def _detect_split_key(self) -> str | None:
        """
        Inspect the dataset class signature to find a known split
        parameter.

        Returns:
            ``"train"``, ``"split"``, or ``None``.
        """
        sig = inspect.signature(self.dataset_cls)
        for key in ("train", "split"):
            if key in sig.parameters:
                return key
        return None

    def prepare_data(self) -> None:
        """
        Download the dataset if applicable.

        Called by Lightning on a single process before `setup`.
        """
        split_key = self._detect_split_key()
        if split_key == "train":
            self.dataset_cls(**self.dataset_kwargs, train=True)
            self.dataset_cls(**self.dataset_kwargs, train=False)
        elif split_key == "split":
            self.dataset_cls(**self.dataset_kwargs, split="train")
            self.dataset_cls(**self.dataset_kwargs, split="test")
        else:
            self.dataset_cls(**self.dataset_kwargs)

    def setup(self, stage: str | None = None) -> None:
        """
        Create train/val/test dataset splits.

        Sets `train_dataset`, `val_dataset`, and `test_dataset`.
        The ``stage`` argument is accepted for Lightning
        compatibility but ignored — all splits are always created.
        """
        split_key = self._detect_split_key()
        rng = torch.Generator().manual_seed(self.seed)
        if split_key == "train":
            full_train = self.dataset_cls(**self.dataset_kwargs, train=True)
            n_val = int(len(full_train) * self.val_ratio)
            n_train = len(full_train) - n_val
            self.train_dataset, self.val_dataset = random_split(
                full_train, [n_train, n_val], generator=rng
            )
            self.test_dataset = self.dataset_cls(
                **self.dataset_kwargs, train=False
            )
        elif split_key == "split":
            full_train = self.dataset_cls(**self.dataset_kwargs, split="train")
            n_val = int(len(full_train) * self.val_ratio)
            n_train = len(full_train) - n_val
            self.train_dataset, self.val_dataset = random_split(
                full_train, [n_train, n_val], generator=rng
            )
            self.test_dataset = self.dataset_cls(
                **self.dataset_kwargs, split="test"
            )
        else:
            full_ds = self.dataset_cls(**self.dataset_kwargs)
            n = len(full_ds)
            n_test, n_val = int(n * self.test_ratio), int(n * self.val_ratio)
            n_train = n - n_val - n_test
            a, b, c = random_split(
                full_ds, [n_train, n_val, n_test], generator=rng
            )
            self.train_dataset, self.val_dataset, self.test_dataset = a, b, c

    def test_dataloader(self) -> DataLoader:
        """Return the test `DataLoader`."""
        return DataLoader(self.test_dataset, **self.test_dataloader_kwargs)

    def train_dataloader(self) -> DataLoader:
        """Return the training `DataLoader`."""
        return DataLoader(self.train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return the validation `DataLoader`."""
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)
