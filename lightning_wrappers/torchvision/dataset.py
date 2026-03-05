"""See `TorchvisionDataModule` documentation."""

import inspect
from typing import Any, Callable

import torch
import torchvision.datasets
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

from ..base.dataset import BaseDataset


def _resolve_dataset_class(
    dataset_class: str | type[VisionDataset],
) -> type[VisionDataset]:
    """
    Resolve a dataset class from a string name or pass through a
    class directly.

    Args:
        dataset_class: A `torchvision.datasets` class or its name
            as a string (e.g. ``"Caltech101"``).
    """
    if isinstance(dataset_class, str):
        cls = getattr(torchvision.datasets, dataset_class, None)
        if cls is None or not (
            isinstance(cls, type) and issubclass(cls, VisionDataset)
        ):
            raise ValueError(f"Unknown torchvision dataset: {dataset_class!r}")
        return cls  # type: ignore
    return dataset_class


class TorchvisionDataModule(BaseDataset):
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

    batch_size: int
    dataset_class: Any
    dataset_kwargs: dict[str, Any]
    num_workers: int
    seed: int
    target_transform: Callable | None
    test_dataset: Dataset
    test_ratio: float
    train_dataset: Dataset
    transform: Callable | None
    val_dataset: Dataset
    val_ratio: float

    def __init__(
        self,
        dataset_class: str | type[VisionDataset],
        batch_size: int = 32,
        num_workers: int = 0,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 0,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            dataset_class: A `torchvision.datasets` class or its
                name as a string (e.g. ``"Caltech101"``).
            batch_size: Batch size for all dataloaders.
            num_workers: Number of dataloader workers.
            val_ratio: Fraction of training data used for
                validation.
            test_ratio: Fraction of data used for testing. Only
                used when the dataset has no built-in split
                mechanism.
            seed: Random seed for reproducible splits.
            transform: Transform applied to images.
            target_transform: Transform applied to targets.
            **dataset_kwargs: Extra keyword arguments forwarded to
                the dataset constructor (e.g. ``download=True``).
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=["dataset_class", "transform", "target_transform"]
        )
        self.dataset_class = _resolve_dataset_class(dataset_class)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_kwargs = kwargs

    def _common_kwargs(self) -> dict[str, Any]:
        return {
            "transform": (
                self.transform
                or v2.Compose(
                    [
                        v2.ToImage(),
                        v2.Resize((500, 500)),
                        v2.ToDtype(torch.float32, scale=True),
                    ]
                )
            ),
            "target_transform": self.target_transform,
            "root": "~/.torchvision/datasets",
            "download": True,
            **self.dataset_kwargs,
        }

    def _detect_split_key(self) -> str | None:
        """
        Inspect the dataset class signature to find a known split
        parameter.

        Returns:
            ``"train"``, ``"split"``, or ``None``.
        """
        sig = inspect.signature(self.dataset_class)
        for key in ("train", "split"):
            if key in sig.parameters:
                return key
        return None

    def prepare_data(self) -> None:
        """Download the dataset if applicable."""
        split_key = self._detect_split_key()
        kwargs = self._common_kwargs()
        if split_key == "train":
            self.dataset_class(**kwargs, train=True)
            self.dataset_class(**kwargs, train=False)
        elif split_key == "split":
            self.dataset_class(**kwargs, split="train")
            self.dataset_class(**kwargs, split="test")
        else:
            self.dataset_class(**kwargs)

    def setup(self, stage: str | None = None) -> None:
        """
        Instantiates all dataset splits. The `stage` argument is ignored.
        """
        split_key = self._detect_split_key()
        kwargs = self._common_kwargs()
        rng = torch.Generator().manual_seed(self.seed)
        if split_key == "train":
            full_train = self.dataset_class(**kwargs, train=True)
            n_val = int(len(full_train) * self.val_ratio)
            n_train = len(full_train) - n_val
            self.train_dataset, self.val_dataset = random_split(
                full_train, [n_train, n_val], generator=rng
            )
            self.test_dataset = self.dataset_class(**kwargs, train=False)
        elif split_key == "split":
            full_train = self.dataset_class(**kwargs, split="train")
            n_val = int(len(full_train) * self.val_ratio)
            n_train = len(full_train) - n_val
            self.train_dataset, self.val_dataset = random_split(
                full_train, [n_train, n_val], generator=rng
            )
            self.test_dataset = self.dataset_class(**kwargs, split="test")
        else:
            full_ds = self.dataset_class(**kwargs)
            n = len(full_ds)
            n_test = int(n * self.test_ratio)
            n_val = int(n * self.val_ratio)
            n_train = n - n_val - n_test
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = random_split(
                full_ds,
                [n_train, n_val, n_test],
                generator=rng,
            )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
