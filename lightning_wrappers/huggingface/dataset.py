"""See `HuggingFaceDataset` documentation."""

from pathlib import Path
from typing import Any, Callable

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Image,
    concatenate_datasets,
    load_dataset,
)

from ..base import BaseDataset

DEFAULT_CACHE_DIR = Path.home() / ".huggingface" / "datasets"
"""
Default download path for huggingface dataset.

See also:
    https://huggingface.co/docs/datasets/cache
"""


class HuggingFaceDataset(BaseDataset):
    """
    A Lightning DataModule wrapping a HuggingFace ``datasets``
    dataset.

    The dataset is loaded via `datasets.load_dataset` and split
    handling is auto-detected:

    - If the loaded dataset is a `DatasetDict` with ``"train"``
      and ``"test"`` splits, they are used directly. A validation
      set is carved out of the training data using `val_ratio`.
    - If the loaded dataset is a `DatasetDict` with a
      ``"validation"`` split, it is used as-is instead of
      carving from the training set.
    - If the loaded dataset is a single `Dataset` (no splits),
      it is split randomly according to `val_ratio` and
      `test_ratio`.

    Example::

        dm = HuggingFaceDataset("cifar10")
        dm = HuggingFaceDataset(
            "imagenet-1k",
            load_dataset_kwargs={"trust_remote_code": True},
        )

    See also:
        `HuggingFace datasets
        <https://huggingface.co/docs/datasets/>`_
    """

    seed: int
    test_dataset: Dataset
    test_ratio: float
    train_dataset: Dataset
    val_dataset: Dataset
    val_ratio: float

    def __init__(
        self,
        path: str,
        name: str | None = None,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 0,
        transform: Callable | None = None,
        load_dataset_kwargs: dict[str, Any] | None = None,
        train_dataloader_kwargs: dict[str, Any] | None = None,
        val_dataloader_kwargs: dict[str, Any] | None = None,
        test_dataloader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            path: Path or name of the dataset (e.g.
                ``"cifar10"``, ``"imagenet-1k"``), as accepted
                by `datasets.load_dataset`.
            name: Name of the dataset configuration, if any
                (e.g. ``"plain_text"`` for ``"wikitext"``).
            val_ratio: Fraction of training data used for
                validation. Ignored if the dataset already has a
                ``"validation"`` split.
            test_ratio: Fraction of data used for testing. Only
                used when the dataset has no built-in splits.
            seed: Random seed for reproducible splits.
            transform: An optional callable applied to each
                sample dict via `Dataset.set_transform`.
            load_dataset_kwargs: Extra keyword arguments
                forwarded to `datasets.load_dataset`.
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
        super().__init__(
            train_dataloader_kwargs=train_dataloader_kwargs,
            val_dataloader_kwargs=val_dataloader_kwargs,
            test_dataloader_kwargs=test_dataloader_kwargs,
        )

        self.path, self.name = path, name
        self.val_ratio, self.test_ratio = val_ratio, test_ratio
        self.seed = seed
        self.transform = transform
        self.load_dataset_kwargs = load_dataset_kwargs or {}
        self.load_dataset_kwargs.setdefault("cache_dir", DEFAULT_CACHE_DIR)

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        self.setup("train")
        for f in self.train_dataset.features.values():
            if isinstance(f, ClassLabel):
                return int(f.num_classes)
        raise ValueError(f"Dataset {self.path!r} has no ClassLabel feature.")

    def _check_image_classification(self, features: Features) -> None:
        """
        Validate that the dataset features contain at least one
        `Image` column and one `ClassLabel` column.

        Raises:
            ValueError: If the required feature types are missing.
        """
        has_image = any(isinstance(f, Image) for f in features.values())
        has_label = any(isinstance(f, ClassLabel) for f in features.values())
        missing: list[str] = []
        if not has_image:
            missing.append("Image")
        if not has_label:
            missing.append("ClassLabel")
        if missing:
            raise ValueError(
                f"Dataset {self.path!r} does not look like an"
                f" image classification dataset: missing"
                f" {', '.join(missing)} feature(s)."
                f" Available features: {dict(features)}"
            )

    def _wrap(self, ds: Dataset) -> Dataset:
        """
        Apply `transform` to a dataset by setting its
        ``set_transform`` if a transform is provided.
        """
        if self.transform is not None:
            ds.set_transform(self.transform)
        return ds

    def _hf_split(self, ds: Dataset, ratio: float) -> tuple[Dataset, Dataset]:
        """
        Split a HuggingFace `Dataset` using its native `train_test_split`.

        Returns:
            A tuple ``(main_part, split_part)``.
        """
        parts = ds.train_test_split(test_size=ratio, seed=self.seed)
        return parts["train"], parts["test"]

    def setup(self, stage: str | None = None) -> None:
        """
        Create train/val/test dataset splits.

        The ``stage`` argument is accepted for Lightning compatibility but
        ignored — all splits are always created.
        """
        raw = load_dataset(self.path, self.name, **self.load_dataset_kwargs)
        features = (
            next(iter(raw.values())).features
            if isinstance(raw, DatasetDict)
            else raw.features
        )
        self._check_image_classification(features)

        if isinstance(raw, DatasetDict):
            if train_ds := raw.get("train"):
                if "validation" in raw:  # Use existing val. split if available
                    self.train_dataset = self._wrap(train_ds)
                    self.val_dataset = self._wrap(raw["validation"])
                else:
                    main, val = self._hf_split(train_ds, self.val_ratio)
                    self.train_dataset = self._wrap(main)
                    self.val_dataset = self._wrap(val)
                if "test" in raw:  # Use existing test split if available
                    self.test_dataset = self._wrap(raw["test"])
                else:
                    self.test_dataset = self.val_dataset
            else:
                # DatasetDict without a "train" key: merge all
                # splits and split randomly
                all_ds = concatenate_datasets(list(raw.values()))
                self._split_single(all_ds)
        else:  # isinstance(raw, Dataset)
            self._split_single(raw)

    def _split_single(self, ds: Dataset) -> None:
        """
        Split a single `Dataset` into train/val/test using
        `val_ratio` and `test_ratio`.
        """
        main, test = self._hf_split(ds, self.test_ratio)
        # val_ratio is relative to the original size, adjust
        # for the remaining data
        adjusted_val = self.val_ratio / (1.0 - self.test_ratio)
        train, val = self._hf_split(main, adjusted_val)
        self.train_dataset = self._wrap(train)
        self.val_dataset = self._wrap(val)
        self.test_dataset = self._wrap(test)
