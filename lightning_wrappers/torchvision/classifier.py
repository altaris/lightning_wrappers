"""See `TorchvisionClassifier` documentation."""

from functools import lru_cache
from typing import Any, Callable

import lightning as pl
from torch import nn
from torchvision.models import get_model, get_model_weights
from torchvision.transforms import v2 as tr


class TorchvisionClassifier(pl.LightningModule):
    """
    A torchvision classifier wrapped in a `LightningModule`.

    See also:
        - [`torchvision` documentation](https://docs.pytorch.org/vision/stable/index.html)
        - [`torchvision` model zoo](https://docs.pytorch.org/vision/stable/models.html#classification)
    """

    model: nn.Module

    def __init__(
        self,
        model_name: str,
        n_classes: int | None = None,
        head_name: str | None = None,
        weights: Any = "DEFAULT",
    ) -> None:
        """
        Args:
            model_name (str): Name of the model architecture. See the [`torchvision` model zoo](https://docs.pytorch.org/vision/stable/models.html#classification).
            n_classes (int | None, optional): Number of output classes. If left
                to `None`, the default number of output classes of the
                pretrained model is used, and the classification head is not
                replaced.
            head_name (str | None, optional): Name of the classification head.
                If None, the default head is used.
            weights (Any, optional): Weights to use for the model. Defaults to
                "DEFAULT".
        """
        self.model = get_model(model_name, weights=weights)
        self.save_hyperparameters()

    @lru_cache(maxsize=1)
    def get_transform(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """
        Creates an image processor based on the transform object of the model's
        chosen weights. For example,

            TorchvisionClassifier.get_image_processor("alexnet")

        is analogous to

            get_model_weights("alexnet")["DEFAULT"].transforms()
        """

        model_name: str = self.hparams.model_name  # type: ignore
        weights: str = self.hparams.weights  # type: ignore
        transform = tr.Compose(
            [
                tr.RGB(),
                get_model_weights(model_name)[weights].transforms(),
            ]
        )

        def _transform(batch: dict[str, Any]) -> dict[str, Any]:
            return {
                k: (
                    [transform(img) for img in v]
                    # TODO: pass image_key from DS ↓
                    if k in ["img", "image", "jpg", "png"]
                    else v
                )
                for k, v in batch.items()
            }

        return _transform
