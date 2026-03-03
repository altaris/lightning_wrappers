"""See `TorchvisionClassifier` documentation."""

from typing import Any, Callable

import torchvision.transforms.v2 as transforms
from torchvision.models import get_model, get_model_weights

from ..base import BaseClassifier


class TorchvisionClassifier(BaseClassifier):
    """
    A torchvision classifier wrapped in a `LightningModule`.

    See also:
        - [`torchvision` documentation](https://docs.pytorch.org/vision/stable/index.html)
        - [`torchvision` model zoo](https://docs.pytorch.org/vision/stable/models.html#classification)
    """

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        head_name: str | None = None,
        weights: Any = "DEFAULT",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_name (str): Name of the model architecture. See the [`torchvision` model zoo](https://docs.pytorch.org/vision/stable/models.html#classification).
            n_classes (int) : Number of output classes.
            head_name (str | None, optional): Name of the classification head.
                If None, the default head is used.
            weights (Any, optional): Weights to use for the model. Defaults to
                "DEFAULT".
        """
        model = get_model(model_name, weights=weights)
        super().__init__(
            model=model, n_classes=n_classes, head_name=head_name, **kwargs
        )
        self.save_hyperparameters()

    def _get_transform(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """
        Creates an image processor based on the transform object of the model's
        chosen weights. For example,

            TorchvisionClassifier.get_transform("alexnet")

        is analogous to

            get_model_weights("alexnet")["DEFAULT"].transforms()
        """

        model_name: str = self.hparams.model_name  # type: ignore
        weights: str = self.hparams.weights  # type: ignore
        transform = transforms.Compose(
            [
                transforms.RGB(),
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
