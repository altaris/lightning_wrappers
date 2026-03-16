"""See `TorchvisionClassifier` documentation."""

from typing import Any, Callable

from torchvision.models import get_model, get_model_weights
from torchvision.transforms import v2

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
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_name: Name of the model architecture.
                See the `torchvision model zoo
                <https://docs.pytorch.org/vision/stable/models.html#classification>`_.
            n_classes: Number of output classes.
            head_name: Name of the classification head.
                If ``None``, the default head is used.
            weights: Weights to use for the model.
                Defaults to ``"DEFAULT"``.
        """
        model = get_model(
            model_name, weights=("DEFAULT" if pretrained else None)
        )
        super().__init__(
            model=model,
            n_classes=n_classes,
            head_name=head_name,
            **kwargs,
        )
        self.save_hyperparameters()

    def _get_transform(
        self, *args: Any, **kwargs: Any
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """
        Create an image processor from the model's weights.

        For example::

            TorchvisionClassifier.get_transform("alexnet")

        is analogous to::

            get_model_weights("alexnet")["DEFAULT"].transforms()
        """

        model_name: str = self.hparams.model_name
        weights: str = "DEFAULT" if self.hparams.pretrained else None  # type: ignore
        transform = v2.Compose(
            [
                v2.RGB(),
                get_model_weights(model_name)[weights].transforms(),
            ]
        )

        def _transform(batch: Any) -> Any:
            if isinstance(batch, dict):
                return {
                    k: (
                        [transform(img) for img in v]
                        # TODO: pass image_key from DS ↓
                        if k in ["img", "image", "jpg", "png"]
                        else v
                    )
                    for k, v in batch.items()
                }
            elif isinstance(batch, list):
                return [transform(img) for img in batch]
            else:
                return transform(batch)

        return _transform
