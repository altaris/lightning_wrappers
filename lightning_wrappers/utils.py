import logging

from torch import nn


def replace_head(
    module: nn.Module, head_name: str, n_classes: int
) -> nn.Module:
    """
    Replaces the last linear layer of a model with a new one with a specified
    number of output neurons (which is not necessarily different from the old
    head's). *However*, if the model's head already has the correct number of
    output neurons, then it is not replaced.

    Args:
        module (nn.Module):
        head_name (str): e.g. `model.classifier.1`. The
            name of a submodule can be retried by inspecting the output of
            `nn.Module.named_modules`.
        n_classes (int): The desired number of output neurons.

    Raises:
        RuntimeError: If the head module is not a
            [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
            module

    Returns:
        The modified module, which is the one passed in argument since the
        modification is performed in-place.
    """
    head = module.get_submodule(head_name)
    if not isinstance(head, nn.Linear):
        raise RuntimeError(
            f"Model head '{head_name}' must have type nn.Linear"
        )
    if head.out_features == n_classes:
        logging.warning(
            "Module head '%s' already has %d output neurons. Not replacing it",
            head_name,
            n_classes,
        )
        return module
    new_head = nn.Linear(
        in_features=head.in_features,
        out_features=n_classes,
        bias=head.bias is not None,
    )
    parent = module.get_submodule(".".join(head_name.split(".")[:-1]))
    if isinstance(parent, nn.Sequential):
        parent[-1] = new_head
    else:
        setattr(parent, head_name.split(".")[-1], new_head)
    return module
