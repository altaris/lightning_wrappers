import logging

from torch import nn


def replace_head(
    module: nn.Module, head_name: str, n_classes: int
) -> nn.Module:
    """
    Replace the last linear layer of a model's head.

    The head is replaced with a new `nn.Linear` having
    ``n_classes`` output neurons. If the head already has the
    correct number of output neurons, it is left unchanged.

    Args:
        module: The model whose head should be replaced.
        head_name: Dot-separated submodule name, e.g.
            ``"classifier.1"``. Retrieved via
            `nn.Module.named_modules`.
        n_classes: The desired number of output neurons.

    Raises:
        RuntimeError: If the head module is not an
            `nn.Linear` layer.

    Returns:
        The modified module (in-place).
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
        parent[int(head_name.split(".")[-1])] = new_head
    else:
        setattr(parent, head_name.split(".")[-1], new_head)
    return module
