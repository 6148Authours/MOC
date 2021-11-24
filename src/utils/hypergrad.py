import torch
from torch import Tensor
from typing import List, Callable


# noinspection PyUnusedLocal
def reverse_unroll(params: List[Tensor],
                   hparams: List[Tensor],
                   outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
                   set_grad=True) -> List[Tensor]:
    """
    Computes the hypergradient by backpropagating through a previously employed inner solver procedure.

    Args:
        params: the output of a torch differentiable inner solver (it must depend on hparams in the torch graph)
        hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
        outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
        set_grad: if True set t.grad to the hypergradient for every t in hparams

    Returns:
        the list of hypergradients for each element in hparams
    """
    o_loss = outer_loss(params, hparams)
    grads = torch.autograd.grad(o_loss, hparams, retain_graph=True, allow_unused=True)
    if set_grad:
        update_tensor_grads(hparams, grads)
    return grads


def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g
