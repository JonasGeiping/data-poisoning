"""Main class, holding information about models and training/testing routines."""

import torch

import numpy as np
from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch
from ..utils import bypass_last_layer

class WitchConvexPolytope(_Witch):
    """Brew poison frogs variant with averaged feature matching instead of sums of feature matches.

    This is also known as BullsEye Polytope Attack.

    """

    def _define_objective(self, inputs, labels, criterion, targets, intended_classes, true_classes):
        """Implement the closure here."""
        def closure(model, optimizer, target_grad, target_clean_grad, target_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            # Iniitalize coefficients
            coeffs = (1 / inputs.shape[0] * torch.ones_like(labels)).to(dtype=inputs.dtype, device=inputs.device)

            # Carve up the model
            feature_model, last_layer = bypass_last_layer(model)

            # Get standard output:
            outputs = feature_model(inputs)
            outputs_targets = feature_model(targets)

            coeffs = _least_squares_simplex(
                A=outputs.t().detach(),
                b=outputs_targets.t().detach().squeeze(),
                x_init=coeffs,
                device=inputs.device
            )

            residual = outputs_targets - torch.sum(coeffs[:, None] * outputs, 0, keepdim=True)
            target_norm_square = torch.sum(outputs_targets ** 2)
            feature_loss = 0.5 * torch.sum(residual ** 2) / target_norm_square

            prediction = (last_layer(outputs).data.argmax(dim=1) == labels).sum()
            feature_loss.backward(retain_graph=self.retain)
            return feature_loss.detach().cpu(), prediction.detach().cpu()
        return closure

def _proj_onto_simplex(coeffs, psum=1.0):
    """Project onto probability simplex by default.

    Code from https://github.com/hsnamkoong/robustopt/blob/master/src/simple_projections.py
    See MIT License there.
    """
    v_np = coeffs.view(-1).detach().cpu().numpy()
    n_features = v_np.shape[0]
    v_sorted = np.sort(v_np)[::-1]
    cssv = np.cumsum(v_sorted) - psum
    ind = np.arange(n_features) + 1
    cond = v_sorted - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w_ = np.maximum(v_np - theta, 0)
    return torch.Tensor(w_.reshape(coeffs.size())).to(coeffs.device)

def _least_squares_simplex(A, b, x_init, tol=1e-6, verbose=False, device="cuda"):
    """Implement the inner loop of Algorithm 1."""
    m, n = A.size()
    assert (
        b.size()[0] == A.size()[0]
    ), "Matrix and vector do not have compatible dimensions"

    # Initialize the optimization variables
    if x_init is None:
        x = torch.zeros(n, 1).to(device)
    else:
        x = x_init

    # Define the objective function and its gradient
    def f(x):
        return torch.norm(A.matmul(x) - b).item()
    # change into a faster version when A is a tall matrix
    AtA = A.t().mm(A)
    Atb = A.t().matmul(b)

    def grad_f(x):
        return AtA.matmul(x) - Atb
    # grad_f = lambda x: A.t().mm(A.mm(x)-b)

    # Estimate the spectral radius of the Matrix A'A
    y = torch.normal(0, torch.ones(n, 1)).to(device)
    lipschitz = torch.norm(A.t().mm(A.mm(y))) / torch.norm(y)

    # The stepsize for the problem should be 2/lipschits.  Our estimator might not be correct, it could be too small.  In
    # this case our learning rate will be too big, and so we need to have a backtracking line search to make sure things converge.
    t = 2 / lipschitz

    # Main iteration
    for iter in range(10000):
        x_hat = x - t * grad_f(
            x
        )  # Forward step:  Gradient decent on the objective term
        if f(x_hat) > f(
            x
        ):  # Check whether the learning rate is small enough to decrease objective
            t = t / 2
        else:
            x_new = _proj_onto_simplex(x_hat)  # Backward step: Project onto prob simplex
            stopping_condition = torch.norm(x - x_new) / max(torch.norm(x), 1e-8)
            if verbose:
                print("iter %d: error = %0.4e" % (iter, stopping_condition))
            if stopping_condition < tol:  # check stopping conditions
                break
            x = x_new

    return x
