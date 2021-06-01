"""Main class, holding information about models and training/testing routines."""

import torch
from ..utils import bypass_last_layer
from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch

class WitchFrogs(_Witch):
    """Brew poison frogs poison with given arguments.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _define_objective(self, inputs, labels, criterion, targets, intended_classes, true_classes):
        """Implement the closure here."""
        def closure(model, optimizer, target_grad, target_clean_grad, target_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            # Carve up the model
            feature_model, last_layer = bypass_last_layer(model)

            # Get standard output:
            outputs = feature_model(inputs)
            outputs_targets = feature_model(targets)
            prediction = (last_layer(outputs).data.argmax(dim=1) == labels).sum()

            feature_loss = (outputs.mean(dim=0, keepdim=True) - outputs_targets).pow(2).mean()
            feature_loss.backward(retain_graph=self.retain)
            return feature_loss.detach().cpu(), prediction.detach().cpu()
        return closure
