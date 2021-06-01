"""Main class, holding information about models and training/testing routines."""

import torch
import higher

from collections import OrderedDict

from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK
from .modules import MetaMonkey

from .witch_base import _Witch


class WitchMetaPoison(_Witch):
    """Brew metapoison with given arguments.

    Note: This function does not work in single-model-multi-GPU mode, due to the weights being fixed to a single GPU.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _define_objective(self, inputs, labels, criterion, targets, intended_classes, *args):
        def closure(model, optimizer, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
            model = MetaMonkey(model)

            for _ in range(self.args.nadapt):
                outputs = model(inputs, model.parameters)
                prediction = (outputs.data.argmax(dim=1) == labels).sum()

                poison_loss = criterion(outputs, labels)
                poison_grad = torch.autograd.grad(poison_loss, model.parameters.values(),
                                                  retain_graph=True, create_graph=True, only_inputs=True)

                current_lr = optimizer.param_groups[0]['lr']
                model.parameters = OrderedDict((name, param - current_lr * grad_part)
                                               for ((name, param), grad_part) in zip(model.parameters.items(), poison_grad))
            # model.eval()
            target_outs = model(targets, model.parameters)
            target_loss = criterion(target_outs, intended_classes)
            target_loss.backward(retain_graph=self.retain)

            return target_loss.detach().cpu(), prediction.detach().cpu()
        return closure


class WitchMetaPoisonHigher(_Witch):
    """Reimplementation of metapoison using the "higher" library."""

    def _define_objective(self, inputs, labels, criterion, targets, intended_classes, *args):
        def closure(model, optimizer, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
            with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fmodel, fopt):
                for _ in range(self.args.nadapt):
                    outputs = fmodel(inputs)
                    poison_loss = criterion(outputs, labels)

                    fopt.step(poison_loss)

            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            # model.eval()
            target_loss = criterion(fmodel(targets), intended_classes)
            target_loss.backward(retain_graph=self.retain)

            return target_loss.detach().cpu(), prediction.detach().cpu()

        return closure



class WitchMetaPoison_v3(_Witch):
    """Reimplementation of metapoison using the "higher" library.

    This version also implements the "shared-batch" between target and inputs.
    """

    def _define_objective(self, inputs, labels, criterion, targets, intended_classes, *args):
        def closure(model, optimizer, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            list(model.children())[-1].train() if model.frozen else model.train()
            batch_size = inputs.shape[0]

            data = torch.cat((inputs, targets), dim=0)

            # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
            with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fmodel, fopt):
                for _ in range(self.args.nadapt):
                    outputs = fmodel(data)
                    poison_loss = criterion(outputs[:batch_size], labels)

                    fopt.step(poison_loss)

            prediction = (outputs[:batch_size].data.argmax(dim=1) == labels).sum()

            target_loss = criterion(outputs[batch_size:], intended_classes)
            target_loss.backward(retain_graph=self.retain)

            return target_loss.detach().cpu(), prediction.detach().cpu()

        return closure
