"""Definition for multiple victims that share a single GPU (sequentially)."""

import torch
import numpy as np

from collections import defaultdict
import copy

from .models import get_model
from ..hyperparameters import training_strategy
from ..utils import set_random_seed, average_dicts
from ..consts import BENCHMARK
from .context import GPUContext
torch.backends.cudnn.benchmark = BENCHMARK

from .victim_base import _VictimBase
from .training import get_optimizers

class _VictimEnsemble(_VictimBase):
    """Implement model-specific code and behavior for multiple models on a single GPU.

    --> Running in sequential mode!

    """

    """ Methods to initialize a model."""

    def initialize(self, pretrain=False, seed=None):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)
        print(f'Initializing ensemble from random key {self.model_init_seed}.')

        self.models, self.definitions, self.optimizers, self.schedulers, self.epochs = [], [], [], [], []
        for idx in range(self.args.ensemble):
            model_name = self.args.net[idx % len(self.args.net)]
            model, defs, optimizer, scheduler = self._initialize_model(model_name, pretrain=pretrain)
            self.models.append(model)
            self.definitions.append(defs)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
            print(f'{model_name} initialized as model {idx}')
            print(repr(defs))
        self.defs = self.definitions[0]

    def reinitialize_last_layer(self, reduce_lr_factor=1.0, seed=None, keep_last_layer=False):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)


        for idx in range(self.args.ensemble):
            model_name = self.args.net[idx % len(self.args.net)]
            if not keep_last_layer:
                # We construct a full replacement model, so that the seed matches up with the initial seed,
                # even if all of the model except for the last layer will be immediately discarded.
                replacement_model = get_model(model_name, self.args.dataset, pretrained=self.args.pretrained_model)

                # Rebuild model with new last layer
                frozen = self.models[idx].frozen
                self.models[idx] = torch.nn.Sequential(*list(self.models[idx].children())[:-1], torch.nn.Flatten(),
                                                       list(replacement_model.children())[-1])
                self.models[idx].frozen = frozen

            # Define training routine
            # Reinitialize optimizers here
            self.definitions[idx] = training_strategy(model_name, self.args)
            self.definitions[idx].lr *= reduce_lr_factor
            self.optimizers[idx], self.schedulers[idx] = get_optimizers(self.models[idx], self.args, self.definitions[idx])
            print(f'{model_name} with id {idx}: linear layer reinitialized.')
            print(repr(self.definitions[idx]))


    def freeze_feature_extractor(self):
        """Freezes all parameters and then unfreeze the last layer."""
        for model in self.models:
            model.frozen = True
            for param in model.parameters():
                param.requires_grad = False

            for param in list(model.children())[-1].parameters():
                param.requires_grad = True

    def save_feature_representation(self):
        self.clean_models = []
        for model in self.models:
            self.clean_models.append(copy.deepcopy(model))

    def load_feature_representation(self):
        self.models = []
        for clean_model in self.clean_models:
            self.models.append(copy.deepcopy(clean_model))


    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def _iterate(self, kettle, poison_delta, max_epoch=None, pretraining_phase=False):
        """Validate a given poison by training the model and checking target accuracy."""
        multi_model_setup = (self.models, self.definitions, self.optimizers, self.schedulers)

        # Only partially train ensemble for poisoning if no poison is present
        if max_epoch is None:
            max_epoch = self.defs.epochs
        if poison_delta is None and self.args.stagger is not None and not pretraining_phase:
            if self.args.stagger == 'firstn':
                stagger_list = [int(epoch) for epoch in range(self.args.ensemble)]
            elif self.args.stagger == 'full':
                stagger_list = [int(epoch) for epoch in np.linspace(0, max_epoch, self.args.ensemble)]
            elif self.args.stagger == 'inbetween':
                stagger_list = [int(epoch) for epoch in np.linspace(0, max_epoch, self.args.ensemble + 2)[1:-1]]
            else:
                raise ValueError(f'Invalid stagger option {self.args.stagger}')
            print(f'Staggered pretraining to {stagger_list}.')
        else:
            stagger_list = [max_epoch] * self.args.ensemble

        run_stats = list()
        for idx, single_model in enumerate(zip(*multi_model_setup)):
            stats = defaultdict(list)
            model, defs, optimizer, scheduler = single_model

            # Move to GPUs
            model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.frozen = model.module.frozen

            for epoch in range(stagger_list[idx]):
                self._step(kettle, poison_delta, epoch, stats, *single_model, pretraining_phase)
                if self.args.dryrun:
                    break
            # Return to CPU
            if torch.cuda.device_count() > 1:
                model = model.module
            model.to(device=torch.device('cpu'))
            run_stats.append(stats)

        if poison_delta is None and self.args.stagger is not None:
            average_stats = run_stats[-1]
        else:
            average_stats = average_dicts(run_stats)

        # Track epoch
        self.epochs = stagger_list

        return average_stats

    def step(self, kettle, poison_delta, poison_targets, true_classes):
        """Step through a model epoch. Optionally minimize target loss during this.

        This function is limited because it assumes that defs.batch_size, defs.max_epoch, defs.epochs
        are equal for all models.
        """
        multi_model_setup = (self.models, self.definitions, self.optimizers, self.schedulers)

        run_stats = list()
        for idx, single_model in enumerate(zip(*multi_model_setup)):
            model, defs, optimizer, scheduler = single_model
            model_name = self.args.net[idx % len(self.args.net)]

            # Move to GPUs
            model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.frozen = model.module.frozen
            self._step(kettle, poison_delta, self.epochs[idx], defaultdict(list), *single_model)
            self.epochs[idx] += 1
            if self.epochs[idx] > defs.epochs:
                self.epochs[idx] = 0
                print(f'Model {idx} reset to epoch 0.')
                model, defs, optimizer, scheduler = self._initialize_model(model_name)
            # Return to CPU
            if torch.cuda.device_count() > 1:
                model = model.module
            model.to(device=torch.device('cpu'))
            self.models[idx], self.definitions[idx], self.optimizers[idx], self.schedulers[idx] = model, defs, optimizer, scheduler

    """ Various Utilities."""

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()
        [model.eval() for model in self.models]
        if dropout:
            [model.apply(apply_dropout) for model in self.models]

    def reset_learning_rate(self):
        """Reset scheduler objects to initial state."""
        for idx in range(self.args.ensemble):
            _, _, optimizer, scheduler = self._initialize_model()
            self.optimizers[idx] = optimizer
            self.schedulers[idx] = scheduler

    def gradient(self, images, labels, criterion=None):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        grad_list, norm_list = [], []
        for model in self.models:
            with GPUContext(self.setup, model) as model:
                if criterion is None:
                    loss = self.loss_fn(model(images), labels)
                else:
                    loss = criterion(model(images), labels)
                differentiable_params = [p for p in model.parameters() if p.requires_grad]
                grad_list.append(torch.autograd.grad(loss, differentiable_params, only_inputs=True))
                grad_norm = 0
                for grad in grad_list[-1]:
                    grad_norm += grad.detach().pow(2).sum()
                norm_list.append(grad_norm.sqrt())
        return grad_list, norm_list

    def compute(self, function, *args):
        """Compute function on all models.

        Function has arguments that are possibly sequences of length args.ensemble
        """
        outputs = []
        for idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            with GPUContext(self.setup, model) as model:
                single_arg = [arg[idx] if hasattr(arg, '__iter__') else arg for arg in args]
                outputs.append(function(model, optimizer, *single_arg))
        # collate
        avg_output = [np.mean([output[idx] for output in outputs]) for idx, _ in enumerate(outputs[0])]
        return avg_output
