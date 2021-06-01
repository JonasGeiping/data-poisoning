"""Single model default victim class."""

import torch
import numpy as np

from collections import defaultdict
import copy

from .models import get_model
from .training import get_optimizers
from ..hyperparameters import training_strategy
from ..utils import set_random_seed
from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from .victim_base import _VictimBase


class _VictimSingle(_VictimBase):
    """Implement model-specific code and behavior for a single model on a single GPU.

    This is the simplest victim implementation.

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
        self.model, self.defs, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0], pretrain=pretrain)

        self.model.to(**self.setup)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model.frozen = self.model.module.frozen
        print(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.')
        print(repr(self.defs))

    def reinitialize_last_layer(self, reduce_lr_factor=1.0, seed=None, keep_last_layer=False):
        if not keep_last_layer:
            if self.args.modelkey is None:
                if seed is None:
                    self.model_init_seed = np.random.randint(0, 2**32 - 1)
                else:
                    self.model_init_seed = seed
            else:
                self.model_init_seed = self.args.modelkey
            set_random_seed(self.model_init_seed)

            # We construct a full replacement model, so that the seed matches up with the initial seed,
            # even if all of the model except for the last layer will be immediately discarded.
            replacement_model = get_model(self.args.net[0], self.args.dataset, pretrained=self.args.pretrained_model)

            # Rebuild model with new last layer
            frozen = self.model.frozen
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1], torch.nn.Flatten(), list(replacement_model.children())[-1])
            self.model.frozen = frozen
            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.model.frozen = self.model.module.frozen

        # Define training routine
        # Reinitialize optimizers here
        self.defs = training_strategy(self.args.net[0], self.args)
        self.defs.lr *= reduce_lr_factor
        self.optimizer, self.scheduler = get_optimizers(self.model, self.args, self.defs)
        print(f'{self.args.net[0]} last layer re-initialized with random key {self.model_init_seed}.')
        print(repr(self.defs))

    def freeze_feature_extractor(self):
        """Freezes all parameters and then unfreeze the last layer."""
        self.model.frozen = True
        for param in self.model.parameters():
            param.requires_grad = False

        for param in list(self.model.children())[-1].parameters():
            param.requires_grad = True

    def save_feature_representation(self):
        self.clean_model = copy.deepcopy(self.model)

    def load_feature_representation(self):
        self.model = copy.deepcopy(self.clean_model)


    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def _iterate(self, kettle, poison_delta, max_epoch=None, pretraining_phase=False):
        """Validate a given poison by training the model and checking target accuracy."""
        stats = defaultdict(list)

        if max_epoch is None:
            max_epoch = self.defs.epochs

        single_setup = (self.model, self.defs, self.optimizer, self.scheduler)
        for self.epoch in range(max_epoch):
            self._step(kettle, poison_delta, self.epoch, stats, *single_setup, pretraining_phase)
            if self.args.dryrun:
                break
        return stats

    def step(self, kettle, poison_delta, poison_targets, true_classes):
        """Step through a model epoch. Optionally: minimize target loss."""
        stats = defaultdict(list)


        single_setup = (self.model, self.defs, self.optimizer, self.scheduler)
        self._step(kettle, poison_delta, self.epoch, stats, *single_setup)
        self.epoch += 1
        if self.epoch > self.defs.epochs:
            self.epoch = 0
            print('Model reset to epoch 0.')
            self.model, self.defs, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0])
            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1 and 'meta' not in self.defs.novel_defense['type']:
                self.model = torch.nn.DataParallel(self.model)
                self.model.frozen = self.model.module.frozen
        return stats

    """ Various Utilities."""

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()
        self.model.eval()
        if dropout:
            self.model.apply(apply_dropout)

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        _, _, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0])

    def gradient(self, images, labels, criterion=None):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        if criterion is None:
            loss = self.loss_fn(self.model(images), labels)
        else:
            loss = criterion(self.model(images), labels)
        differentiable_params = [p for p in self.model.parameters() if p.requires_grad]
        gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return gradients, grad_norm

    def compute(self, function, *args):
        r"""Compute function on the given optimization problem, defined by criterion \circ model.

        Function has arguments: model, criterion
        """
        return function(self.model, self.optimizer, *args)
