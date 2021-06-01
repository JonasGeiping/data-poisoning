"""Data class, holding information about dataloaders and poison ids."""

import torch
import numpy as np

from .kettle_base import _Kettle
from ..utils import set_random_seed
from .datasets import Subset


class KettleRandom(_Kettle):
    """Generate parameters for an experiment randomly.

    If --poisonkey is provided, then it will be used to seed the randomization.

    """

    def prepare_experiment(self):
        """Choose targets from some label which will be poisoned toward some other chosen label, by modifying some
        subset of the training data within some bounds."""
        self.random_construction()


    def random_construction(self):
        """Construct according to random selection.

        The setup can be repeated from its key (which initializes the random generator).
        This method sets
         - poison_setup
         - poisonset / targetset / validset

        """
        if self.args.local_rank is None:
            if self.args.poisonkey is None:
                self.init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.init_seed = int(self.args.poisonkey)
            set_random_seed(self.init_seed)
            print(f'Initializing Poison data (chosen images, examples, targets, labels) with random seed {self.init_seed}')
        else:
            rank = torch.distributed.get_rank()
            if self.args.poisonkey is None:
                init_seed = torch.randint(0, 2**32 - 1, [1], device=self.setup['device'])
            else:
                init_seed = torch.as_tensor(int(self.args.poisonkey), dtype=torch.int64, device=self.setup['device'])
            torch.distributed.broadcast(init_seed, src=0)
            if rank == 0:
                print(f'Initializing Poison data (chosen images, examples, targets, labels) with random seed {init_seed.item()}')
            self.init_seed = init_seed.item()
            set_random_seed(self.init_seed)
        # Parse threat model
        self.poison_setup = self._parse_threats_randomly()
        self.poisonset, self.targetset, self.validset = self._choose_poisons_randomly()

    def _parse_threats_randomly(self):
        """Parse the different threat models.

        The threat-models are [In order of expected difficulty]:

        single-class replicates the threat model of feature collision attacks,
        third-party draws all poisons from a class that is unrelated to both target and intended label.
        random-subset draws poison images from all classes.
        random-subset draw poison images from all classes and draws targets from different classes to which it assigns
        different labels.
        """
        num_classes = len(self.trainset.classes)

        target_class = np.random.randint(num_classes)
        list_intentions = list(range(num_classes))
        list_intentions.remove(target_class)
        intended_class = [np.random.choice(list_intentions)] * self.args.targets

        if self.args.targets < 1:
            poison_setup = dict(poison_budget=0, target_num=0,
                                poison_class=np.random.randint(num_classes), target_class=None,
                                intended_class=[np.random.randint(num_classes)])
            warnings.warn('Number of targets set to 0.')
            return poison_setup

        if self.args.threatmodel == 'single-class':
            poison_class = intended_class[0]
            poison_setup = dict(poison_budget=self.args.budget, target_num=self.args.targets,
                                poison_class=poison_class, target_class=target_class, intended_class=intended_class)
        elif self.args.threatmodel == 'third-party':
            list_intentions.remove(intended_class[0])
            poison_class = np.random.choice(list_intentions)
            poison_setup = dict(poison_budget=self.args.budget, target_num=self.args.targets,
                                poison_class=poison_class, target_class=target_class, intended_class=intended_class)
        elif self.args.threatmodel == 'self-betrayal':
            poison_class = target_class
            poison_setup = dict(poison_budget=self.args.budget, target_num=self.args.targets,
                                poison_class=poison_class, target_class=target_class, intended_class=intended_class)
        elif self.args.threatmodel == 'random-subset':
            poison_class = None
            poison_setup = dict(poison_budget=self.args.budget,
                                target_num=self.args.targets, poison_class=None, target_class=target_class,
                                intended_class=intended_class)
        elif self.args.threatmodel == 'random-subset-random-targets':
            target_class = None
            intended_class = np.random.randint(num_classes, size=self.args.targets)
            poison_class = None
            poison_setup = dict(poison_budget=self.args.budget,
                                target_num=self.args.targets, poison_class=None, target_class=None,
                                intended_class=intended_class)
        else:
            raise NotImplementedError('Unknown threat model.')

        return poison_setup

    def _choose_poisons_randomly(self):
        """Subconstruct poison and targets.

        The behavior is different for poisons and targets. We still consider poisons to be part of the original training
        set and load them via trainloader (And then add the adversarial pattern Delta)
        The targets are fully removed from the validation set and returned as a separate dataset, indicating that they
        should not be considered during clean validation using the validloader

        """
        # Poisons:
        if self.poison_setup['poison_class'] is not None:
            class_ids = []
            for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
                target, idx = self.trainset.get_target(index)
                if target == self.poison_setup['poison_class']:
                    class_ids.append(idx)

            poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
            if len(class_ids) < poison_num:
                warnings.warn(f'Training set is too small for requested poison budget. \n'
                              f'Budget will be reduced to maximal size {len(class_ids)}')
                poison_num = len(class_ids)
            self.poison_ids = torch.tensor(np.random.choice(
                class_ids, size=poison_num, replace=False), dtype=torch.long)
        else:
            total_ids = []
            for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
                _, idx = self.trainset.get_target(index)
                total_ids.append(idx)
            poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
            if len(total_ids) < poison_num:
                warnings.warn(f'Training set is too small for requested poison budget. \n'
                              f'Budget will be reduced to maximal size {len(total_ids)}')
                poison_num = len(total_ids)
            self.poison_ids = torch.tensor(np.random.choice(
                total_ids, size=poison_num, replace=False), dtype=torch.long)

        # Targets:
        if self.poison_setup['target_class'] is not None:
            class_ids = []
            for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
                target, idx = self.validset.get_target(index)
                if target == self.poison_setup['target_class']:
                    class_ids.append(idx)
            self.target_ids = np.random.choice(class_ids, size=self.args.targets, replace=False)
        else:
            total_ids = []
            for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
                _, idx = self.validset.get_target(index)
                total_ids.append(idx)
            self.target_ids = np.random.choice(total_ids, size=self.args.targets, replace=False)

        targetset = Subset(self.validset, indices=self.target_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            if idx not in self.target_ids:
                valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids.tolist(), range(poison_num)))
        return poisonset, targetset, validset
