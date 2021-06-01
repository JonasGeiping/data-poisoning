"""Data class, holding information about dataloaders and poison ids."""

import numpy as np
from .kettle_base import _Kettle
from .datasets import Subset


class KettleDeterministic(_Kettle):
    """Generate parameters for an experiment based on a fixed triplet a-b-c given via --poisonkey.

    This construction replicates the experiment definitions for MetaPoison.

    The triplet key, e.g. 5-3-1 denotes in order:
    target_class - poison_class - target_id
    """

    def prepare_experiment(self):
        """Choose targets from some label which will be poisoned toward some other chosen label, by modifying some
        subset of the training data within some bounds."""
        self.deterministic_construction()

    def deterministic_construction(self):
        """Construct according to the triplet input key.

        Poisons are always the first n occurences of the given class.
        [This is the same setup as in metapoison]
        """
        if self.args.threatmodel != 'single-class':
            raise NotImplementedError()

        split = self.args.poisonkey.split('-')
        if len(split) != 3:
            raise ValueError('Invalid poison triplet supplied.')
        else:
            target_class, poison_class, target_id = [int(s) for s in split]
        self.init_seed = self.args.poisonkey
        print(f'Initializing Poison data (chosen images, examples, targets, labels) as {self.args.poisonkey}')

        self.poison_setup = dict(poison_budget=self.args.budget,
                                 target_num=self.args.targets, poison_class=poison_class, target_class=target_class,
                                 intended_class=[poison_class])
        self.poisonset, self.targetset, self.validset = self._choose_poisons_deterministic(target_id)

    def _choose_poisons_deterministic(self, target_id):
        # poisons
        class_ids = []
        for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
            target, idx = self.trainset.get_target(index)
            if target == self.poison_setup['poison_class']:
                class_ids.append(idx)

        poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
        if len(class_ids) < poison_num:
            warnings.warn(f'Training set is too small for requested poison budget.')
            poison_num = len(class_ids)
        self.poison_ids = class_ids[:poison_num]

        # the target
        # class_ids = []
        # for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
        #     target, idx = self.validset.get_target(index)
        #     if target == self.poison_setup['target_class']:
        #         class_ids.append(idx)
        # self.target_ids = [class_ids[target_id]]
        # Disable for now for benchmark sanity check. This is a breaking change.
        self.target_ids = [target_id]

        targetset = Subset(self.validset, indices=self.target_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            if idx not in self.target_ids:
                valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids, range(poison_num)))
        dict(zip(self.poison_ids, range(poison_num)))
        return poisonset, targetset, validset
