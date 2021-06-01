"""Data class, holding information about dataloaders and poison ids."""

import torch
import pickle

from .kettle_base import _Kettle
from ..consts import PIN_MEMORY, NORMALIZE, cifar10_mean, cifar10_std

class KettleExternal(_Kettle):
    """Generate a dataset definition completely from file."""

    def __init__(self, args, batch_size, augmentations, mixing_method=dict(type=None, strength=0.0),
                 setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize with given specs..."""
        self.args, self.setup = args, setup
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.mixing_method = mixing_method

        with open(args.file.name, 'rb') as handle:
            data_package = pickle.load(handle)

        if 'xtrain' in data_package.keys():
            # Load metapoison packages from
            # https://github.com/wronnyhuang/metapoison
            self.trainset, self.validset, self.poisonset, self.targetset = self._load_metapoison_files(data_package)

        self.prepare_diff_data_augmentations(normalize=NORMALIZE)
        num_workers = self.get_num_workers()
        if self.args.lmdb_path is not None:
            from .lmdb_datasets import LMDBDataset  # this also depends on py-lmdb
            self.trainset = LMDBDataset(self.trainset, self.args.lmdb_path, 'train')
            self.validset = LMDBDataset(self.validset, self.args.lmdb_path, 'val')

        if self.args.cache_dataset:
            self.trainset = CachedDataset(self.trainset, num_workers=num_workers)
            self.validset = CachedDataset(self.validset, num_workers=num_workers)
            num_workers = 0


        # Generate loaders:
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=min(self.batch_size, len(self.trainset)),
                                                       shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        self.validloader = torch.utils.data.DataLoader(self.validset, batch_size=min(self.batch_size, len(self.validset)),
                                                       shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        validated_batch_size = max(min(args.pbatch, len(self.poisonset)), 1)
        self.poisonloader = torch.utils.data.DataLoader(self.poisonset, batch_size=validated_batch_size,
                                                        shuffle=self.args.pshuffle, drop_last=False, num_workers=num_workers,
                                                        pin_memory=PIN_MEMORY)

        # Save clean ids for later? This might be impossible if ids are unknown.
        self.clean_ids = []
        # Finally: status
        self.print_status()


    def _load_metapoison_files(self, data_package):
        """Load a metapoison package.

        xtrain: CIFAR-10 training set images with a subset of them poisoned/perturbed
        ytrain: CIFAR-10 training set labels
        xtarget: Target image
        ytarget: Target true label
        ytargetadv: Target adversarial label
        xvalid: CIFAR-10 test set images
        yvalid: CIFAR-10 test set labels

        The IDs of the poisoned subset are 25000 to 25000 + num_poisons.
        Note that in a realistic setting, the IDs of the poisoned subset are unknown.
        """
        # this set is effectively unknown, but can be estimated as:
        possible_poison_ids = torch.arange(len(data_package['xtrain']))[data_package['ytrain'] == data_package['ytargetadv']]

        if NORMALIZE:
            data_mean, data_std = [m * 255 for m in cifar10_mean], [s * 255 for s in cifar10_std]

        for key in ['xtrain', 'xvalid', 'xtarget']:
            if NORMALIZE:
                dm, ds = torch.as_tensor(data_mean)[None, :, None, None], torch.as_tensor(data_std)[None, :, None, None]
                data_package[key] = torch.as_tensor(data_package[key]).permute(0, 3, 1, 2).sub_(dm).div_(ds)
            else:
                data_package[key] = torch.as_tensor(data_package[key]).permute(0, 3, 1, 2).div_(255)

        ids = torch.arange(len(data_package['xtrain']))
        trainset = torch.utils.data.TensorDataset(data_package['xtrain'], torch.as_tensor(data_package['ytrain']), ids)
        validset = torch.utils.data.TensorDataset(data_package['xvalid'], torch.as_tensor(data_package['yvalid']),
                                                  ids[:len(data_package['yvalid'])])
        poisonset = torch.utils.data.TensorDataset(data_package['xtrain'][possible_poison_ids],
                                                   torch.as_tensor(data_package['ytrain'])[possible_poison_ids],
                                                   possible_poison_ids)
        targetset = torch.utils.data.TensorDataset(data_package['xtarget'], torch.as_tensor(data_package['ytargetadv']),
                                                   torch.arange(len(data_package['xtarget'])))

        self.poison_setup = dict(poison_budget=self.args.budget, target_num=len(data_package['xtarget']),
                                 poison_class=data_package['ytargetadv'][0], target_class=data_package['ytarget'][0],
                                 intended_class=data_package['ytargetadv'])

        # Save normalizations for later, but these are baked into the tensors already
        trainset.data_mean, trainset.data_std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        trainset.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.target_ids = [None]
        self.init_seed = f'{self.poison_setup["target_class"]} - {self.poison_setup["poison_class"]} - ?'

        return trainset, validset, poisonset, targetset
