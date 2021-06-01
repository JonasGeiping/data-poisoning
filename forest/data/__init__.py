"""Basic data handling."""
import torch

from .kettle_random_experiment import KettleRandom
from .kettle_det_experiment import KettleDeterministic
from .kettle_benchmark_experiment import KettleBenchmark
from .kettle_external import KettleExternal

__all__ = ['Kettle', 'KettleExternal']


def Kettle(args, batch_size, augmentations, mixing_method=dict(type=None, strength=0.0),
           setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Interface to connect to a kettle [data] child class."""

    if args.poisonkey is None:
        if args.benchmark != '':
            return KettleBenchmark(args, batch_size, augmentations, mixing_method, setup)
        else:
            return KettleRandom(args, batch_size, augmentations, mixing_method, setup)

    else:
        if '-' in args.poisonkey:
            # If the poisonkey contains a dash-separated triplet like 5-3-1, then poisons are drawn
            # entirely deterministically.
            return KettleDeterministic(args, batch_size, augmentations, mixing_method, setup)
        else:
            # Otherwise the poisoning process is random.
            # If the poisonkey is a random integer, then this integer will be used
            # as a key to seed the random generators.
            return KettleRandom(args, batch_size, augmentations, mixing_method, setup)
