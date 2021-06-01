"""This is a specialized interface that can be used to load
a poisoned dataset and evaluate its effectiveness.
This script does not generate poisoned data!

It can be used as a sanity check, or to check poisoned data from another repository.
"""

import torch

import datetime
import time
import argparse

import forest
from forest.filtering_defenses import get_defense
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
parser = forest.options()
parser.add_argument('file', type=argparse.FileType())
args = parser.parse_args()

# 100% reproducibility?
if args.deterministic:
    forest.utils.set_deterministic()


if __name__ == "__main__":

    setup = forest.utils.system_startup(args)

    model = forest.Victim(args, setup=setup)
    data = forest.KettleExternal(args, model.defs.batch_size, model.defs.augmentations,
                                 model.defs.mixing_method, setup=setup)
    poison_delta = None  # poison locations are unknown

    start_time = time.time()
    # Optional: apply a filtering defense
    if args.filter_defense != '':
        # Crucially any filtering defense would not have access to the final clean model used by the attacker,
        # as such we need to retrain a poisoned model to use as basis for a filter defense if we are in the from-scratch
        # setting where no pretrained feature representation is available to both attacker and defender
        if args.scenario == 'from-scratch':
            model.validate(data, poison_delta)
        print('Attempting to filter poison images...')
        defense = get_defense(args)
        clean_ids = defense(data, model, poison_delta)
        poison_ids = set(range(len(data.trainset))) - set(clean_ids)
        removed_images = len(data.trainset) - len(clean_ids)
        removed_poisons = len(set(data.poison_ids.tolist()) & poison_ids)

        data.reset_trainset(clean_ids)
        print(f'Filtered {removed_images} images out of {len(data.trainset.dataset)}. {removed_poisons} were poisons.')
        filter_stats = dict(removed_poisons=removed_poisons, removed_images_total=removed_images)
    else:
        filter_stats = dict()

    if args.vnet is not None:  # Validate the transfer model given by args.vnet
        train_net = args.net
        args.net = args.vnet
        if args.vruns > 0:
            model = forest.Victim(args, setup=setup)  # this instantiates a new model with a different architecture
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
        args.net = train_net
    else:  # Validate the main model
        if args.vruns > 0:
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
    test_time = time.time()

    timestamps = dict(train_time=None,
                      brew_time=None,
                      test_time=str(datetime.timedelta(seconds=test_time - start_time)).replace(',', ''))
    # Save run to table
    results = (None, None, stats_results)
    forest.utils.record_results(data, None, results,
                                args, model.defs, model.model_init_seed, extra_stats={**filter_stats, **timestamps})

    # Export into a different format?
    if args.save is not None:
        data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with test time: {str(datetime.timedelta(seconds=test_time - start_time))}')
    print('-------------Job finished.-------------------------')
