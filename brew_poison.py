"""General interface script to launch poisoning jobs.

This script is the multi-functional central interface to the forest library.
Skipping this script [unfortunately] requires spoofing the config args structure given in options.py .

This script goes through the following steps:
1) Load a dataset and generate a test example [this includes choosing which targets to poison with which poisons]
2) Pretrain or download a clean model.
3) Brew poisoned data for the given dataset. Often by taking the pretrained model into account.
3b) Optionally: Attempt to remove poisoned data by filtering. This may entail another model training
4) "Validate" the effect of the poisoned data by training a new model.
5) Record tabular information about the experiment and optionally export poisoned data into various formats.

- All model trainings can optionally include robust training by "adversarial poisoning" which reduces poison success.
- For finetuning/transfer settings, the pretrained model from 2) is used as a "base model" and later trainings only retrain
  starting from this base model.
"""

import torch

import datetime
import time

import forest
from forest.filtering_defenses import get_defense
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()
# 100% reproducibility?
if args.deterministic:
    forest.utils.set_deterministic()


if __name__ == "__main__":

    setup = forest.utils.system_startup(args)

    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup)
    witch = forest.Witch(args, setup=setup)
    witch.patch_targets(data)

    start_time = time.time()
    if args.pretrained_model:
        print('Loading pretrained model...')
        stats_clean = None
    elif args.skip_clean_training:
        print('Skipping clean training...')
        stats_clean = None
    else:
        stats_clean = model.train(data, max_epoch=args.max_epoch)
    train_time = time.time()

    poison_delta = witch.brew(model, data)
    brew_time = time.time()

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

    if not args.pretrained_model and args.retrain_from_init:
        # retraining from the same seed is incompatible --pretrained as we do not know the initial seed..
        stats_rerun = model.retrain(data, poison_delta)
    else:
        stats_rerun = None

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

    timestamps = dict(train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(',', ''),
                      brew_time=str(datetime.timedelta(seconds=brew_time - train_time)).replace(',', ''),
                      test_time=str(datetime.timedelta(seconds=test_time - brew_time)).replace(',', ''))
    # Save run to table
    results = (stats_clean, stats_rerun, stats_results)
    forest.utils.record_results(data, witch.stat_optimal_loss, results,
                                args, model.defs, model.model_init_seed, extra_stats={**filter_stats, **timestamps})

    # Export
    if args.save is not None:
        data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
    print(f'--------------------------- brew time: {str(datetime.timedelta(seconds=brew_time - train_time))}')
    print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - brew_time))}')
    print('-------------Job finished.-------------------------')
