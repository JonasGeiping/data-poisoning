"""General interface script to launch distributed poisoning jobs. Launch only through the pytorch launch utility.

This is the distributed equivalent to brew_poison.py. The poisoning process is split over multiple GPUs
to construct an ensemble of poisoning models (like --ensemble N in the single GPU case).
This only parallelizes the ensembling! 

"""

import socket
import datetime
import time


import torch
import forest
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()
# Parse training strategy
defs = forest.training_strategy(args)
# 100% reproducibility?
if args.deterministic:
    forest.utils.set_deterministic()

if args.local_rank is None:
    raise ValueError('This script should only be launched via the pytorch launch utility!')


if __name__ == "__main__":

    if torch.cuda.device_count() < args.local_rank:
        raise ValueError('Process invalid, oversubscribing to GPUs is not possible in this mode.')
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
    setup = dict(device=device, dtype=torch.float, non_blocking=forest.consts.NON_BLOCKING)
    torch.distributed.init_process_group(backend=forest.consts.DISTRIBUTED_BACKEND, init_method='env://')
    if args.ensemble != 1 and args.ensemble != torch.distributed.get_world_size():
        raise ValueError('Argument given to ensemble does not match number of launched processes!')
    else:
        args.ensemble = torch.distributed.get_world_size()
        if torch.distributed.get_rank() == 0:
            print('Currently evaluating -------------------------------:')
            print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
            print(args)
            print(repr(defs))
            print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}')
            print(f'Ensemble launched on {torch.distributed.get_world_size()} GPUs'
                  f' with backend {forest.consts.DISTRIBUTED_BACKEND}.')

    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')

    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup)
    witch = forest.Witch(args, setup=setup)
    witch.patch_targets(data)

    start_time = time.time()
    if args.pretrained_model:
        print('Loading pretrained model...')
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
            stats_baseline = model.validate(data, poison_delta)
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
        stats_rerun = model.retrain(data, poison_delta)
    else:
        stats_rerun = None  # we dont know the initial seed for a pretrained model so retraining makes no sense

    if args.vnet is not None:  # Validate the transfer model given by args.vnet
        train_net = args.net
        args.net = args.vnet
        if args.vruns > 0:
            model = forest.Victim(args, setup=setup)
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

    if torch.distributed.get_rank() == 0:
        timestamps = dict(train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(',', ''),
                          brew_time=str(datetime.timedelta(seconds=brew_time - train_time)).replace(',', ''),
                          test_time=str(datetime.timedelta(seconds=test_time - brew_time)).replace(',', ''))
        # Save run to table
        results = (stats_clean, stats_rerun, stats_results)
        forest.utils.record_results(data, witch.stat_optimal_loss, results,
                                    args, defs, model.model_init_seed, extra_stats={**filter_stats, **timestamps})

        # Export
        if args.save:
            data.export_poison(poison_delta, path=None, mode='full')

        print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
        print('---------------------------------------------------')
        print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
        print(f'--------------------------- brew time: {str(datetime.timedelta(seconds=brew_time - train_time))}')
        print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - brew_time))}')
        print('-------------Job finished.-------------------------')
