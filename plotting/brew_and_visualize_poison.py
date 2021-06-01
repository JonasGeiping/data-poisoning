"""General interface script to launch poisoning jobs.

Run this script from the top folder."""

import torch

import datetime
import time
import os
import numpy as np
import pickle

import forest
from forest.filtering_defenses import get_defense
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()
# 100% reproducibility?
if args.deterministic:
    forest.utils.set_deterministic()

subfolder = args.modelsave_path
clean_path = os.path.join(subfolder, 'clean_model')
def_model_path = os.path.join(subfolder, 'defended_model')
os.makedirs(clean_path, exist_ok=True)
os.makedirs(def_model_path, exist_ok=True)

def get_features(model, data, poison_delta):
    feats = np.array([])
    targets = []
    indices = []
    layer_cake = list(model.model.children())
    feature_extractor = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
    with torch.no_grad():
        for i, (img, target, idx) in enumerate(data.trainset):
            lookup = data.poison_lookup.get(idx)
            if lookup is not None and poison_delta is not None:
                img += poison_delta[lookup, :, :, :]
            img = img.unsqueeze(0).to(**data.setup)
            f = feature_extractor(img).detach().cpu().numpy()
            if i == 0:
                feats = np.copy(f)
            else:
                feats = np.append(feats, f, axis=0)
#             if i%1000==0:
#                 print(i)
            targets.append(target)
            indices.append(idx)

        for enum, (img, target, idx) in enumerate(data.targetset):
            targets.append(target)
            indices.append('target')
            img = img.unsqueeze(0).to(**data.setup)
            f = feature_extractor(img).detach().cpu().numpy()
            feats = np.append(feats, f, axis=0)
    return feats, targets, indices

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

    torch.save(model.model.state_dict(), os.path.join(clean_path, 'clean.pth'))
    model.model.eval()
    feats, targets, indices = get_features(model, data, poison_delta=None)
    with open(os.path.join(clean_path, 'clean_features.pickle'), 'wb+') as file:
        pickle.dump([feats, targets, indices], file, protocol=pickle.HIGHEST_PROTOCOL)
    model.model.train()

    poison_delta = witch.brew(model, data)
    brew_time = time.time()
    with open(os.path.join(subfolder, 'poison_indices.pickle'), 'wb+') as file:
        pickle.dump(data.poison_ids, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Poison ids saved')

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
    torch.save(model.model.state_dict(), os.path.join(def_model_path, 'def.pth'))
    model.model.eval()
    feats, targets, indices = get_features(model, data, poison_delta=poison_delta)
    with open(os.path.join(def_model_path, 'def_features.pickle'), 'wb+') as file:
        pickle.dump([feats, targets, indices], file, protocol=pickle.HIGHEST_PROTOCOL)
    model.model.train()

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
