"""Implement an ArgParser common to both brew_poison.py and dist_brew_poison.py ."""

import argparse

def options():
    """Construct the central argument parser, filled with useful defaults.

    The first block is essential to test poisoning in different scenarios.
    The options following afterwards change the algorithm in various ways and are set to reasonable defaults.
    """
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')


    ###########################################################################
    # Central:
    parser.add_argument('--net', default='ResNet18', type=lambda s: [str(item) for item in s.split(',')])
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNet1k', 'MNIST', 'TinyImageNet'])
    parser.add_argument('--recipe', default='gradient-matching', type=str, choices=['gradient-matching', 'gradient-matching-private', 'gradient-matching-mt',
                                                                                    'watermark', 'poison-frogs', 'metapoison', 'hidden-trigger',
                                                                                    'metapoison-v2', 'metapoison-v3', 'bullseye', 'patch',
                                                                                    'gradient-matching-hidden', 'convex-polytope'])
    parser.add_argument('--threatmodel', default='single-class', type=str, choices=['single-class', 'third-party', 'random-subset'])
    parser.add_argument('--scenario', default='from-scratch', type=str, choices=['from-scratch', 'transfer', 'finetuning'])

    # Reproducibility management:
    parser.add_argument('--poisonkey', default=None, type=str, help='Initialize poison setup with this key.')  # Also takes a triplet 0-3-1
    parser.add_argument('--modelkey', default=None, type=int, help='Initialize the model with this key.')
    parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')

    # Poison properties / controlling the strength of the attack:
    parser.add_argument('--eps', default=16, type=float, help='Epsilon bound of the attack in a ||.||_p norm. p=Inf for all recipes except for "patch".')
    parser.add_argument('--budget', default=0.01, type=float, help='Fraction of training data that is poisoned')
    parser.add_argument('--targets', default=1, type=int, help='Number of targets')
    parser.add_argument('--patch_size', default=8, type=int, help='Size of patch to be added to test data (htbd attack)')
    # Files and folders
    parser.add_argument('--name', default='', type=str, help='Name tag for the result table and possibly for export folders.')
    parser.add_argument('--table_path', default='tables/', type=str)
    parser.add_argument('--poison_path', default='poisons/', type=str)
    parser.add_argument('--data_path', default='~/data', type=str)
    parser.add_argument('--modelsave_path', default='./models/', type=str)
    ###########################################################################

    # Mixing defense
    parser.add_argument('--mixing_method', default=None, type=str, help='Which mixing data augmentation to use.')
    parser.add_argument('--mixing_disable_correction', action='store_false', help='Disable correcting the loss term appropriately after data mixing.')
    parser.add_argument('--mixing_strength', default=None, type=float, help='How strong is the mixing.')

    parser.add_argument('--disable_adaptive_attack', action='store_false', help='Do not use a defended model as input for poisoning. [Defend only in poison validation]')
    parser.add_argument('--defend_features_only', action='store_true', help='Only defend during the initial pretraining before poisoning. [Defend only in pretraining]')
    # Note: If --disable_adaptive_attack and --defend_features_only, then the defense is never activated

    # Privacy defenses
    parser.add_argument('--gradient_noise', default=None, type=float, help='Add custom gradient noise during training.')
    parser.add_argument('--gradient_clip', default=None, type=float, help='Add custom gradient clip during training.')

    # Adversarial defenses
    parser.add_argument('--defense_type', default=None, type=str, help='Add custom novel defenses.')
    parser.add_argument('--defense_strength', default=None, type=float, help='Add custom strength to novel defenses.')
    parser.add_argument('--defense_steps', default=None, type=int, help='Override default number of adversarial steps taken by the defense.')
    parser.add_argument('--defense_targets', default=None, type=str, help='Different choices for target selection. Options: shuffle/sep-half/sep-1/sep-10')

    # Filter defenses
    parser.add_argument('--filter_defense', default='', type=str, help='Which filtering defense to use.', choices=['spectral_signatures', 'deepknn', 'activation_clustering'])

    # Adaptive attack variants
    parser.add_argument('--padversarial', default=None, type=str, help='Use adversarial steps during poison brewing.')
    parser.add_argument('--pmix', action='store_true', help='Use mixing during poison brewing [Uses the mixing specified in mixing_type].')

    # Poison brewing:
    parser.add_argument('--attackoptim', default='signAdam', type=str)
    parser.add_argument('--attackiter', default=250, type=int)
    parser.add_argument('--init', default='randn', type=str)  # randn / rand
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--scheduling', action='store_false', help='Disable step size decay.')
    parser.add_argument('--target_criterion', default='cross-entropy', type=str, help='Loss criterion for target loss')
    parser.add_argument('--restarts', default=8, type=int, help='How often to restart the attack.')
    parser.add_argument('--load_patch', default='', type=str, help='Path to load image for patch attack')
    parser.add_argument('--pbatch', default=512, type=int, help='Poison batch size during optimization')
    parser.add_argument('--pshuffle', action='store_true', help='Shuffle poison batch during optimization')
    parser.add_argument('--paugment', action='store_false', help='Do not augment poison batch during optimization')
    parser.add_argument('--data_aug', type=str, default='default', help='Mode of diff. data augmentation.')

    # Poisoning algorithm changes
    parser.add_argument('--full_data', action='store_true', help='Use full train data (instead of just the poison images)')
    parser.add_argument('--ensemble', default=1, type=int, help='Ensemble of networks to brew the poison on')
    parser.add_argument('--stagger', default=None, type=str, help='Stagger the network ensemble if it exists', choices=['firstn', 'full', 'inbetween'])
    parser.add_argument('--step', action='store_true', help='Optimize the model for one epoch.')
    parser.add_argument('--max_epoch', default=None, type=int, help='Train only up to this epoch before poisoning.')

    # Use only a subset of the dataset:
    parser.add_argument('--ablation', default=1.0, type=float, help='What percent of data (including poisons) to use for validation')

    # Gradient Matching - Specific Options
    parser.add_argument('--loss', default='similarity', type=str)  # similarity is stronger in  difficult situations

    # These are additional regularization terms for gradient matching. We do not use them, but it is possible
    # that scenarios exist in which additional regularization of the poisoned data is useful.
    parser.add_argument('--centreg', default=0, type=float)
    parser.add_argument('--normreg', default=0, type=float)
    parser.add_argument('--repel', default=0, type=float)

    # Specific Options for a metalearning recipe
    parser.add_argument('--nadapt', default=2, type=int, help='Meta unrolling steps')
    parser.add_argument('--clean_grad', action='store_true', help='Compute the first-order poison gradient.')

    # Validation behavior
    parser.add_argument('--vruns', default=1, type=int, help='How often to re-initialize and check target after retraining')
    parser.add_argument('--vnet', default=None, type=lambda s: [str(item) for item in s.split(',')], help='Evaluate poison on this victim model. Defaults to --net')
    parser.add_argument('--retrain_from_init', action='store_true', help='Additionally evaluate by retraining on the same model initialization.')
    parser.add_argument('--skip_clean_training', action='store_true', help='Skip clean training. This is only suggested for attacks that do not depend on a clean model.')

    # Optimization setup
    parser.add_argument('--pretrained_model', action='store_true', help='Load pretrained models from torchvision, if possible [only valid for ImageNet].')

    # Pretrain on a different dataset. This option is only relevant for finetuning/transfer:
    parser.add_argument('--pretrain_dataset', default=None, type=str, choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNet1k', 'MNIST', 'TinyImageNet'])
    parser.add_argument('--optimization', default='conservative', type=str, help='Optimization Strategy')
    # Strategy overrides:
    parser.add_argument('--epochs', default=None, type=int, help='Override default epochs of --optimization strategy')
    parser.add_argument('--lr', default=None, type=float, help='Override default learning rate of --optimization strategy')
    parser.add_argument('--noaugment', action='store_true', help='Do not use data augmentation during training.')

    # Optionally, datasets can be stored as LMDB or within RAM:
    parser.add_argument('--lmdb_path', default=None, type=str)
    parser.add_argument('--cache_dataset', action='store_true', help='Cache the entire thing :>')

    # These options allow for testing against the toxicity benchmark found at
    # https://github.com/aks2203/poisoning-benchmark
    parser.add_argument('--benchmark', default='', type=str, help='Path to benchmarking setup (pickle file)')
    parser.add_argument('--benchmark_idx', default=0, type=int, help='Index of benchmark test')

    # Debugging:
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--save', default=None, help='Export poisons into a given format. Options are full/limited/automl/numpy.')

    # Distributed Computations
    parser.add_argument("--local_rank", default=None, type=int, help='Distributed rank. This is an INTERNAL ARGUMENT! '
                                                                     'Only the launch utility should set this argument!')

    return parser
