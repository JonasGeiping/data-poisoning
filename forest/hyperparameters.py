"""Optimization setups."""

from dataclasses import dataclass, asdict

BRITTLE_NETS = ['convnet', 'mobilenet', 'vgg', 'alexnet']  # handled with lower learning rate

def training_strategy(model_name, args):
    """Parse training strategy."""
    if args.optimization == 'conservative':
        defaults = CONSERVATIVE
    elif args.optimization == 'private-gaussian':
        defaults = PRIVACY_GAUSSIAN
    elif args.optimization == 'private-laplacian':
        defaults = PRIVACY_LAPLACIAN
    elif args.optimization == 'adversarial':
        defaults = ADVERSARIAL
    elif args.optimization == 'basic':
        defaults = BASIC
    elif args.optimization == 'defensive':
        defaults = DEFENSE
    else:
        raise ValueError(f'Unknown opt. strategy {args.optimization}.')
    defs = Hyperparameters(**defaults.asdict())

    # Overwrite some hyperparameters from args
    if args.epochs is not None:
        defs.epochs = args.epochs
    if args.lr is not None:
        defs.lr = args.lr
    if args.noaugment:
        defs.augmentations = False
    else:
        defs.augmentations = args.data_aug
    if any(net in model_name.lower() for net in BRITTLE_NETS):
        defs.lr *= 0.1

    # Modifications to gradient noise settings
    if args.gradient_noise is not None:
        defs.privacy['noise'] = args.gradient_noise
    if args.gradient_clip is not None:
        defs.privacy['clip'] = args.gradient_clip

    # Modifications to defense settings:
    if args.defense_type is not None:
        defs.novel_defense['type'] = args.defense_type
    if args.defense_strength is not None:
        defs.novel_defense['strength'] = args.defense_strength
    else:
        defs.novel_defense['strength'] = args.eps
    if args.defense_targets is not None:
        defs.novel_defense['target_selection'] = args.defense_targets
    if args.defense_steps is not None:
        defs.novel_defense['steps'] = args.adversarial_steps

    # Modify data mixing arguments:
    if args.mixing_method is not None:
        defs.mixing_method['type'] = args.mixing_method

    defs.mixing_method['correction'] = args.mixing_disable_correction

    if args.mixing_strength is not None:
        defs.mixing_method['strength'] = args.mixing_strength

    # Modify defense behavior
    defs.adaptive_attack = args.disable_adaptive_attack
    defs.defend_features_only = args.defend_features_only

    return defs


@dataclass
class Hyperparameters:
    """Hyperparameters used by this framework."""

    name : str

    epochs : int
    batch_size : int
    optimizer : str
    lr : float
    scheduler : str
    weight_decay : float
    augmentations : bool
    privacy : dict
    validate : int
    novel_defense: dict
    mixing_method : dict
    adaptive_attack : bool
    defend_features_only: bool

    def asdict(self):
        return asdict(self)


CONSERVATIVE = Hyperparameters(
    name='conservative',
    lr=0.1,
    epochs=40,
    batch_size=128,
    optimizer='SGD',
    scheduler='linear',
    weight_decay=5e-4,
    augmentations=True,
    privacy=dict(clip=None, noise=None, distribution=None),
    validate=10,
    novel_defense=dict(type='', strength=16.0, target_selection='sep-half', steps=5),
    mixing_method=dict(type='', strength=0.0, correction=False),
    adaptive_attack=True,
    defend_features_only=False,
)


PRIVACY_GAUSSIAN = Hyperparameters(
    name='private-gaussian',
    lr=0.1,
    epochs=40,
    batch_size=128,
    optimizer='SGD',
    scheduler='linear',
    weight_decay=5e-4,
    augmentations=True,
    privacy=dict(clip=1.0, noise=0.01, distribution='gaussian'),
    validate=10,
    novel_defense=dict(type='', strength=16.0, target_selection='sep-half', steps=5),
    mixing_method=dict(type='', strength=0.0, correction=False),
    adaptive_attack=True,
    defend_features_only=False,
)


PRIVACY_LAPLACIAN = Hyperparameters(
    name='private-gaussian',
    lr=0.1,
    epochs=40,
    batch_size=128,
    optimizer='SGD',
    scheduler='linear',
    weight_decay=5e-4,
    augmentations=True,
    privacy=dict(clip=1.0, noise=0.01, distribution='laplacian'),
    validate=10,
    novel_defense=dict(type='', strength=16.0, target_selection='sep-half', steps=5),
    mixing_method=dict(type='', strength=0.0, correction=False),
    adaptive_attack=True,
    defend_features_only=False,
)

"""Most simple stochastic gradient descent.

This setup resembles the training procedure in MetaPoison.
"""
BASIC = Hyperparameters(
    name='basic',
    lr=0.1,
    epochs=80,
    batch_size=128,
    optimizer='SGD-basic',
    scheduler='none',
    weight_decay=0,
    augmentations=False,
    privacy=dict(clip=None, noise=None, distribution=None),
    validate=10,
    novel_defense=dict(type='', strength=16.0, target_selection='sep-half', steps=5),
    mixing_method=dict(type='', strength=0.0, correction=False),
    adaptive_attack=True,
    defend_features_only=False,
)


"""Implement adversarial training to defend against the poisoning."""
ADVERSARIAL = Hyperparameters(
    name='adversarial',
    lr=0.1,
    epochs=40,
    batch_size=128,
    optimizer='SGD',
    scheduler='linear',
    weight_decay=5e-4,
    augmentations=True,
    privacy=dict(clip=None, noise=None, distribution=None),
    validate=10,
    novel_defense=dict(type='adversarial-evasion', strength=8.0, target_selection='sep-p128', steps=5),
    mixing_method=dict(type='', strength=0.0, correction=False),
    adaptive_attack=True,
    defend_features_only=False,
)

"""Implement novel defensive training to defend against the poisoning.
Supply modifications via extra args"""
DEFENSE = Hyperparameters(
    name='noveldefense',
    lr=0.1,
    epochs=40,
    batch_size=128,
    optimizer='SGD',
    scheduler='linear',
    weight_decay=5e-4,
    augmentations=True,
    privacy=dict(clip=None, noise=None, distribution=None),
    validate=10,
    novel_defense=dict(type='adversarial-wb-recombine', strength=16.0, target_selection='sep-half', steps=5),
    mixing_method=dict(type='', strength=0.0, correction=False),
    adaptive_attack=True,
    defend_features_only=False,
)
