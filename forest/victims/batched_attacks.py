"""Implement batch-level attack steps."""

import torch
import higher
import random

from ..utils import _gradient_matching, bypass_last_layer


def construct_attack(novel_defense, model, loss_fn, dm, ds, tau, init, optim, num_classes, setup):
    """Interface for this submodule."""
    eps = novel_defense['strength']  # The defense parameter encodes the eps bound used during training
    if 'adversarial-evasion' in novel_defense['type']:
        return AdversarialAttack(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-wb' in novel_defense['type']:
        return AlignmentPoisoning(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-se' in novel_defense['type']:
        return MatchingPoisoning(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-mp' in novel_defense['type']:
        return MetaPoisoning(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-fc' in novel_defense['type'] or 'adversarial-cp' in novel_defense['type']:
        return FeatureCollisionPoisoning(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-random' in novel_defense['type']:
        return RandomAttack(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-laplacian' in novel_defense['type']:
        return RandomAttack(model, loss_fn, dm, ds, tau, eps, 'laplacian', optim, num_classes, setup)
    elif 'adversarial-bernoulli' in novel_defense['type']:
        return RandomAttack(model, loss_fn, dm, ds, tau, eps, 'bernoulli', optim, num_classes, setup)
    elif 'adversarial-watermark' in novel_defense['type']:
        return WatermarkPoisoning(model, loss_fn, dm, ds, setup=setup)
    elif 'adversarial-patch' in novel_defense['type']:
        return PatchAttack(model, loss_fn, dm, ds, tau, eps, 'zero', 'none', num_classes, setup)
    elif 'adversarial-paired-patch' in novel_defense['type']:
        return PatchAttackPairs(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-variant-patch' in novel_defense['type']:
        return PatchAttackVariant(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-eps-patch' in novel_defense['type']:
        return PatchAttackVariantKnownSize(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-image-patch' in novel_defense['type']:
        return PatchAttackImageBased(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-matched-patch' in novel_defense['type']:
        return PatchAttackFixedLocation(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-adaptive-patch' in novel_defense['type']:
        return AdaptivePatchAttack(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-adaptiveV2-patch' in novel_defense['type']:
        return AdaptivePatchAttack(model, loss_fn, dm, ds, tau, eps, 'bernoulli', 'Adam', num_classes, setup)
    elif 'adversarial-adaptiveUC-patch' in novel_defense['type']:
        return AdaptivePatchAttackUnconstrained(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-htbd' in novel_defense['type']:
        return HTBD(model, loss_fn, dm, ds, setup=setup)
    else:
        raise ValueError(f'Invalid adversarial training objective specified: {novel_defense["type"]}.')


class BaseAttack(torch.nn.Module):
    """Implement a variety of input-altering attacks."""

    def __init__(self, model, loss_fn, dm=(0, 0, 0), ds=(1, 1, 1), tau=0.1, eps=16, init='zero',
                 optim='signAdam', num_classes=10, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize with dict containing type and strength of attack and model info."""
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.setup = setup
        self.num_classes = num_classes

        self.dm, self.ds, = dm, ds
        self.tau, self.eps = tau, eps
        self.bound = self.eps / self.ds / 255

        self.init = init
        self.optim = optim

    def attack(self, inputs, labels, temp_targets, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        if delta is None:
            delta = self._init_perturbation(inputs.shape)
        optimizer = self._init_optimizer([delta])

        for step in range(steps):
            optimizer.zero_grad()
            # Gradient step
            loss = self._objective(inputs + delta, labels, temp_targets, temp_fake_labels)
            delta.grad, = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False, only_inputs=True)
            # Optim step
            if 'sign' in self.optim:
                delta.grad.sign_()
            optimizer.step()
            # Projection step
            with torch.no_grad():
                delta.data = torch.max(torch.min(delta, self.bound), -self.bound)
                delta.data = torch.max(torch.min(delta, (1 - self.dm) / self.ds - inputs), - self.dm / self.ds - inputs)

        delta.requires_grad = False
        additional_info = None
        return delta, additional_info

    def _objective(self, inputs, labels, temp_targets, temp_fake_labels):
        raise NotImplementedError()

    def _init_perturbation(self, input_shape):
        if self.init == 'zero':
            delta = torch.zeros(input_shape, device=self.setup['device'], dtype=self.setup['dtype'])
        elif self.init == 'rand':
            delta = (torch.rand(input_shape, device=self.setup['device'], dtype=self.setup['dtype']) - 0.5) * 2
            delta *= self.eps / self.ds / 255
        elif self.init == 'bernoulli':
            delta = (torch.rand(input_shape, device=self.setup['device'], dtype=self.setup['dtype']) > 0.5).float() * 2 - 1
            delta *= self.eps / self.ds / 255
        elif self.init == 'randn':
            delta = torch.randn(input_shape, device=self.setup['device'], dtype=self.setup['dtype'])
            delta *= self.eps / self.ds / 255
        elif self.init == 'laplacian':
            loc = torch.as_tensor(0.0, device=self.setup['device'])
            scale = torch.as_tensor(self.eps / self.ds / 255, device=self.setup['device']).mean()
            generator = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            delta = generator.sample(input_shape)
        elif self.init == 'normal':
            delta = torch.randn(input_shape, device=self.setup['device'], dtype=self.setup['dtype'])
        else:
            raise ValueError(f'Invalid init {self.init} given.')

        # Clamp initialization in all cases
        delta.data = torch.max(torch.min(delta, self.eps / self.ds / 255), -self.eps / self.ds / 255)
        delta.requires_grad_()
        return delta

    def _init_optimizer(self, delta_iterable):
        tau_sgd = (self.bound * self.tau).mean()
        if 'Adam' in self.optim:
            return torch.optim.Adam(delta_iterable, lr=self.tau, weight_decay=0)
        elif 'momSGD' in self.optim:
            return torch.optim.SGD(delta_iterable, lr=tau_sgd, momentum=0.9, weight_decay=0)
        else:
            return torch.optim.SGD(delta_iterable, lr=tau_sgd, momentum=0.0, weight_decay=0)


class AdversarialAttack(BaseAttack):
    """Implement a basic untargeted attack objective."""

    def _objective(self, inputs, labels, temp_targets, temp_labels):
        """Evaluate negative CrossEntropy for a gradient ascent."""
        outputs = self.model(inputs)
        loss = -self.loss_fn(outputs, labels)
        return loss


class RandomAttack(BaseAttack):
    """Sanity check: do not actually attack - just use the random initialization."""

    def attack(self, inputs, labels, temp_targets, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        if delta is None:
            delta = self._init_perturbation(inputs.shape)

        # skip optimization
        pass

        delta.requires_grad = False
        return delta, None


class WatermarkPoisoning(BaseAttack):
    """Sanity check: attack by watermarking."""

    def attack(self, inputs, labels, temp_targets, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective. This is effectively a slight mixing.

        with mixing factor lmb = 1 - eps / 255.
        """
        img_shape = temp_targets.shape[1:]
        num_targets = temp_targets.shape[0]
        num_inputs = inputs.shape[0]

        # Place
        if num_targets == num_inputs:
            delta = temp_targets - inputs
        elif num_targets < num_inputs:
            delta = temp_targets.repeat(num_inputs // num_targets, 1, 1, 1)[:num_inputs] - inputs
        else:
            factor = num_targets // num_inputs
            delta = temp_targets[:(factor * num_targets)].reshape(num_inputs, -1, *img_shape).mean(dim=1) - inputs
        delta *= self.eps / self.ds / 255

        return delta, None


class AlignmentPoisoning(BaseAttack):
    """Implement limited steps for data poisoning via gradient alignment."""

    def _objective(self, inputs, labels, temp_targets, temp_fake_labels):
        """Evaluate Gradient Alignment and descend."""
        differentiable_params = [p for p in self.model.parameters() if p.requires_grad]

        poison_loss = self.loss_fn(self.model(inputs), labels)
        poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True)

        target_loss = self.loss_fn(self.model(temp_targets), temp_fake_labels)
        target_grad = torch.autograd.grad(target_loss, differentiable_params, retain_graph=True, create_graph=True)

        return _gradient_matching(poison_grad, target_grad)


class MatchingPoisoning(BaseAttack):
    """Implement limited steps for data poisoning via gradient alignment."""

    def _objective(self, inputs, labels, temp_targets, temp_fake_labels):
        """Evaluate Gradient Alignment and descend."""
        differentiable_params = [p for p in self.model.parameters() if p.requires_grad]

        poison_loss = self.loss_fn(self.model(inputs), labels)
        poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True)

        target_loss = self.loss_fn(self.model(temp_targets), temp_fake_labels)
        target_grad = torch.autograd.grad(target_loss, differentiable_params, retain_graph=True, create_graph=True)

        objective, tnorm = 0, 0
        for pgrad, tgrad in zip(poison_grad, target_grad):
            objective += 0.5 * (tgrad - pgrad).pow(2).sum()
            tnorm += tgrad.detach().pow(2).sum()
        return objective / tnorm.sqrt()  # tgrad is a constant normalization factor as in witch_matching


class MetaPoisoning(BaseAttack):
    """Implement limited steps for data poisoning via MetaPoison."""

    NADAPT = 2

    def _objective(self, inputs, labels, temp_targets, temp_fake_labels):
        """Evaluate Metapoison."""
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.1)
        # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
        with higher.innerloop_ctx(self.model, optimizer, copy_initial_weights=False) as (fmodel, fopt):
            for _ in range(self.NADAPT):
                outputs = fmodel(inputs)
                poison_loss = self.loss_fn(outputs, labels)

                fopt.step(poison_loss)

        prediction = (outputs.data.argmax(dim=1) == labels).sum()
        # model.eval()
        target_loss = self.loss_fn(fmodel(temp_targets), temp_fake_labels)
        return target_loss


class FeatureCollisionPoisoning(BaseAttack):
    """Implement limited steps for data poisoning via feature collision (with the bullseye polytope variant)."""

    def _objective(self, inputs, labels, temp_targets, temp_labels):
        """Evaluate Gradient Alignment and descend."""
        feature_model, last_layer = bypass_last_layer(self.model)

        # Get standard output:
        outputs = feature_model(inputs)
        outputs_targets = feature_model(temp_targets)

        return (outputs.mean(dim=0) - outputs_targets.mean(dim=0)).pow(2).mean()


class HTBD(BaseAttack):
    """Implement limited steps for data poisoning via hidden trigger backdoor.

    Note that this attack modifies temp_targets as a side-effect!
    """

    def attack(self, inputs, labels, temp_targets, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        if delta is None:
            delta = self._init_perturbation(inputs.shape)
        optimizer = self._init_optimizer([delta])

        temp_targets = self._apply_patch(temp_targets)
        for step in range(steps):
            input_indcs, target_indcs = self._index_mapping(inputs, temp_targets)
            optimizer.zero_grad()
            # Gradient step
            loss = self._objective(inputs + delta, temp_targets, input_indcs, target_indcs)
            delta.grad, = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False, only_inputs=True)
            # Optim step
            if 'sign' in self.optim:
                delta.grad.sign_()
            optimizer.step()
            # Projection step
            with torch.no_grad():
                delta.data = torch.max(torch.min(delta, self.bound), -self.bound)
                delta.data = torch.max(torch.min(delta, (1 - self.dm) / self.ds - inputs), - self.dm / self.ds - inputs)

        delta.requires_grad = False
        return delta, None

    def _objective(self, inputs, temp_targets, input_indcs, target_indcs):
        """Evaluate Gradient Alignment and descend."""
        feature_model, last_layer = bypass_last_layer(self.model)
        new_inputs = torch.zeros_like(inputs)
        new_targets = torch.zeros_like(temp_targets)
        for i in range(len(input_indcs)):
            new_inputs[i] = inputs[input_indcs[i]]
            new_targets[i] = temp_targets[target_indcs[i]]

        outputs = feature_model(new_inputs)
        outputs_targets = feature_model(new_targets)
        return (outputs - outputs_targets).pow(2).mean(dim=1).sum()

    def _apply_patch(self, temp_targets):
        patch_shape = [[3, random.randint(int(0.2 * temp_targets.shape[2]), int(0.4 * temp_targets.shape[2])),
                        random.randint(int(0.1 * temp_targets.shape[3]), int(0.2 * temp_targets.shape[3]))] for _ in range(temp_targets.shape[0])]

        patch = self._create_patch(patch_shape)
        patch = [p.to(**self.setup) for p in patch]
        x_locations, y_locations = self._set_locations(temp_targets.shape, patch_shape)
        for i in range(len(patch)):
            temp_targets[i, :, x_locations[i]:x_locations[i] + patch_shape[i][1], y_locations[i]:y_locations[i]
                         + patch_shape[i][2]] = patch[i] - temp_targets[i, :, x_locations[i]:x_locations[i] + patch_shape[i][1],
                                                                        y_locations[i]:y_locations[i] + patch_shape[i][2]]
        return temp_targets

    def _index_mapping(self, inputs, temp_targets):
        with torch.no_grad():
            feature_model, last_layer = bypass_last_layer(self.model)
            feat_source = feature_model(inputs)
            feat_target = feature_model(temp_targets)
            dist = torch.cdist(feat_source, feat_target)
            input_indcs = []
            target_indcs = []
            for _ in range(feat_target.size(0)):
                dist_min_index = (dist == torch.min(dist)).nonzero(as_tuple=False).squeeze()
                input_indcs.append(dist_min_index[0])
                target_indcs.append(dist_min_index[1])
                dist[dist_min_index[0], dist_min_index[1]] = 1e5
        return input_indcs, target_indcs

    def _set_locations(self, input_shape, patch_shape):
        """Fix locations where we’ll put the patches."""
        x_locations = []
        y_locations = []
        for i in range(input_shape[0]):
            x_locations.append(random.randint(0, input_shape[2] - patch_shape[i][1]))
            y_locations.append(random.randint(0, input_shape[3] - patch_shape[i][2]))
        return x_locations, y_locations

    def _create_patch(self, patch_shape):
        # create same patch or different one?
        patches = []
        for i in range(len(patch_shape)):
            temp_patch = 0.5 * torch.ones(patch_shape[i][0], patch_shape[i][1], patch_shape[i][2])
            patch = torch.bernoulli(temp_patch)
            patches.append(patch)
        return patches


class PatchAttack(BaseAttack):
    """Randomly patch 2 classes."""

    def attack(self, inputs, labels, temp_targets, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        patch_shape = [[3, random.randint(int(0.1 * inputs.shape[2]), int(0.4 * inputs.shape[2])),
                        random.randint(int(0.1 * inputs.shape[3]), int(0.4 * inputs.shape[3]))] for _ in range(self.num_classes)]

        x_locations, y_locations = self._set_locations(inputs.shape, labels, patch_shape)
        # Maybe different patch per class?
        patch = self._create_patch(patch_shape)

        if delta is None:
            delta1 = self._init_perturbation(inputs.shape)
            delta2 = self._init_perturbation(temp_targets.shape)
        delta1.requires_grad = False
        delta2.requires_grad = False

        for i in range(delta1.shape[0]):
            # Patch every class
            temp_label = labels[i]
            delta1[i, :, x_locations[i]:x_locations[i] + patch_shape[temp_label][1], y_locations[i]:y_locations[i]
                   + patch_shape[temp_label][2]] = patch[temp_label] - inputs[i, :, x_locations[i]:x_locations[i]
                                                                              + patch_shape[temp_label][1], y_locations[i]:y_locations[i]
                                                                              + patch_shape[temp_label][2]]

        # Maybe different patch per class?
        # patch = [self._create_patch(patch_shape).to(**self.setup) for _ in range(num_classes)]
        permute_list = self._random_derangement(self.num_classes)
        temp_target_labels = [permute_list[temp_true_label] for temp_true_label in temp_true_labels]
        x_locations, y_locations = self._set_locations(temp_targets.shape, temp_target_labels, patch_shape)
        for i in range(delta2.shape[0]):
            temp_label = permute_list[temp_true_labels[i]]
            delta2[i, :, x_locations[i]:x_locations[i] + patch_shape[temp_label][1], y_locations[i]:y_locations[i]
                   + patch_shape[temp_label][2]] = patch[temp_label] - temp_targets[i, :, x_locations[i]:x_locations[i]
                                                                                    + patch_shape[temp_label][1], y_locations[i]:y_locations[i]
                                                                                    + patch_shape[temp_label][2]]
        #
        return [delta1, delta2]

    def _set_locations(self, input_shape, labels, patch_shape):
        """.Fix locations where we’ll put the patches."""
        x_locations = []
        y_locations = []
        for i in range(input_shape[0]):
            x_locations.append(random.randint(0, input_shape[2] - patch_shape[labels[i]][1]))
            y_locations.append(random.randint(0, input_shape[3] - patch_shape[labels[i]][2]))
        return x_locations, y_locations

    def _create_patch(self, patch_shape):
        # create same patch or different one?
        patches = []
        for i in range(len(patch_shape)):
            param = random.random()
            # temp_patch = 0.5*torch.ones(patch_shape[i][0], patch_shape[i][1], patch_shape[i][2])
            temp_patch = param * torch.ones(patch_shape[i][0], patch_shape[i][1], patch_shape[i][2])
            patch = torch.bernoulli(temp_patch)
            patches.append(patch.to(**self.setup) / self.ds)
        return patches

    def _random_derangement(self, n):
        while True:
            v = [i for i in range(n)]
            for j in range(n - 1, -1, -1):
                p = random.randint(0, j)
                if v[p] == j:
                    break
                else:
                    v[j], v[p] = v[p], v[j]
            else:
                if v[0] != 0:
                    return v


class PatchAttackVariant(PatchAttack):
    """Randomly patch 2 classes."""

    def attack(self, inputs, labels, temp_targets, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        patch_shape = [[inputs.shape[1], random.randint(int(0.1 * inputs.shape[2]), int(0.4 * inputs.shape[2])),
                        random.randint(int(0.1 * inputs.shape[3]), int(0.4 * inputs.shape[3]))] for _ in range(self.num_classes)]

        permute_list = self._random_derangement(self.num_classes)
        temp_target_labels = [permute_list[temp_true_label] for temp_true_label in temp_true_labels]
        x_in, y_in = self._set_locations(inputs.shape, labels, patch_shape)
        x_t, y_t = self._set_locations(temp_targets.shape, temp_target_labels, patch_shape)

        patches = self._create_patch(patch_shape)

        inputs_mask = self._patch(inputs, labels, patches, x_in, y_in)
        targets_mask = self._patch(temp_targets, temp_target_labels, patches, x_t, y_t)

        return inputs_mask, targets_mask

    def _patch(self, inputs, labels, patches, x_locations, y_locations):
        outputs = torch.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            # Patch every class
            temp_label = labels[i]
            x, y = x_locations[i], y_locations[i]
            xo, yo = patches[temp_label].shape[2], patches[temp_label].shape[3]
            outputs[i, :, x:x + xo, y:y + yo] = patches[temp_label] - inputs[i, :, x:x + xo, y:y + yo]
        return outputs

class PatchAttackVariantKnownSize(PatchAttack):
    """Draw random patch shapes from the interval [0.5 * eps, 1.75 * eps].

    This interval is slightly biased: Its mean length is not eps, but:
    eps: interval:
    1   0.5
    2   2.0
    3   3.0
    4   4.5
    5   5.0
    6   6.5
    7   7.5
    8   9.0
    9   9.5
    10  11.0
    11  12.0
    12  13.5
    13  14.0
    14  15.5
    15  16.5
    16  18.0
    17  18.5
    18  20.0
    19  20.9
    20  22.5
    """

    def attack(self, inputs, labels, temp_targets, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        patch_shape = [[inputs.shape[1], random.randint(int(0.5 * self.eps), int(1.75 * self.eps)),
                        random.randint(int(0.5 * self.eps), int(1.75 * self.eps))] for _ in range(self.num_classes)]
        permute_list = self._random_derangement(self.num_classes)
        temp_target_labels = [permute_list[temp_true_label] for temp_true_label in temp_true_labels]
        x_in, y_in = self._set_locations(inputs.shape, labels, patch_shape)
        x_t, y_t = self._set_locations(temp_targets.shape, temp_target_labels, patch_shape)

        patches = self._create_patch(patch_shape)

        inputs_mask = self._patch(inputs, labels, patches, x_in, y_in)
        targets_mask = self._patch(temp_targets, temp_target_labels, patches, x_t, y_t)

        return inputs_mask, targets_mask

    def _patch(self, inputs, labels, patches, x_locations, y_locations):
        outputs = torch.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            # Patch every class
            temp_label = labels[i]
            x, y = x_locations[i], y_locations[i]
            xo, yo = patches[temp_label].shape[2], patches[temp_label].shape[3]
            outputs[i, :, x:x + xo, y:y + yo] = patches[temp_label] - inputs[i, :, x:x + xo, y:y + yo]
        return outputs


class PatchAttackFixedLocation(PatchAttackVariant):
    """Randomly patch 2 classes."""

    def attack(self, inputs, labels, temp_targets, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        patch_shape = [[inputs.shape[1], random.randint(int(0.1 * inputs.shape[2]), int(0.4 * inputs.shape[2])),
                        random.randint(int(0.1 * inputs.shape[3]), int(0.4 * inputs.shape[3]))] for _ in range(self.num_classes)]

        permute_list = self._random_derangement(self.num_classes)
        temp_target_labels = [permute_list[temp_true_label] for temp_true_label in temp_true_labels]
        x, y = self._set_location(inputs.shape, patch_shape)

        patches = self._create_patch(patch_shape)

        inputs_mask = self._patch(inputs, labels, patches, x, y)
        targets_mask = self._patch(temp_targets, temp_target_labels, patches, x, y)
        return inputs_mask, targets_mask


    def _set_location(self, input_shape, patch_shape):
        """.Fix locations where we’ll put the patches."""
        x_locations = []
        y_locations = []
        for p in patch_shape:
            x_locations.append(random.randint(0, input_shape[2] - p[1]))
            y_locations.append(random.randint(0, input_shape[3] - p[2]))
        return x_locations, y_locations


    def _patch(self, inputs, labels, patches, x_locations, y_locations):
        outputs = torch.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            # Patch every class
            temp_label = labels[i]
            x, y = x_locations[labels[i]], y_locations[labels[i]]
            xo, yo = patches[temp_label].shape[2], patches[temp_label].shape[3]
            outputs[i, :, x:x + xo, y:y + yo] = patches[temp_label] - inputs[i, :, x:x + xo, y:y + yo]
        return outputs


class PatchAttackImageBased(PatchAttack):
    """Randomly patch classes with patches consisting of randomly drawn image patches from the inputs."""

    def attack(self, inputs, labels, temp_targets, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        patch_shape = [[inputs.shape[1], random.randint(int(0.1 * inputs.shape[2]), int(0.4 * inputs.shape[2])),
                        random.randint(int(0.1 * inputs.shape[3]), int(0.4 * inputs.shape[3]))] for _ in range(self.num_classes)]

        permute_list = self._random_derangement(self.num_classes)
        temp_target_labels = [permute_list[temp_true_label] for temp_true_label in temp_true_labels]
        x_in, y_in = self._set_locations(inputs.shape, labels, patch_shape)
        x_t, y_t = self._set_locations(temp_targets.shape, temp_target_labels, patch_shape)

        x_patch, y_patch = self._set_locations(inputs.shape, labels, patch_shape)
        patches = self._create_patch(inputs, patch_shape)

        inputs_mask = self._patch(inputs, labels, patches, x_in, y_in)
        targets_mask = self._patch(temp_targets, temp_target_labels, patches, x_t, y_t)

        return inputs_mask, targets_mask

    def _create_patch(self, inputs, patch_shapes):
        # create same patch by stealing a part of some input image
        patches = []
        for (c, xo, yo) in patch_shapes:
            img_source = inputs[torch.randint(0, inputs.shape[0], (1,))]

            x = torch.randint(0, inputs.shape[2] - xo, (1,))
            y = torch.randint(0, inputs.shape[3] - yo, (1,))
            patch = img_source[0, :, x:x + xo, y:y + yo]
            patches.append(patch.to(**self.setup))
        return patches


    def _patch(self, inputs, labels, patches, x_locations, y_locations):
        outputs = torch.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            # Patch every class
            temp_label = labels[i]
            x, y = x_locations[i], y_locations[i]
            xo, yo = patches[temp_label].shape[1], patches[temp_label].shape[2]

            outputs[i, :, x:x + xo, y:y + yo] = patches[temp_label] - inputs[i, :, x:x + xo, y:y + yo]
        return outputs


class AdaptivePatchAttack(PatchAttackVariant, AlignmentPoisoning):
    """Randomly patch pairs of classes as in Liam's implementation and optimize over these triggers."""

    def attack(self, inputs, labels, temp_targets, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        patch_shape = [[inputs.shape[1], random.randint(int(0.1 * inputs.shape[2]), int(0.4 * inputs.shape[2])),
                        random.randint(int(0.1 * inputs.shape[3]), int(0.4 * inputs.shape[3]))] for _ in range(self.num_classes)]

        permute_list = self._random_derangement(self.num_classes)
        temp_target_labels = torch.zeros_like(temp_true_labels)
        for idx, temp_true_label in enumerate(temp_true_labels):
            temp_target_labels[idx] = permute_list[temp_true_label]

        x_in, y_in = self._set_locations(inputs.shape, labels, patch_shape)
        x_t, y_t = self._set_locations(temp_targets.shape, temp_target_labels, patch_shape)

        patches = self._create_patch(patch_shape)
        patches = [p.requires_grad_() for p in patches]
        optimizer = self._init_optimizer(patches)

        for step in range(steps):
            optimizer.zero_grad()
            inputs_mask = self._patch(inputs, labels, patches, x_in, y_in)
            targets_mask = self._patch(temp_targets, temp_target_labels, patches, x_t, y_t)

            patched_inputs = inputs + inputs_mask
            patched_temp_targets = temp_targets + targets_mask
            loss = self._objective(patched_inputs, labels, patched_temp_targets, temp_target_labels)
            grads = torch.autograd.grad(loss, patches, retain_graph=False, create_graph=False, allow_unused=True)
            # Optim step
            for p, g in zip(patches, grads):
                if g is not None and 'sign' in self.optim:
                    p.grad = g.sign()
                else:
                    p.grad = g
            optimizer.step()
            # Projection step
            with torch.no_grad():
                for patch in patches:
                    patch.data = torch.max(torch.min(patch, (1 - self.dm) / self.ds), - self.dm / self.ds)

        with torch.no_grad():
            inputs_mask = self._patch(inputs, labels, patches, x_in, y_in)
            targets_mask = self._patch(temp_targets, temp_target_labels, patches, x_t, y_t)

        return inputs_mask, targets_mask


class AdaptivePatchAttackUnconstrained(AdaptivePatchAttack):
    """Randomly patch pairs of classes as in Liam's implementation and optimize over these triggers.

    Allow the target patch to differ from the input patch.
    """

    def attack(self, inputs, labels, temp_targets, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        patch_shape_inputs = [[inputs.shape[1], random.randint(int(0.1 * inputs.shape[2]), int(0.4 * inputs.shape[2])),
                               random.randint(int(0.1 * inputs.shape[3]), int(0.4 * inputs.shape[3]))] for _ in range(self.num_classes)]
        patch_shape_targets = [[inputs.shape[1], random.randint(int(0.1 * inputs.shape[2]), int(0.4 * inputs.shape[2])),
                                random.randint(int(0.1 * inputs.shape[3]), int(0.4 * inputs.shape[3]))] for _ in range(self.num_classes)]

        permute_list = self._random_derangement(self.num_classes)
        temp_target_labels = torch.zeros_like(temp_true_labels)
        for idx, temp_true_label in enumerate(temp_true_labels):
            temp_target_labels[idx] = permute_list[temp_true_label]

        x_in, y_in = self._set_locations(inputs.shape, labels, patch_shape_inputs)
        x_t, y_t = self._set_locations(temp_targets.shape, temp_target_labels, patch_shape_targets)

        patches_input = self._create_patch(patch_shape_inputs)
        patches_target = self._create_patch(patch_shape_targets)
        patches = [*patches_input, *patches_target]
        patches = [p.requires_grad_() for p in patches]
        optimizer = self._init_optimizer(patches)

        for step in range(steps):
            optimizer.zero_grad()
            inputs_mask = self._patch(inputs, labels, patches_input, x_in, y_in)
            targets_mask = self._patch(temp_targets, temp_target_labels, patches_target, x_t, y_t)

            patched_inputs = inputs + inputs_mask
            patched_temp_targets = temp_targets + targets_mask
            loss = self._objective(patched_inputs, labels, patched_temp_targets, temp_target_labels)
            grads = torch.autograd.grad(loss, patches, retain_graph=False, create_graph=False, allow_unused=True)
            # Optim step
            for p, g in zip(patches, grads):
                if g is not None and 'sign' in self.optim:
                    p.grad = g.sign()
                else:
                    p.grad = g
            optimizer.step()
            # Projection step
            with torch.no_grad():
                for patch in patches:
                    patch.data = torch.max(torch.min(patch, (1 - self.dm) / self.ds), - self.dm / self.ds)

        with torch.no_grad():
            inputs_mask = self._patch(inputs, labels, patches_input, x_in, y_in)
            targets_mask = self._patch(temp_targets, temp_target_labels, patches_target, x_t, y_t)

        return inputs_mask, targets_mask


class PatchAttackPairs(BaseAttack):
    """Randomly patch pairs of images."""

    def attack(self, inputs, labels, temp_targets, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        assert inputs.shape == temp_targets.shape
        mask = self._get_mask(inputs)

        self.init = 'bernoulli'  # enforce this?!?
        delta = self._init_perturbation(inputs.shape)
        delta.requires_grad = False

        return mask * (delta - inputs), mask * (delta - temp_targets)

    def _get_mask(self, inputs):
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        for example in range(inputs.shape[0]):
            x_length = random.randint(int(0.1 * inputs.shape[2]), int(0.4 * inputs.shape[2]))
            y_length = random.randint(int(0.1 * inputs.shape[3]), int(0.4 * inputs.shape[3]))
            x_pos = random.randint(0, inputs.shape[2] - x_length)
            y_pos = random.randint(0, inputs.shape[3] - y_length)
            mask[example, :, x_pos:(x_pos + x_length), y_pos:(y_pos + y_length)] = 1
        return mask
