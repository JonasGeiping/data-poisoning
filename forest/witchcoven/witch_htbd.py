"""Main class, holding information about models and training/testing routines."""

import torch
import torchvision
from PIL import Image
from ..utils import bypass_last_layer
from ..consts import BENCHMARK, NON_BLOCKING
from forest.data import datasets
torch.backends.cudnn.benchmark = BENCHMARK
import random

from .witch_base import _Witch


class WitchHTBD(_Witch):
    def _run_trial(self, victim, kettle):
        """Run a single trial."""
        poison_delta = kettle.initialize_poison()
        if self.args.full_data:
            dataloader = kettle.trainloader
        else:
            dataloader = kettle.poisonloader

        validated_batch_size = max(min(kettle.args.pbatch, len(kettle.poisonset)), 1)
        self.temp_targetset = self._get_temp_targets(kettle)
        self.patch_temp_targets(kettle)
        '''
        num_workers = kettle.get_num_workers()
        targetloader = torch.utils.data.DataLoader(kettle.targetset, batch_size=validated_batch_size,
                                                   shuffle=True, drop_last=False, num_workers=num_workers,
                                                   pin_memory=PIN_MEMORY)
        tloader_iter = iter(targetloader)
        '''

        if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
            # poison_delta.requires_grad_()
            if self.args.attackoptim in ['Adam', 'signAdam']:
                att_optimizer = torch.optim.Adam([poison_delta], lr=self.tau0, weight_decay=0)
            else:
                att_optimizer = torch.optim.SGD([poison_delta], lr=self.tau0, momentum=0.9, weight_decay=0)
            if self.args.scheduling:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[self.args.attackiter // 2.667, self.args.attackiter // 1.6,
                                                                                            self.args.attackiter // 1.142], gamma=0.1)
            poison_delta.grad = torch.zeros_like(poison_delta)
            dm, ds = kettle.dm.to(device=torch.device('cpu')), kettle.ds.to(device=torch.device('cpu'))
            poison_bounds = torch.zeros_like(poison_delta)
        else:
            poison_bounds = None

        for step in range(self.args.attackiter):
            target_losses = 0
            poison_correct = 0
            for batch, example in enumerate(dataloader):
                targets, target_labels = [], []
                indcs = random.sample(list(range(len(self.temp_targetset))), validated_batch_size)
                for i in indcs:
                    temp_target, temp_label, _ = self.temp_targetset[i]
                    targets.append(temp_target)
                    # target_labels.append(temp_label)
                targets = torch.stack(targets)
                loss, prediction = self._batched_step(poison_delta, poison_bounds, example, victim, kettle, targets)
                target_losses += loss
                poison_correct += prediction

                if self.args.dryrun:
                    break

            # Note that these steps are handled batch-wise for PGD in _batched_step
            # For the momentum optimizers, we only accumulate gradients for all poisons
            # and then use optimizer.step() for the update. This is math. equivalent
            # and makes it easier to let pytorch track momentum.
            if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                if self.args.attackoptim in ['momPGD', 'signAdam']:
                    poison_delta.grad.sign_()
                att_optimizer.step()
                if self.args.scheduling:
                    scheduler.step()
                att_optimizer.zero_grad()
                with torch.no_grad():
                    # Projection Step
                    poison_delta.data = torch.max(torch.min(poison_delta, self.args.eps /
                                                            ds / 255), -self.args.eps / ds / 255)
                    poison_delta.data = torch.max(torch.min(poison_delta, (1 - dm) / ds -
                                                            poison_bounds), -dm / ds - poison_bounds)

            target_losses = target_losses / (batch + 1)
            poison_acc = poison_correct / len(dataloader.dataset)
            if step % (self.args.attackiter // 5) == 0 or step == (self.args.attackiter - 1):
                print(f'Iteration {step}: Target loss is {target_losses:2.4f}, '
                      f'Poison clean acc is {poison_acc * 100:2.2f}%')

            if self.args.step:
                if self.args.clean_grad:
                    victim.step(kettle, None, self.targets, self.true_classes)
                else:
                    victim.step(kettle, poison_delta, self.targets, self.true_classes)

            if self.args.dryrun:
                break

        return poison_delta, target_losses



    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle, targets):
        """Take a step toward minmizing the current target loss."""
        inputs, labels, ids = example

        inputs = inputs.to(**self.setup)
        targets = targets.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)

        # Check adversarial pattern ids
        poison_slices, batch_positions = kettle.lookup_poison_indices(ids)

        # This is a no-op in single network brewing
        # In distributed brewing, this is a synchronization operation
        inputs, labels, poison_slices, batch_positions, randgen = victim.distributed_control(
            inputs, labels, poison_slices, batch_positions)

        if len(batch_positions) > 0:
            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
            if self.args.clean_grad:
                delta_slice = torch.zeros_like(delta_slice)
            delta_slice.requires_grad_()  # TRACKING GRADIENTS FROM HERE
            poison_images = inputs[batch_positions]
            inputs[batch_positions] += delta_slice

            # Perform differentiable data augmentation
            if self.args.paugment:
                inputs = kettle.augment(inputs, randgen=randgen)

            # Perform mixing
            if self.args.pmix:
                inputs, extra_labels, mixing_lmb = kettle.mixer(inputs, labels)

            if self.args.padversarial is not None:
                delta = self.attacker.attack(inputs.detach(), labels, None, None, steps=5)  # the 5-step here is FOR TESTING ONLY
                inputs = inputs + delta  # Kind of a reparametrization trick


            # Define the loss objective and compute gradients
            if self.args.target_criterion in ['cw', 'carlini-wagner']:
                loss_fn = cw_loss
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
            # Change loss function to include corrective terms if mixing with correction
            if self.args.pmix:
                def criterion(outputs, labels):
                    loss, pred = kettle.mixer.corrected_loss(outputs, extra_labels, lmb=mixing_lmb, loss_fn=loss_fn)
                    return loss
            else:
                criterion = loss_fn

            closure = self._define_objective(inputs, labels, criterion, targets, self.intended_classes,
                                             self.true_classes)
            loss, prediction = victim.compute(closure, self.target_grad, self.target_clean_grad, self.target_gnorm)
            delta_slice = victim.sync_gradients(delta_slice)

            if self.args.clean_grad:
                delta_slice.data = poison_delta[poison_slices].detach().to(**self.setup)

            # Update Step
            if self.args.attackoptim in ['PGD', 'GD']:
                delta_slice = self._pgd_step(delta_slice, poison_images, self.tau0, kettle.dm, kettle.ds)

                # Return slice to CPU:
                poison_delta[poison_slices] = delta_slice.detach().to(device=torch.device('cpu'))
            elif self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                poison_delta.grad[poison_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
                poison_bounds[poison_slices] = poison_images.detach().to(device=torch.device('cpu'))
            else:
                raise NotImplementedError('Unknown attack optimizer.')
        else:
            loss, prediction = torch.tensor(0), torch.tensor(0)

        return loss.item(), prediction.item()

    def _define_objective(self, inputs, labels, criterion, targets, intended_classes, true_classes):
        """Implement the closure here."""
        def closure(model, optimizer, target_grad, target_clean_grad, target_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            input_indcs, target_indcs = self._index_mapping(model, inputs, targets)

            feature_model, last_layer = bypass_last_layer(model)
            new_inputs = torch.zeros_like(inputs)
            new_targets = torch.zeros_like(targets)
            for i in range(len(input_indcs)):
                new_inputs[i] = inputs[input_indcs[i]]
                new_targets[i] = targets[target_indcs[i]]

            outputs = feature_model(new_inputs)
            prediction = (last_layer(outputs).data.argmax(dim=1) == labels).sum()
            outputs_targets = feature_model(new_targets)
            prediction = (last_layer(outputs).data.argmax(dim=1) == labels).sum()
            feature_loss = (outputs - outputs_targets).pow(2).mean(dim=1).sum()
            feature_loss.backward(retain_graph=self.retain)
            return feature_loss.detach().cpu(), prediction.detach().cpu()
        return closure

    def _create_patch(self, patch_shape):
        temp_patch = 0.5 * torch.ones(3, patch_shape[1], patch_shape[2])
        patch = torch.bernoulli(temp_patch)
        return patch

    def patch_targets(self, kettle):
        if self.args.load_patch == '':
            # here args.eps is a standin for sqrt(l0) bounds on our patch :)
            patch = self._create_patch([3, int(self.args.patch_size), int(self.args.patch_size)])
        else:
            patch = Image.open(self.args.load_patch)
            totensor = torchvision.transforms.ToTensor()
            resize = torchvision.transforms.Resize(int(self.args.patch_size))
            patch = totensor(resize(patch))

        patch = patch.to(**kettle.setup) / kettle.ds
        self.patch = patch.squeeze(0)
        target_delta = []
        for idx, (target_img, label, image_id) in enumerate(kettle.targetset):
            target_img = target_img.to(**self.setup)
            delta_slice = torch.zeros_like(target_img).squeeze(0)
            diff_patch = self.patch - target_img[:, target_img.shape[1] - self.patch.shape[1]:, target_img.shape[2] - self.patch.shape[2]:]
            delta_slice[:, delta_slice.shape[1] - self.patch.shape[1]:, delta_slice.shape[2] - self.patch.shape[2]:] = diff_patch
            target_delta.append(delta_slice.cpu())
        kettle.targetset = datasets.Deltaset(kettle.targetset, target_delta)

    def patch_temp_targets(self, kettle):
        target_delta = []
        for idx, (target_img, label, image_id) in enumerate(self.temp_targetset):
            target_img = target_img.to(**self.setup)
            delta_slice = torch.zeros_like(target_img).squeeze(0)
            diff_patch = self.patch - target_img[:, target_img.shape[1] - self.patch.shape[1]:, target_img.shape[2] - self.patch.shape[2]:]
            delta_slice[:, delta_slice.shape[1] - self.patch.shape[1]:, delta_slice.shape[2] - self.patch.shape[2]:] = diff_patch
            target_delta.append(delta_slice.cpu())
        self.temp_targetset = datasets.Deltaset(self.temp_targetset, target_delta)

    def _get_temp_targets(self, kettle):
        indcs = []
        for i in range(len(kettle.trainset)):
            target, idx = kettle.trainset.get_target(i)
            if target == kettle.poison_setup['target_class']:
                indcs.append(idx)
        return torch.utils.data.Subset(kettle.trainset, indcs)


    def _index_mapping(self, model, inputs, temp_targets):
        with torch.no_grad():
            feature_model, last_layer = bypass_last_layer(model)
            feat_source = feature_model(inputs)
            feat_target = feature_model(temp_targets)
            dist = torch.cdist(feat_source, feat_target)
            input_indcs = []
            target_indcs = []
            for _ in range(feat_target.size(0)):
                dist_min_index = (dist == torch.min(dist)).nonzero(as_tuple=False).squeeze()
                if len(dist_min_index[0].shape) != 0:
                    input_indcs.append(dist_min_index[0][0])
                    target_indcs.append(dist_min_index[1][0])
                    dist[dist_min_index[0][0], dist_min_index[1][0]] = 1e5
                else:
                    input_indcs.append(dist_min_index[0])
                    target_indcs.append(dist_min_index[1])
                    dist[dist_min_index[0], dist_min_index[1]] = 1e5
        return input_indcs, target_indcs
