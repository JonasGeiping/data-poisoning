"""Implement modules for mixup and its variants.

All forward methods respect the following signature:
Input: x, y [image-batch, label-batch]
Output: mixed_x, additional_labels, mixing_lambda

except for maxup, which returns n tuples of this output signature.
"""

import torch
import numpy as np


class Mixup(torch.nn.Module):
    """This is data augmentation via mixup. https://arxiv.org/abs/1710.09412."""

    def __init__(self, nway=2, alpha=1.0):
        """Implement differentiable mixup, mixing nway-many examples with the given mixing factor alpha."""
        super().__init__()
        self.nway = nway
        self.mixing_alpha = alpha


    def forward(self, x, y, epoch=None):
        if self.mixing_alpha > 0:
            lmb = np.random.dirichlet([self.mixing_alpha] * self.nway, size=1).tolist()[0]
            batch_size = x.shape[0]
            indices = [torch.randperm(batch_size, device=x.device) for _ in range(self.nway)]
            mixed_x = sum([l * x[index, :] for l, index in zip(lmb, indices)])
            y_s = [y[index] for index in indices]
        else:
            mixed_x = x
            y_s = y
            lmb = 1

        return mixed_x, y_s, lmb

    def corrected_loss(self, outputs, extra_labels, lmb=1.0, loss_fn=torch.nn.CrossEntropyLoss()):
        """Compute the corrected loss under consideration of the mixing."""
        predictions = torch.argmax(outputs.data, dim=1)
        correct_preds = sum([w * predictions.eq(l.data).sum().float().item() for w, l in zip(lmb, extra_labels)])
        loss = sum([weight * loss_fn(outputs, label) for weight, label in zip(lmb, extra_labels)])
        return loss, correct_preds


class Cutout(torch.nn.Module):
    """This is data augmentation via Cutout. https://arxiv.org/abs/1708.04552."""

    def __init__(self, alpha=1.0):
        """Cut-out with given alpha value.

        0.66 is CIFAR-specific so that # s.t. sqrt(1 - 0.66) * 28 approx 16
        """
        super().__init__()
        self.lmb = alpha * 0.66

    def forward(self, x, y, epoch=None):
        """run cutout."""
        # generate mixed sample
        rand_index = torch.randperm(x.shape[0], device=x.device)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), self.lmb)
        x[:, :, bbx1:bbx2, bby1:bby2] = torch.zeros_like(x)[rand_index, :, bbx1:bbx2, bby1:bby2]
        return x, y, None

    def corrected_loss(self, outputs, extra_labels, lmb=1.0, loss_fn=torch.nn.CrossEntropyLoss()):
        """Compute loss. This is just a normal loss for cutout."""
        predictions = torch.argmax(outputs.data, dim=1)
        correct_preds = predictions.eq(extra_labels.data).sum().float().item()
        loss = loss_fn(outputs, extra_labels)
        return loss, correct_preds

    @staticmethod
    def _rand_bbox(size, lmb):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lmb)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2


class Cutmix(Mixup, Cutout):
    """Implement cutmix, a data augmentation combining cutout and mixup. https://arxiv.org/abs/1905.04899.

    This is fixed to nway=2 for now and hardcodes the original cutmix modification (activating the augmentation randomly
    50% of the time)

    This class inherits the corrected loss from mixup!
    """

    def __init__(self, alpha=1.0):
        """Initialize with mixing factor alpha."""
        torch.nn.Module.__init__(self)
        self.alpha = alpha

    def forward(self, x, y, epoch=None):
        """run cutmix."""
        r = np.random.rand(1)
        if r < 0.5:
            # generate mixed sample
            lmb = np.random.beta(self.alpha, self.alpha)
            rand_index = torch.randperm(x.shape[0], device=x.device)
            labels_a = y
            labels_b = y[rand_index]
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lmb)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lmb = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        else:
            lmb = 1
            labels_a = y
            labels_b = y
        return x, [labels_a, labels_b], [lmb, 1 - lmb]


class Maxup(torch.nn.Module):
    """A meta-augmentation, returning the worst result from a range of augmentations.

    As in the orignal paper, https://arxiv.org/abs/2002.09024,
    this augmentation is not active for the first warm_up epochs.
    """

    def __init__(self, given_data_augmentation, ntrials=4, warmup_epochs=5):
        """Initialize with a given data augmentation module."""
        super().__init__()
        self.augment = given_data_augmentation
        self.ntrials = ntrials
        self.warmup_epochs = warmup_epochs

        self.max_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, y, epoch=None):
        if epoch > self.warmup_epochs:
            mixed_x, additional_labels, mixing_lambda = [], [], []
            for trial in range(self.ntrials):
                x_out, y_out, l_out = self.augment(x, y)
                mixed_x.append(x_out)
                additional_labels.append(y_out)
                mixing_lambda.append(l_out)

            mixed_x = torch.cat(mixed_x, dim=0)
            additional_labels = torch.cat(additional_labels, dim=0)
            mixing_lambda = torch.cat(mixing_lambda, dim=0) if mixing_lambda[0] is not None else None
            return mixed_x, additional_labels, mixing_lambda
        else:
            return x, y, None

    def corrected_loss(self, outputs, extra_labels, lmb=1.0, loss_fn=torch.nn.CrossEntropyLoss()):
        """Compute loss. Here the loss is computed as worst-case estimate over the trials."""
        batch_size = outputs.shape[0] // self.ntrials
        correct_preds = (torch.argmax(outputs.data, dim=1) == extra_labels).sum().item() / self.ntrials
        if lmb is not None:
            stacked_loss = self.max_criterion(outputs, extra_labels).view(batch_size, self.ntrials, -1)
            loss = stacked_loss.max(dim=1)[0].mean()
        else:
            loss = loss_fn(outputs, extra_labels)

        return loss, correct_preds
