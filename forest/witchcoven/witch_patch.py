"""Main class, holding information about models and training/testing routines."""

import torch
import torchvision
from PIL import Image
from ..consts import BENCHMARK
from forest.data import datasets
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch




class WitchPatch(_Witch):
    """Brew poison with given arguments.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _brew(self, victim, kettle):
        self._initialize_brew(victim, kettle)
        poison_delta = kettle.initialize_poison()
        for poison_id, (img, label, image_id) in enumerate(kettle.poisonset):
            poison_img = img.to(**self.setup)
            delta_slice = torch.zeros_like(poison_img)
            # hacky, but best way I can think of to not redo brewing
            diff_patch = self.patch - poison_img[:, poison_img.shape[1] - self.patch.shape[1]:, poison_img.shape[2] - self.patch.shape[2]:]
            delta_slice[:, delta_slice.shape[1] - self.patch.shape[1]:, delta_slice.shape[2] - self.patch.shape[2]:] = diff_patch
            poison_delta[poison_id] = delta_slice.cpu()

        # here we hackily add the patches to the target images
        return poison_delta.cpu()

    def _create_patch(self, patch_shape):
        temp_patch = 0.5 * torch.ones(3, patch_shape[1], patch_shape[2])
        patch = torch.bernoulli(temp_patch)
        return patch

    def patch_targets(self, kettle):
        if self.args.load_patch == '':
            # here args.eps is a standin for sqrt(l0) bounds on our patch :)
            patch = self._create_patch([3, int(self.args.eps), int(self.args.eps)])
        else:
            patch = Image.open(self.args.load_patch)
            totensor = torchvision.transforms.ToTensor()
            resize = torchvision.transforms.Resize(int(self.args.eps))
            patch = totensor(resize(patch))

        patch = patch.to(**self.setup) / kettle.ds
        self.patch = patch.squeeze(0)
        target_delta = []
        for idx, (target_img, label, image_id) in enumerate(kettle.targetset):
            target_img = target_img.to(**self.setup)
            delta_slice = torch.zeros_like(target_img).squeeze(0)
            diff_patch = self.patch - target_img[:, target_img.shape[1] - self.patch.shape[1]:, target_img.shape[2] - self.patch.shape[2]:]
            delta_slice[:, delta_slice.shape[1] - self.patch.shape[1]:, delta_slice.shape[2] - self.patch.shape[2]:] = diff_patch
            target_delta.append(delta_slice.cpu())
        kettle.targetset = datasets.Deltaset(kettle.targetset, target_delta)
