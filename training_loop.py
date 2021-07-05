"""Separate file showing only robust(er) training and data augmentations.

This is not runnable code, but a template to implement these defenses into your own code!

Several helper files from forest/ are imported below which have to be bundled when copying this snippet.
"""


import torch

from forest.victims.batched_attacks import construct_attack
from forest.data.mixing_data_augmentations import Cutmix

# hyperparameters:
epochs = 40
defense = dict(type='adversarial-wb', target_selection='sep-p96', steps=5)
mixing_method = dict(type='CutMix', correction=True, strength=1.0)
num_classes = 10

setup = dict(device=torch.device('cuda'), dtype=torch.float)


# Define model
# ...


# Define optimizer, dataloader and loss_fn
# ...



# Prepare data_mean and data_std
dm = torch.tensor(data_mean)[None, :, None, None].to(**setup)
ds = torch.tensor(data_std)[None, :, None, None].to(**setup)

# Prepare defense:
attacker = construct_attack(defense, model, loss_fn, dm, ds, tau=0.1, init='randn', optim='signAdam',
                            num_classes=num_classes, setup=setup)
mixer = Cutmix(alpha=mixing_method['strength'])


# Training loop:
for epoch in range(epochs):
    for batch, (inputs, labels, ids) in enumerate(dataloader):
        # Prep Mini-Batch
        # ...

        # Transfer to GPU
        # ...

        # Add basic data augmentation
        # ...


        # ###  Mixing defense ###
        if mixing_method['type'] != '':
            inputs, extra_labels, mixing_lmb = mixer(inputs, labels, epoch=epoch)

        # ### AT defense: ###
        # Split Data
        [temp_targets, inputs, temp_true_labels, labels, temp_fake_label] = _split_data(inputs, labels, p=0.75)
        # Apply poison attack
        model.eval()
        delta, additional_info = attacker.attack(inputs, labels, temp_targets, temp_true_labels, temp_fake_label,
                                                 steps=defense['steps'])
        # temp targets are modified for trigger attacks:
        if 'patch' in defense['type']:
            temp_targets = temp_targets + additional_info
        inputs = inputs + delta


        # Switch into training mode
        model.train()

        # Change loss function to include corrective terms if mixing with correction
        if (mixing_method['type'] != '' and mixing_method['correction']):
            def criterion(outputs, labels):
                return mixer.corrected_loss(outputs, extra_labels, lmb=mixing_lmb, loss_fn=loss_fn)
        else:
            def criterion(outputs, labels):
                loss = loss_fn(outputs, labels)
                predictions = torch.argmax(outputs.data, dim=1)
                correct_preds = (predictions == labels).sum().item()
                return loss, correct_preds


        # Recombine poisoned inputs and targets into a single batch
        inputs = torch.cat((inputs, temp_targets))
        labels = torch.cat((labels, temp_true_labels))

        # Normal training from here on: ....
        outputs = model(inputs)
        loss, preds = criterion(outputs, labels)
        loss.backward()

        # Optimizer step
        # ...



def _split_data(inputs, labels, p=0.75):
    """Split data for meta update steps and other defenses."""
    batch_size = inputs.shape[0]
    p_actual = int(p * batch_size)

    inputs, temp_targets, = inputs[0:p_actual], inputs[p_actual:]
    labels, temp_true_labels = labels[0:p_actual], labels[p_actual:]
    temp_fake_label = labels.mode(keepdim=True)[0].repeat(batch_size - p_actual)
    return temp_targets, inputs, temp_true_labels, labels, temp_fake_label
