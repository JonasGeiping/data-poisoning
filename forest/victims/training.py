"""Repeatable code parts concerning optimization and training schedules."""

import torch

from collections import defaultdict

from .utils import print_and_save_stats
from .batched_attacks import construct_attack

from ..consts import NON_BLOCKING, BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

def run_step(kettle, poison_delta, epoch, stats, model, defs, optimizer, scheduler, loss_fn, pretraining_phase=False):

    epoch_loss, total_preds, correct_preds = 0, 0, 0

    if pretraining_phase:
        train_loader = kettle.pretrainloader
        valid_loader = kettle.validloader
    else:
        if kettle.args.ablation < 1.0:
            # run ablation on a subset of the training set
            train_loader = kettle.partialloader
        else:
            train_loader = kettle.trainloader
        valid_loader = kettle.validloader

    if 'adversarial' in defs.novel_defense['type']:
        attacker = construct_attack(defs.novel_defense, model, loss_fn, kettle.dm, kettle.ds,
                                    tau=kettle.args.tau, init='randn', optim='signAdam',
                                    num_classes=len(kettle.trainset.classes), setup=kettle.setup)

    # Compute flag to activate defenses:
    # Here we are writing these conditions out explicitely:
    if poison_delta is None:  # this is the case if the training set is clean
        if defs.adaptive_attack:
            activate_defenses = True
        else:
            activate_defenses = False
    else:  # this is a poisoned training set
        if defs.defend_features_only:
            activate_defenses = False
        else:
            activate_defenses = True

    for batch, (inputs, labels, ids) in enumerate(train_loader):
        # Prep Mini-Batch
        optimizer.zero_grad()

        # Transfer to GPU
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)


        # #### Add poison pattern to data #### #
        if poison_delta is not None:
            poison_slices, batch_positions = kettle.lookup_poison_indices(ids)
            if len(batch_positions) > 0:
                inputs[batch_positions] += poison_delta[poison_slices].to(**kettle.setup)


        # Add data augmentation
        if defs.augmentations:  # defs.augmentations is actually a string, but it is False if --noaugment
            inputs = kettle.augment(inputs)


        # #### Run defenses based on modifying input data #### #
        if activate_defenses:
            if defs.mixing_method['type'] != '':
                inputs, extra_labels, mixing_lmb = kettle.mixer(inputs, labels, epoch=epoch)

            # Split Data
            if any([s in defs.novel_defense['type'] for s in ['adversarial', 'combine']]):
                [temp_targets, inputs,
                 temp_true_labels, labels,
                 temp_fake_label] = _split_data(inputs, labels, target_selection=defs.novel_defense['target_selection'])
            # Poison given data ('adversarial patch' for patch attacks lol)
            if 'adversarial' in defs.novel_defense['type']:
                model.eval()
                delta, additional_info = attacker.attack(inputs, labels, temp_targets, temp_true_labels, temp_fake_label,
                                                         steps=defs.novel_defense['steps'])

                # temp targets are modified for trigger attacks:
                # this already happens as a side effect for hidden-trigger, but not for patch
                if 'patch' in defs.novel_defense['type']:
                    temp_targets = temp_targets + additional_info

                inputs = inputs + delta  # Kind of a reparametrization trick


        # Switch into training mode
        list(model.children())[-1].train() if model.frozen else model.train()

        # Change loss function to include corrective terms if mixing with correction
        if (defs.mixing_method['type'] != '' and defs.mixing_method['correction']) and activate_defenses:
            def criterion(outputs, labels):
                return kettle.mixer.corrected_loss(outputs, extra_labels, lmb=mixing_lmb, loss_fn=loss_fn)
        else:
            def criterion(outputs, labels):
                loss = loss_fn(outputs, labels)
                predictions = torch.argmax(outputs.data, dim=1)
                correct_preds = (predictions == labels).sum().item()
                return loss, correct_preds


        # #### Run defenses modifying the loss function #### #
        if activate_defenses:
            # Compute loss
            if 'recombine' in defs.novel_defense['type']:
                # Recombine poisoned inputs and targets into a single batch
                inputs = torch.cat((inputs, temp_targets))
                labels = torch.cat((labels, temp_true_labels))

        # Do normal model updates, possibly on modified inputs
        outputs = model(inputs)
        loss, preds = criterion(outputs, labels)
        correct_preds += preds

        total_preds += labels.shape[0]
        differentiable_params = [p for p in model.parameters() if p.requires_grad]

        loss.backward()
        epoch_loss += loss.item()

        if activate_defenses:
            with torch.no_grad():
                # Enforce batch-wise privacy if necessary
                # This is a defense discussed in Hong et al., 2020
                # We enforce privacy on mini batches instead of instances to cope with effects on batch normalization
                # This is reasonble as Hong et al. discuss that defense against poisoning mostly arises from the addition
                # of noise to the gradient signal
                if defs.privacy['clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(differentiable_params, defs.privacy['clip'])
                if defs.privacy['noise'] is not None:
                    loc = torch.as_tensor(0.0, device=kettle.setup['device'])
                    clip_factor = defs.privacy['clip'] if defs.privacy['clip'] is not None else 1.0
                    scale = torch.as_tensor(clip_factor * defs.privacy['noise'], device=kettle.setup['device'])
                    if defs.privacy['distribution'] == 'gaussian':
                        generator = torch.distributions.normal.Normal(loc=loc, scale=scale)
                    elif defs.privacy['distribution'] == 'laplacian':
                        generator = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
                    else:
                        raise ValueError(f'Invalid distribution {defs.privacy["distribution"]} given.')
                    for param in differentiable_params:
                        param.grad += generator.sample(param.shape)


        optimizer.step()

        if defs.scheduler == 'cyclic':
            scheduler.step()
        if kettle.args.dryrun:
            break
    if defs.scheduler == 'linear':
        scheduler.step()

    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        predictions, valid_loss = run_validation(model, loss_fn, valid_loader,
                                                 kettle.poison_setup['intended_class'],
                                                 kettle.poison_setup['target_class'],
                                                 kettle.setup, kettle.args.dryrun)
        target_acc, target_loss, target_clean_acc, target_clean_loss = check_targets(
            model, loss_fn, kettle.targetset, kettle.poison_setup['intended_class'],
            kettle.poison_setup['target_class'],
            kettle.setup)
    else:
        predictions, valid_loss = None, None
        target_acc, target_loss, target_clean_acc, target_clean_loss = [None] * 4

    current_lr = optimizer.param_groups[0]['lr']
    print_and_save_stats(epoch, stats, current_lr, epoch_loss / (batch + 1), correct_preds / total_preds,
                         predictions, valid_loss,
                         target_acc, target_loss, target_clean_acc, target_clean_loss)


def run_validation(model, criterion, dataloader, intended_class, original_class, setup, dryrun=False):
    """Get accuracy of model relative to dataloader.

    Hint: The validation numbers in "base" and "target" explicitely reference the first label in intended_class and
    the first label in original_class.
    """
    model.eval()
    intended_class = torch.tensor(intended_class).to(device=setup['device'], dtype=torch.long)
    original_class = torch.tensor(original_class).to(device=setup['device'], dtype=torch.long)
    predictions = defaultdict(lambda: dict(correct=0, total=0))

    loss = 0

    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, targets).item()
            predictions['all']['total'] += targets.shape[0]
            predictions['all']['correct'] += (predicted == targets).sum().item()

            predictions['base']['total'] += (targets == intended_class[0]).sum().item()
            predictions['base']['correct'] += (predicted == targets)[targets == intended_class[0]].sum().item()

            predictions['target']['total'] += (targets == original_class).sum().item()
            predictions['target']['correct'] += (predicted == targets)[targets == original_class].sum().item()

            if dryrun:
                break

    for key in predictions.keys():
        if predictions[key]['total'] > 0:
            predictions[key]['avg'] = predictions[key]['correct'] / predictions[key]['total']
        else:
            predictions[key]['avg'] = float('nan')

    loss_avg = loss / (i + 1)
    return predictions, loss_avg

def check_targets(model, criterion, targetset, intended_class, original_class, setup):
    """Get accuracy and loss for all targets on their intended class."""
    model.eval()
    if len(targetset) > 0:

        target_images = torch.stack([data[0] for data in targetset]).to(**setup)
        intended_labels = torch.tensor(intended_class).to(device=setup['device'], dtype=torch.long)
        original_labels = torch.stack([torch.as_tensor(data[1], device=setup['device'], dtype=torch.long) for data in targetset])
        with torch.no_grad():
            outputs = model(target_images)
            predictions = torch.argmax(outputs, dim=1)

            loss_intended = criterion(outputs, intended_labels)
            accuracy_intended = (predictions == intended_labels).sum().float() / predictions.size(0)
            loss_clean = criterion(outputs, original_labels)
            predictions_clean = torch.argmax(outputs, dim=1)
            accuracy_clean = (predictions == original_labels).sum().float() / predictions.size(0)

            # print(f'Raw softmax output is {torch.softmax(outputs, dim=1)}, intended: {intended_class}')

        return accuracy_intended.item(), loss_intended.item(), accuracy_clean.item(), loss_clean.item()
    else:
        return 0, 0, 0, 0


def _split_data(inputs, labels, target_selection='sep-half'):
    """Split data for meta update steps and other defenses."""
    batch_size = inputs.shape[0]
    #  shuffle/sep-half/sep-1/sep-10
    if target_selection == 'shuffle':
        shuffle = torch.randperm(batch_size, device=inputs.device)
        temp_targets = inputs[shuffle].detach().clone()
        temp_true_labels = labels[shuffle].clone()
        temp_fake_label = labels
    elif target_selection == 'michael':  # experimental option
        temp_targets = inputs
        temp_true_labels = labels
        bins = torch.bincount(labels, minlength=10)  # this is only valid for CIFAR-10
        least_appearing_label = bins.argmin()
        temp_fake_label = least_appearing_label.repeat(batch_size)
    elif target_selection == 'sep-half':
        temp_targets, inputs = inputs[:batch_size // 2], inputs[batch_size // 2:]
        temp_true_labels, labels = labels[:batch_size // 2], labels[batch_size // 2:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(batch_size // 2)
    elif target_selection == 'sep-1':
        temp_targets, inputs = inputs[0:1], inputs[1:]
        temp_true_labels, labels = labels[0:1], labels[1:]
        temp_fake_label = labels.mode(keepdim=True)[0]
    elif target_selection == 'sep-10':
        temp_targets, inputs = inputs[0:10], inputs[10:]
        temp_true_labels, labels = labels[0:10], labels[10:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(10)
    elif 'sep-p' in target_selection:
        p = int(target_selection.split('sep-p')[1])
        p_actual = int(p * batch_size / 128)
        if p_actual > batch_size or p_actual < 1:
            raise ValueError(f'Invalid sep-p option given with p={p}. Should be p in [1, 128], '
                             f'which will be scaled to the current batch size.')
        inputs, temp_targets, = inputs[0:p_actual], inputs[p_actual:]
        labels, temp_true_labels = labels[0:p_actual], labels[p_actual:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(batch_size - p_actual)

    else:
        raise ValueError(f'Invalid selection strategy {target_selection}.')
    return temp_targets, inputs, temp_true_labels, labels, temp_fake_label


def get_optimizers(model, args, defs):
    """Construct optimizer as given in defs."""
    # For transfer learning, we restrict the parameters to be optimized.
    # This filter should only trigger if self.args.scenario == 'transfer'
    optimized_parameters = filter(lambda p: p.requires_grad, model.parameters())

    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(optimized_parameters, lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'SGD-basic':
        optimizer = torch.optim.SGD(optimized_parameters, lr=defs.lr, momentum=0.0,
                                    weight_decay=defs.weight_decay, nesterov=False)
    elif defs.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(optimized_parameters, lr=defs.lr, weight_decay=defs.weight_decay)
    elif defs.optimizer == 'Adam':
        optimizer = torch.optim.Adam(optimized_parameters, lr=defs.lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'cyclic':
        effective_batches = (50_000 // defs.batch_size) * defs.epochs
        # todo: unmake this magic number that only works on cifar.
        # but when are we going to use this scheduler anyway?
        print(f'Optimization will run over {effective_batches} effective batches in a 1-cycle policy.')
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=defs.lr / 100, max_lr=defs.lr,
                                                      step_size_up=effective_batches // 2,
                                                      cycle_momentum=True if defs.optimizer in ['SGD'] else False)
    elif defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[defs.epochs // 2.667, defs.epochs // 1.6,
                                                                     defs.epochs // 1.142], gamma=0.1)
    elif defs.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[10_000, 15_000, 25_000], gamma=1)

        # Example: epochs=160 leads to drops at 60, 100, 140.
    return optimizer, scheduler
