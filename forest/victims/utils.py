"""Utilites related to training models."""


def print_and_save_stats(epoch, stats, current_lr, train_loss, train_acc, predictions, valid_loss,
                         target_acc, target_loss, target_clean_acc, target_clean_loss):
    """Print info into console and into the stats object."""
    stats['train_losses'].append(train_loss)
    stats['train_accs'].append(train_acc)

    if predictions is not None:
        stats['valid_accs'].append(predictions['all']['avg'])
        stats['valid_accs_base'].append(predictions['base']['avg'])
        stats['valid_accs_target'].append(predictions['target']['avg'])
        stats['valid_losses'].append(valid_loss)

        print(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
              f'Training    loss is {stats["train_losses"][-1]:7.4f}, train acc: {stats["train_accs"][-1]:7.2%} | '
              f'Validation   loss is {stats["valid_losses"][-1]:7.4f}, valid acc: {stats["valid_accs"][-1]:7.2%} | ')

        stats['target_accs'].append(target_acc)
        stats['target_losses'].append(target_loss)
        stats['target_accs_clean'].append(target_clean_acc)
        stats['target_losses_clean'].append(target_clean_loss)
        print(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
              f'Target adv. loss is {target_loss:7.4f}, fool  acc: {target_acc:7.2%} | '
              f'Target orig. loss is {target_clean_loss:7.4f}, orig. acc: {target_clean_acc:7.2%} | ')

    else:
        if 'valid_accs' in stats:
            # Repeat previous answers if validation is not recomputed
            stats['valid_accs'].append(stats['valid_accs'][-1])
            stats['valid_accs_base'].append(stats['valid_accs_base'][-1])
            stats['valid_accs_target'].append(stats['valid_accs_target'][-1])
            stats['valid_losses'].append(stats['valid_losses'][-1])
            stats['target_accs'].append(stats['target_accs'][-1])
            stats['target_losses'].append(stats['target_losses'][-1])
            stats['target_accs_clean'].append(stats['target_accs_clean'][-1])
            stats['target_losses_clean'].append(stats['target_losses_clean'][-1])

        print(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
              f'Training    loss is {stats["train_losses"][-1]:7.4f}, train acc: {stats["train_accs"][-1]:7.2%} | ')
