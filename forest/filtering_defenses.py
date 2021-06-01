"""Implement three filter-based defenses."""
import torch
import numpy as np

def get_defense(args):
    if args.filter_defense.lower() == 'spectral_signatures':
        return _SpectralSignaturesDefense
    elif args.filter_defense.lower() == 'deepknn':
        return _DeepKNN
    elif args.filter_defense.lower() == 'activation_clustering':
        return _ActivationClustering


def _get_poisoned_features(kettle, victim, poison_delta, dryrun=False):
    class_indices = [[] for _ in range(len(kettle.trainset.classes))]
    feats = []
    layer_cake = list(victim.model.children())
    feature_extractor = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
    with torch.no_grad():
        for i, (img, target, idx) in enumerate(kettle.trainset):
            lookup = kettle.poison_lookup.get(idx)
            if lookup is not None:
                img += poison_delta[lookup, :, :, :]
            img = img.unsqueeze(0).to(**kettle.setup)
            feats.append(feature_extractor(img))
            class_indices[target].append(i)
            if dryrun and i == 3:  # use a few values to populate these adjancency matrices
                break
    return feats, class_indices

def _DeepKNN(kettle, victim, poison_delta, overestimation_factor=2.0):
    """deepKNN as in Peri et al. "Deep k-NN Defense against Clean-label Data Poisoning Attacks".

    An overestimation factor of 2 is motivated as necessary in that work.
    """
    clean_indices = []
    num_poisons_expected = int(overestimation_factor * kettle.args.budget * len(kettle.trainset)) if not kettle.args.dryrun else 0
    feats, _ = _get_poisoned_features(kettle, victim, poison_delta, dryrun=kettle.args.dryrun)

    feats = torch.stack(feats, dim=0)
    dist_matrix = torch.zeros((len(feats), len(feats)))
    for i in range(dist_matrix.shape[0]):
        temp_matrix = torch.stack([feats[i] for _ in range(dist_matrix.shape[1])], dim=0)
        dist_matrix[i, :] = torch.norm((temp_matrix - feats).squeeze(1), dim=1)
    for i in range(dist_matrix.shape[0]):
        vec = dist_matrix[i, :]
        point_label, _ = kettle.trainset.get_target(i)
        _, nearest_indices = vec.topk(num_poisons_expected + 1, largest=False)
        count = 0
        for j in range(1, num_poisons_expected + 1):
            neighbor_label, _ = kettle.trainset.get_target(nearest_indices[j])
            if neighbor_label == point_label:
                count += 1
            else:
                count -= 1
        if count >= 0:
            clean_indices.append(i)
    return clean_indices


def _SpectralSignaturesDefense(kettle, victim, poison_delta, overestimation_factor=1.5):
    """Implement the spectral signautres defense proposed by Tran et al. in "Spectral Signatures in Backdoor Attacks".

    https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf
    The overestimation factor of 1.5 is proposed in the paper.
    """
    clean_indices = []
    num_poisons_expected = kettle.args.budget * len(kettle.trainset)
    feats, class_indices = _get_poisoned_features(kettle, victim, poison_delta, dryrun=kettle.args.dryrun)

    for i in range(len(class_indices)):
        if len(class_indices[i]) > 1:
            temp_feats = []
            for temp_index in class_indices[i]:
                temp_feats.append(feats[temp_index])
            temp_feats = torch.cat(temp_feats)
            mean_feat = torch.mean(temp_feats, dim=0)
            temp_feats = temp_feats - mean_feat
            _, _, V = torch.svd(temp_feats, compute_uv=True, some=False)
            vec = V[:, 0]  # the top right singular vector is the first column of V
            vals = []
            for j in range(temp_feats.shape[0]):
                vals.append(torch.dot(temp_feats[j], vec).pow(2))

            k = min(int(overestimation_factor * num_poisons_expected), len(vals) - 1)
            _, indices = torch.topk(torch.tensor(vals), k)
            bad_indices = []
            for temp_index in indices:
                bad_indices.append(class_indices[i][temp_index])
            clean = list(set(class_indices[i]) - set(bad_indices))
            clean_indices = clean_indices + clean
    return clean_indices

def _ActivationClustering(kettle, victim, poison_delta, clusters=2):
    """Implement Chen et al. "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering"."""
    # lazy sklearn import:
    from sklearn.cluster import KMeans

    clean_indices = []
    feats, class_indices = _get_poisoned_features(kettle, victim, poison_delta, dryrun=kettle.args.dryrun)

    for i in range(len(class_indices)):
        if len(class_indices[i]) > 1:
            temp_feats = np.array([feats[temp_idx].squeeze(0).cpu().numpy() for temp_idx in class_indices[i]])
            kmeans = KMeans(n_clusters=clusters).fit(temp_feats)
            if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
                clean_label = 1
            else:
                clean_label = 0
            clean = []
            for (bool, idx) in zip((kmeans.labels_ == clean_label).tolist(), list(range(len(kmeans.labels_)))):
                if bool:
                    clean.append(class_indices[i][idx])
            clean_indices = clean_indices + clean
    return clean_indices
