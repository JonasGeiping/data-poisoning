import torch
from forest.victims.models import ResNet, resnet_picker

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def generate_plot_centroid(feat_path,model_path,target_class,base_class, poison_ids, title, device):

    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if i in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append(str(target_class))
        else:
            tags.append(str(base_class))

    tags = np.array(tags)  
    print(np.sum(tags == str(target_class)), 
          np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str(base_class)]
    targfeats = left_ops[tags == str(target_class)]
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    distcent = targcent - basecent
    distcent /= np.linalg.norm(distcent)
    distcent *= -1

    basefeats_ = basefeats.dot(distcent)
    basefeats_ = np.outer(basefeats_, distcent)
    basefeats_ = basefeats - basefeats_

    targfeats_ = targfeats.dot(distcent)
    targfeats_ = np.outer(targfeats_, distcent)
    targfeats_ = targfeats - targfeats_

    a = np.concatenate([basefeats_, targfeats_])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([basefeats_, targfeats_]))
    orthcent = pca.components_[0]
#     print(pca.explained_variance_ratio_)

    baseproj = np.stack([basefeats.dot(distcent), basefeats.dot(orthcent)], axis=1)
    targproj = np.stack([targfeats.dot(distcent), targfeats.dot(orthcent)], axis=1)


    plt.plot(*baseproj.T, '.g', alpha=.03, markeredgewidth=0)
    plt.plot(*targproj.T, '.b', alpha=.03, markeredgewidth=0)

    poisonfeats = left_ops[tags == str('poison')]
    poisoncent = np.mean(poisonfeats, axis=0)
#     print("Printing below distance between centroids")
#     print(np.linalg.norm(basecent-targcent),np.linalg.norm(basecent-poisoncent), np.linalg.norm(poisoncent-targcent))
    poisonproj = np.stack([poisonfeats.dot(distcent), poisonfeats.dot(orthcent)], axis=1)
    plt.plot(*poisonproj.T, 'or', alpha=1, markeredgewidth=0, markersize=7, label='poisons')

    targetfeats = left_ops[tags == str('target')]
    targetproj = np.stack([targetfeats.dot(distcent), targetfeats.dot(orthcent)], axis=1)
    plt.plot(*targetproj.T, '^b', markersize=12, markeredgewidth=0, label='target')

#     plt.xlim(-6, 6)
    # plt.ylim(-4, 52)
    plt.xlabel('distance along centroids')
    plt.ylabel('dist along orthonormal')
    plt.legend(frameon=False, loc='lower left')
    plt.title(title)
#     plt.text(-5, 5, 'target class')
#     plt.text(2,5, 'base class')
    plt.show()
    
def generate_plot_pca(feat_path,model_path, target_class,base_class, poison_ids, title, device):

    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if i in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append(str(target_class))
        else:
            tags.append(str(base_class))

    tags = np.array(tags)  
    print(np.sum(tags == str(target_class)), 
          np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str(base_class)]
    targfeats = left_ops[tags == str(target_class)]
    
    a = np.concatenate([basefeats, targfeats])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(np.concatenate([basefeats, targfeats]))
    distcent = pca.components_[0]
    orthcent = pca.components_[1]
#     print(pca.explained_variance_ratio_)

    baseproj = np.stack([basefeats.dot(distcent), basefeats.dot(orthcent)], axis=1)
    targproj = np.stack([targfeats.dot(distcent), targfeats.dot(orthcent)], axis=1)


    plt.plot(*baseproj.T, '.g', alpha=.03, markeredgewidth=0)
    plt.plot(*targproj.T, '.b', alpha=.03, markeredgewidth=0)

    poisonfeats = left_ops[tags == str('poison')]
    poisoncent = np.mean(poisonfeats, axis=0)
#     print("Printing below distance between centroids")
#     print(np.linalg.norm(basecent-targcent),np.linalg.norm(basecent-poisoncent), np.linalg.norm(poisoncent-targcent))
    poisonproj = np.stack([poisonfeats.dot(distcent), poisonfeats.dot(orthcent)], axis=1)
    plt.plot(*poisonproj.T, 'or', alpha=1, markeredgewidth=0, markersize=7, label='poisons')

    targetfeats = left_ops[tags == str('target')]
    targetproj = np.stack([targetfeats.dot(distcent), targetfeats.dot(orthcent)], axis=1)
    plt.plot(*targetproj.T, '^b', markersize=12, markeredgewidth=0, label='target')

#     plt.xlim(-6, 6)
    # plt.ylim(-4, 52)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(frameon=False, loc='lower left')
    plt.title(title)
    plt.text(-5, 5, 'target class')
    plt.text(2,5, 'base class')
    plt.show()
    

    
def bypass_last_layer(model):
    """Hacky way of separating features and classification head for many models.
    Patch this function if problems appear.
    """
    layer_cake = list(model.children())
    last_layer = layer_cake[-1]
    headless_model = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())  # this works most of the time all of the time :<
    return headless_model, last_layer
    
def generate_plot_centroid_3d_labels(feat_path, model_path, target_class,base_class, poison_ids, title, device):

    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    model = resnet_picker('ResNet18', 'CIFAR10')
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)
    headless_model, last_layer  = bypass_last_layer(model)
    last_layer_weights = last_layer.weight.detach().cpu().numpy()
    logit_matrix = np.matmul(ops_all, last_layer_weights.T)
    classif = np.argmax(logit_matrix, axis=1)
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    classif = classif[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if i in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append(str(target_class))
        else:
            tags.append(str(base_class))

    tags = np.array(tags)  
    print(np.sum(tags == str(target_class)), 
          np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str(base_class)]
    targfeats = left_ops[tags == str(target_class)]
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    distcent = targcent - basecent
    distcent /= np.linalg.norm(distcent)
    distcent *= -1

    basefeats_ = basefeats.dot(distcent)
    basefeats_ = np.outer(basefeats_, distcent)
    basefeats_ = basefeats - basefeats_

    targfeats_ = targfeats.dot(distcent)
    targfeats_ = np.outer(targfeats_, distcent)
    targfeats_ = targfeats - targfeats_

    a = np.concatenate([basefeats_, targfeats_])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([basefeats_, targfeats_]))
    orthcent = pca.components_[0]
#     print(pca.explained_variance_ratio_)


    allproj = np.stack([left_ops.dot(distcent), left_ops.dot(orthcent)], axis=1)
    
    data_3ax = np.column_stack((allproj, classif))
    from mpl_toolkits.mplot3d import Axes3D
#     fig = plt.figure()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    uniq_tags = np.unique(tags)
    colors = ['blue', 'green','red', 'black']
    markers = [',',',','o','^']
    alphas = [0.03,0.03,0.2,1]
    sizes = [5,5,10,50]
    col_scheme = dict(zip(uniq_tags, zip(colors, markers,alphas,sizes)))
#     print(col_scheme)
    for i in range(len(tags)):
        tag = tags[i]
        c,m,al,ms = col_scheme[tag]
        ax.scatter(data_3ax[i,0], data_3ax[i,1], data_3ax[i,2], c=c, marker=m,
                  alpha=al,s=ms)
        
    ax.set_xlabel('distance along centroids')
    ax.set_ylabel('dist along orthonormal')
    ax.set_zlabel('Predicted class')
#     handles, labels = ax.legend_elements(prop="colors", alpha=0.6)
#     legend2 = ax.legend(handles, labels, loc="upper right", title="colors")

#     plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.show()
    
def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p    
    
def generate_plot_lda_patch(feat_path,model_path, target_class,base_class, poison_ids, title, device):
    
    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if i in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append(str(target_class))
        else:
            tags.append(str(base_class))

    tags = np.array(tags)  
#     print(np.sum(tags == str(target_class)), 
#           np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))

    basefeats = left_ops[tags == str(base_class)]
    targfeats = left_ops[tags == str(target_class)]
    poisonfeats = left_ops[tags == 'poison']
    targetfeats = left_ops[tags == 'target']
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    poisoncent = np.mean(poisonfeats, axis=0)
#     print("Printing below distance between centroids")
#     print(np.linalg.norm(basecent-targcent),np.linalg.norm(basecent-poisoncent), np.linalg.norm(poisoncent-targcent))
    
    ol_tags = np.concatenate([tags[tags == str(base_class)], tags[tags == str(target_class)], tags[tags == str('poison')]])
    ol_feats = np.concatenate([basefeats, targfeats, poisonfeats])
#     print(ol_tags.shape, ol_feats.shape)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(ol_feats, ol_tags).transform(ol_feats)
    
    colors = ['green', 'blue','red']
    target_names = [str(base_class), str(target_class), 'poison']
    alphas = [0.05,0.05, 0.05]
    sizes = [5,5,5]
    plt.figure()
    for color, i, target_name,al,si in zip(colors, target_names, target_names,alphas,sizes):
        plt.scatter(X_r2[ol_tags == i, 0], X_r2[ol_tags == i, 1], alpha= al, color=color,
                    label=i,s=si)
    target_proj = lda.fit(ol_feats, ol_tags).transform(targetfeats)
    plt.scatter(target_proj[:,0], target_proj[:,1], alpha=0.4, color='black', marker='^',
                    label='target',s=5)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.show()
    
def genplot_centroid_prob_2d_patch(feat_path, model_path, target_class,base_class, poison_ids, title, device):

    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    model = resnet_picker('ResNet18', 'CIFAR10')
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)
    headless_model, last_layer  = bypass_last_layer(model)
    last_layer_weights = last_layer.weight.detach().cpu().numpy()
    logit_matrix = np.matmul(ops_all, last_layer_weights.T)
    softmax_scores = softmax(logit_matrix, theta = 1, axis = 1)
    target_lab_confidence = softmax_scores[:,base_class]
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    target_lab_confidence = target_lab_confidence[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if i in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append('target_class')
        else:
            tags.append('poison_class')

    tags = np.array(tags)  
#     print(np.sum(tags == str(target_class)), 
#           np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str('poison_class')]
    targfeats = left_ops[tags == str('target_class')]
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    distcent = targcent - basecent
    distcent /= np.linalg.norm(distcent)
    distcent *= -1

    basefeats_ = basefeats.dot(distcent)
    basefeats_ = np.outer(basefeats_, distcent)
    basefeats_ = basefeats - basefeats_

    targfeats_ = targfeats.dot(distcent)
    targfeats_ = np.outer(targfeats_, distcent)
    targfeats_ = targfeats - targfeats_

    a = np.concatenate([basefeats_, targfeats_])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([basefeats_, targfeats_]))
    orthcent = pca.components_[0]


    allproj = np.stack([left_ops.dot(distcent), left_ops.dot(orthcent)], axis=1)
    
    data_3ax = np.column_stack((allproj, target_lab_confidence*30+10))
    plt.figure()
    uniq_tags = np.array(['target_class','poison_class','poison','target'])#np.unique(tags)
    colors = ['royalblue', 'green','red', 'black']
    markers = [',',',','o','^']
    alphas = [0.05,0.05,0.05,0.4]
    sizes = [5,5,5,5]
    for color, i, target_name,al,m in zip(colors, uniq_tags, uniq_tags,alphas,markers):
        s = data_3ax[tags == i, 2]/3
        if i == 'target':
            s = s*2
        else:
            s = 10
        plt.scatter(data_3ax[tags == i, 0], data_3ax[tags == i, 1], alpha= al, color=color,
                    label=i,s= s,marker=m)
    plt.xlabel('distance along centroids',fontsize=15,fontweight='medium',fontvariant='small-caps')
    plt.ylabel('distance orthonormal',fontsize=15,fontweight='medium',fontvariant='small-caps')
    figname = title.replace(" ", "_")+ "_2d.pdf"
    plt.savefig(os.path.join('./plots/2d', figname), bbox_inches='tight')
    #     plt.legend(loc='best', shadow=False, scatterpoints=1)
#     plt.title(title)
    plt.show()
    
def genplot_centroid_3d_patch(feat_path, model_path, target_class,base_class, poison_ids, title, device):

    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    model = resnet_picker('ResNet18', 'CIFAR10')
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)
    headless_model, last_layer  = bypass_last_layer(model)
    last_layer_weights = last_layer.weight.detach().cpu().numpy()
    logit_matrix = np.matmul(ops_all, last_layer_weights.T)
    softmax_scores = softmax(logit_matrix, theta = 1, axis = 1)
    target_lab_confidence = softmax_scores[:,base_class]
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    target_lab_confidence = target_lab_confidence[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if i in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append(str('target_class'))
        else:
            tags.append(str('poison_class'))

    tags = np.array(tags)  
#     print(np.sum(tags == str(target_class)), 
#           np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str('poison_class')]
    targfeats = left_ops[tags == str('target_class')]
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    distcent = targcent - basecent
    distcent /= np.linalg.norm(distcent)
    distcent *= -1

    basefeats_ = basefeats.dot(distcent)
    basefeats_ = np.outer(basefeats_, distcent)
    basefeats_ = basefeats - basefeats_

    targfeats_ = targfeats.dot(distcent)
    targfeats_ = np.outer(targfeats_, distcent)
    targfeats_ = targfeats - targfeats_

    a = np.concatenate([basefeats_, targfeats_])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([basefeats_, targfeats_]))
    orthcent = pca.components_[0]

    allproj = np.stack([left_ops.dot(distcent), left_ops.dot(orthcent)], axis=1)
    
    data_3ax = np.column_stack((allproj, target_lab_confidence))
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    uniq_tags = np.array(['target_class','poison_class','poison','target'])
    colors = ['royalblue', 'green','red', 'dimgray']
    markers = [',',',','o','^']
    alphas = [0.03,0.03,0.05,0.2]
    sizes = [5,5,5,5]
    for color, i, target_name,al,si,m in zip(colors, uniq_tags, uniq_tags,alphas,sizes,markers):
        
        ax.scatter(data_3ax[tags == i, 0], data_3ax[tags == i, 1], 
                   data_3ax[tags == i, 2],alpha= al, color=color,
                     marker=m, label=i,s=si)
        
    ax.set_xlabel('distance along centroids',fontsize=15,fontweight='bold',fontvariant='small-caps')
    ax.set_ylabel('distance orthonormal',fontsize=15,fontweight='bold',fontvariant='small-caps')
    ax.set_zlabel('poison class probability',fontsize=15,fontweight='bold',fontvariant='small-caps')
    figname = title.replace(" ", "_")+ "_3d.pdf"
    plt.savefig(os.path.join('./plots', figname), bbox_inches='tight')
#     plt.title(title)
    plt.show()
    
def generate_plot_lda(feat_path,model_path, target_class,base_class, poison_ids, title, device):
    
    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if i in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append(str(target_class))
        else:
            tags.append(str(base_class))

    tags = np.array(tags)  

    basefeats = left_ops[tags == str(base_class)]
    targfeats = left_ops[tags == str(target_class)]
    poisonfeats = left_ops[tags == 'poison']
    targetfeats = left_ops[tags == 'target']
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    poisoncent = np.mean(poisonfeats, axis=0)
   
    ol_tags = np.concatenate([tags[tags == str(base_class)], tags[tags == str(target_class)], tags[tags == str('poison')]])
    ol_feats = np.concatenate([basefeats, targfeats, poisonfeats])
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(ol_feats, ol_tags).transform(ol_feats)
    
    colors = ['green', 'royalblue','red']
    target_names = [str(base_class), str(target_class), 'poison']
    alphas = [0.02,0.02,0.5]
    sizes = [5,5,10]
    plt.figure()
    for color, i, target_name,al,si in zip(colors, target_names, target_names,alphas,sizes):
        plt.scatter(X_r2[ol_tags == i, 0], X_r2[ol_tags == i, 1], alpha= al, color=color,
                    label=i,s=si)
    target_proj = lda.fit(ol_feats, ol_tags).transform(targetfeats)
    plt.scatter(target_proj[0][0], target_proj[0][1], alpha=1, color='dimgray', marker='^',label='target',s=50,edgecolors = 'black', linewidth=3)
    plt.xlabel('LD1',fontsize=15,fontweight='medium',fontvariant='small-caps')
    plt.ylabel('LD2',fontsize=15,fontweight='medium',fontvariant='small-caps')
    figname = title.replace(" ", "_")+ "_3d_lda.pdf"
    plt.savefig(os.path.join('./plots', figname), bbox_inches='tight')
    #     plt.legend(loc='best', shadow=False, scatterpoints=1)
#     plt.title(title)
    plt.show()

def genplot_centroid_prob_3d(feat_path, model_path, target_class,base_class, poison_ids, title, device):

    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    model = resnet_picker('ResNet18', 'CIFAR10')
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)
    headless_model, last_layer  = bypass_last_layer(model)
    last_layer_weights = last_layer.weight.detach().cpu().numpy()
    logit_matrix = np.matmul(ops_all, last_layer_weights.T)
    softmax_scores = softmax(logit_matrix, theta = 1, axis = 1)
    target_lab_confidence = softmax_scores[:,base_class]
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    target_lab_confidence = target_lab_confidence[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if i in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append('target_class')
        else:
            tags.append('poison_class')

    tags = np.array(tags)  
#     print(np.sum(tags == str(target_class)), 
#           np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str('poison_class')]
    targfeats = left_ops[tags == str('target_class')]
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    distcent = targcent - basecent
    distcent /= np.linalg.norm(distcent)
    distcent *= -1

    basefeats_ = basefeats.dot(distcent)
    basefeats_ = np.outer(basefeats_, distcent)
    basefeats_ = basefeats - basefeats_

    targfeats_ = targfeats.dot(distcent)
    targfeats_ = np.outer(targfeats_, distcent)
    targfeats_ = targfeats - targfeats_

    a = np.concatenate([basefeats_, targfeats_])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([basefeats_, targfeats_]))
    orthcent = pca.components_[0]


    allproj = np.stack([left_ops.dot(distcent), left_ops.dot(orthcent)], axis=1)
    
    data_3ax = np.column_stack((allproj, target_lab_confidence))
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    uniq_tags = np.array(['target_class','poison_class','poison','target'])#np.unique(tags)
    colors = ['royalblue', 'green','red', 'dimgray']
    markers = [',',',','o','^']
    alphas = [0.03,0.03,0.3,1]
    sizes = [5,5,10,200]
    for color, i, target_name,al,si,m in zip(colors, uniq_tags, uniq_tags,alphas,sizes,markers):
        if i == 'target':
            edgecolors = 'black'
            linewidth=3
        else:
            edgecolors= None
            linewidth =0
        ax.scatter(data_3ax[tags == i, 0], data_3ax[tags == i, 1], 
                   data_3ax[tags == i, 2],alpha= al, color=color,
                     marker=m, label=i,s=si,
                   edgecolors=edgecolors,linewidth=linewidth)
        
    ax.set_xlabel('distance along centroids',fontsize=15,fontweight='bold',fontvariant='small-caps')
    ax.set_ylabel('distance orthonormal',fontsize=15,fontweight='bold',fontvariant='small-caps')
    ax.set_zlabel('poison class probability',fontsize=15,fontweight='bold',fontvariant='small-caps')
    figname = title.replace(" ", "_")+ "_3d.pdf"
    plt.savefig(os.path.join('./plots/3d', figname), bbox_inches='tight')
#     plt.title(title)
    plt.show()
    
    
def genplot_centroid_prob_2d(feat_path, model_path, target_class,base_class, poison_ids, title, device):

    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    model = resnet_picker('ResNet18', 'CIFAR10')
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)
    headless_model, last_layer  = bypass_last_layer(model)
    last_layer_weights = last_layer.weight.detach().cpu().numpy()
    logit_matrix = np.matmul(ops_all, last_layer_weights.T)
    softmax_scores = softmax(logit_matrix, theta = 1, axis = 1)
    target_lab_confidence = softmax_scores[:,base_class]
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    target_lab_confidence = target_lab_confidence[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if i in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append('target_class')
        else:
            tags.append('poison_class')

    tags = np.array(tags)  
#     print(np.sum(tags == str(target_class)), 
#           np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str('poison_class')]
    targfeats = left_ops[tags == str('target_class')]
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    distcent = targcent - basecent
    distcent /= np.linalg.norm(distcent)
    distcent *= -1

    basefeats_ = basefeats.dot(distcent)
    basefeats_ = np.outer(basefeats_, distcent)
    basefeats_ = basefeats - basefeats_

    targfeats_ = targfeats.dot(distcent)
    targfeats_ = np.outer(targfeats_, distcent)
    targfeats_ = targfeats - targfeats_

    a = np.concatenate([basefeats_, targfeats_])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([basefeats_, targfeats_]))
    orthcent = pca.components_[0]


    allproj = np.stack([left_ops.dot(distcent), left_ops.dot(orthcent)], axis=1)
    
    data_3ax = np.column_stack((allproj, target_lab_confidence*30+10))
    plt.figure()
    uniq_tags = np.array(['target_class','poison_class','poison','target'])#np.unique(tags)
    colors = ['royalblue', 'green','red', 'black']
    markers = [',',',','o','^']
    alphas = [0.01,0.01,0.08,1]
    sizes = [5,5,10,200]
    for color, i, target_name,al,m in zip(colors, uniq_tags, uniq_tags,alphas,markers):
        s = data_3ax[tags == i, 2]
        if i == 'target':
            s = s*10
        else:
            s = 10
        plt.scatter(data_3ax[tags == i, 0], data_3ax[tags == i, 1], alpha= al, color=color,
                    label=i,s= s,marker=m)
    plt.xlabel('distance along centroids',fontsize=15,fontweight='medium',fontvariant='small-caps')
    plt.ylabel('distance orthonormal',fontsize=15,fontweight='medium',fontvariant='small-caps')
    figname = title.replace(" ", "_")+ "_2d.pdf"
    plt.savefig(os.path.join('./plots/2d', figname), bbox_inches='tight')
    #     plt.legend(loc='best', shadow=False, scatterpoints=1)
#     plt.title(title)
    plt.show()

def generate_plots(main_path,model_name, plot_function, target_class, base_class,poison_ids,device):
    os.makedirs('./plots/2d', exist_ok=True)
    os.makedirs('./plots/3d', exist_ok=True)
    feat_path = os.path.join(main_path+'_undefended', 'clean_model','clean_features.pickle')
    model_path = os.path.join(main_path+'_undefended', 'clean_model','clean.pth')
    plot_function(feat_path,model_path, target_class,base_class, poison_ids,
                        model_name + " "+ "undefended", device)
    
    feat_path = os.path.join(main_path+'_undefended', 'defended_model','def_features.pickle')
    model_path = os.path.join(main_path+'_undefended', 'defended_model','def.pth')
    plot_function(feat_path,model_path, target_class,base_class, poison_ids,
                           model_name + " "+ "attacked_undefended",device)
    
    feat_path = os.path.join(main_path+'_defended', 'clean_model','clean_features.pickle')
    model_path = os.path.join(main_path+'_defended', 'clean_model','clean.pth')
    plot_function(feat_path,model_path, target_class,base_class, poison_ids,
                           model_name + " "+ "defended",device)

    feat_path = os.path.join(main_path+'_defended', 'defended_model','def_features.pickle')
    model_path = os.path.join(main_path+'_defended', 'defended_model','def.pth')
    plot_function(feat_path,model_path, target_class,base_class, poison_ids,
                           model_name + " "+  "attacked_defended",device)
    
    feat_path = os.path.join(main_path+'_defended_featuresonly', 'defended_model','def_features.pickle')
    model_path = os.path.join(main_path+'_defended_featuresonly', 'defended_model','def.pth')
    try:
        plot_function(feat_path,model_path, target_class,base_class, poison_ids,
               model_name + " "+ "defended_base",device)
    except:
        print('Defended base model is not available')



    
