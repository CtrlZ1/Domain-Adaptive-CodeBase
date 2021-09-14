import torch
from typing import Optional, Sequence
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as col
import torch.autograd as autograd

# the root of dataset and the size of images,[source data root, target data root, size of images]
datasetRootAndImageSize=[
    # office-31 a-w 0
    [r"E:\transferlearning\data\office-31\Original_images\amazon",r"E:\transferlearning\data\office-31\Original_images\webcam",224],
    # svhn->mnist 1
    [r'E:\transferlearning\data\SVHN',r'E:\transferlearning\data\MNIST',28],
    #mnist-mnist-m 2
    [r'E:\transferlearning\data\MNIST',r'E:\transferlearning\data\MNIST-M\mnist_m',28],
    #ImageCLEF 2014 3
    [r'E:\transferlearning\data\ImageCLEF 2014\b',r'E:\transferlearning\data\ImageCLEF 2014\c'],
    # usps-mnist 4
    [r'E:\transferlearning\data\usps',r'E:\transferlearning\data\MNIST',28],
    # office_caltech_10 a-w 5
    [r"E:\transferlearning\data\office_caltech_10\caltech",r"E:\transferlearning\data\office_caltech_10\webcam",224],
    # mnist-usps 6
    [r'E:\transferlearning\data\MNIST', r'E:\transferlearning\data\usps',16],
    # SVHN、USPS->MNIST 7
    [[r'E:\transferlearning\data\SVHN',r'E:\transferlearning\data\usps'], r'E:\transferlearning\data\MNIST',28],

]


def model_feature_tSNE(model,sourceDataLoader,targetDataLoader,image_name,device,model_name):
    '''
    :param model: model
    :param sourceDataLoader: source data loader
    :param targetDataLoader: target data loader
    :param image_name: image name
    :param device: cpu or gpu
    :param model_name: the name of model, used for making new folder to save images
    '''
    source_feature = collect_feature(sourceDataLoader, model, device)
    target_feature = collect_feature(targetDataLoader, model, device)
    tSNE_filename = os.path.join('../images/'+model_name, image_name+'.png')
    if not os.path.exists('../images/'+model_name):
        os.makedirs('../images/'+model_name)
    tSNE(source_feature,target_feature,tSNE_filename)
def tSNE(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=2)
    plt.savefig(filename)
def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                                   device: torch.device, max_num_features=None) -> torch.Tensor:
    """
        Fetch data from `data_loader`, and then use `feature_extractor` to collect features

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader.
            feature_extractor (torch.nn.Module): A feature extractor.
            device (torch.device)
            max_num_features (int): The max number of features to return

        Returns:
            Features in shape (min(len(data_loader), max_num_features), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    print("start to extract features...")
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
            images = images.to(device)
            if feature_extractor.backbone:
                fc=feature_extractor.backbone(images)
            else:
                fc=feature_extractor.feature_extractor(images)

            if isinstance(fc,tuple):
                feature = fc[2] # if backbone = AlexNetFc_for_layerWiseAdaptation
            else:
                feature=fc
            if feature_extractor.bottleneck:
                feature = feature_extractor.bottleneck(feature).cpu()

            all_features.append(feature)
            if max_num_features is not None and i >= max_num_features:
                break
    return torch.cat(all_features, dim=0)




def one_hot_label(Labels,n_labels):
    '''
    :param Labels: the label of source or target data
    :param n_labels: the number of kinds of labels
    :return: one hot label
    '''
    ont_hot_l = torch.eye(n_labels)[Labels]

    return ont_hot_l


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad



def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss"""
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0),1,1,1))).to(real_samples.device)
    interpolates=(alpha * real_samples + (1-alpha) * fake_samples).requires_grad_(True)
    D_interprolates=D(interpolates)

    # Get gradient w.r.t interpolates
    gradients=autograd.grad(
        outputs=D_interprolates,
        inputs=interpolates,
        grad_outputs=torch.ones(interpolates.size(0),1).float().cuda(),
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]
    gradients=gradients.view(gradients.size(0),-1)
    gradients_penalty=((gradients.norm(2,dim=1)-1)**2).mean()
    return gradients_penalty