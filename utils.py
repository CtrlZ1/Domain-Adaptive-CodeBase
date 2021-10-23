import torch
from typing import Optional, Sequence
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import os
from torch.autograd import Function
import matplotlib.pyplot as plt
import matplotlib.colors as col
import torch.autograd as autograd
from typing import Optional, Any, Tuple

# the root of dataset and the size of images,[source data root, target data root, size of images]
datasetRootAndImageSize=[
    # office-31 a-w 0
    ["Office31-amazon","Office31-webcam",224],
    # svhn->mnist 1
    ['SVHN','MNIST',28],
    #mnist-mnist-m 2
    ['MNIST','MNIST-M',28],
    #ImageCLEF 2014 3
    ['ImageCLEF_2014-b','ImageCLEF_2014-c',28],
    # usps-mnist 4
    ['USPS','MNIST',28],
    # office_caltech_10 a-w 5
    ["Office_celtech10-caltech","Office_celtech10-webcam",224],
    # mnist-usps 6
    ['MNIST', 'USPS',28],
    # SVHN、USPS->MNIST 7
    [['SVHN','USPS'], 'MNIST',28],

]


def model_feature_tSNE(args,image_size,model,sourceDataLoader,targetDataLoader,image_name,device):
    '''
    :param model: model
    :param sourceDataLoader: source data loader
    :param targetDataLoader: target data loader
    :param image_name: image name
    :param device: cpu or gpu
    :param model_name: the name of model, used for making new folder to save images
    '''
    source_feature = collect_feature(args,image_size,sourceDataLoader, model, device)
    target_feature = collect_feature(args,image_size,targetDataLoader, model, device)
    tSNE_filename = os.path.join('../images/'+args.model_name, image_name+'.png')
    if not os.path.exists('../images/'+args.model_name):
        os.makedirs('../images/'+args.model_name)
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
def collect_feature(args,image_size,data_loader: DataLoader, feature_extractor: nn.Module,
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
            images = images.expand(len(images), args.n_dim, image_size, image_size).to(device)
            try:
                fc=feature_extractor.backbone(images)
            except Exception as e:
                fc = feature_extractor.feature_extractor(images)



            if isinstance(fc,tuple):
                feature = fc[2] # if backbone = AlexNetFc_for_layerWiseAdaptation
            else:
                feature=fc

            try:
                feature = feature_extractor.bottleneck(feature).cpu()
            except Exception as e:
                feature=feature.cpu()

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
    '''
    :param model: the model that need to update
    :param requires_grad: if the model need to update
    :return:
    '''
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


def mini_batch_class_balanced(label, sample_size=20, shuffle=False):
    ''' sample the mini-batch with class balanced
    '''
    if shuffle:
        rindex = np.random.permutation(len(label))
        label = np.array(label)[rindex]

    n_class = len(np.unique(label))
    index = []
    for i in range(n_class):
        s_index = np.nonzero(label == i)
        s_ind = np.random.permutation(s_index[0])
        index = np.append(index, s_ind[0:sample_size])
        #          print(index)
    index = np.array(index, dtype=int)
    return index


# Label propagation
def Label_propagation(Xt, Ys, g, n_labels):
    '''
    :param Xt: batch target Data
    :param Ys: batch source labels
    :param g:
    :param n_labels:
    :return:
    '''

    ys = Ys
    xt = Xt
    yt = np.zeros((n_labels, xt.shape[0]))  # [n_labels,n_target_sample]
    # let labels start from a number
    ysTemp = np.copy(ys)  # ys、ysTemp:[n_source_samples,]
    # classes = np.unique(ysTemp)
    n = n_labels
    ns = len(ysTemp)

    # perform label propagation
    transp = g / np.sum(g, 1)[:, None]  # coupling_[i]:[n_source_samples,n_target_samples]

    # set nans to 0
    transp[~ np.isfinite(transp)] = 0

    D1 = np.zeros((n, ns))  # [n_labels,n_source_samples]

    for c in range(n_labels):
        D1[int(c), ysTemp == c] = 1

    # compute propagated labels
    # / len(ys)=/ k, means uniform sources transfering
    yt = yt + np.dot(D1, transp) / len(
        ys)  # np.dot(D1, transp):[n_labels,n_target_samples] show the mass of every class for transfering to target samples

    return yt.T  # n_samples,n_labels



class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return ReverseLayerF.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start
        The forward and backward behaviours are:
        .. math::
            \mathcal{R}(x) = x,
            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.
        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:
        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo
        where :math:`i` is the iteration step.
        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return ReverseLayerF.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1