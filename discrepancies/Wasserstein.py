# author:LENOVO
# contact: ACoderlyy@163.com
# datetime:2021/9/11 9:59
# software: PyCharm

import torch
import torch.autograd as autograd


def getGradientPenalty(critic,real_samples,fake_samples,args):
    alpha=torch.rand(real_samples.size(0), 1)
    if not args.noCuda and torch.cuda.is_available():
        alpha=alpha.cuda()
    interpolates=alpha*real_samples+(1-alpha)*fake_samples
    interpolates=torch.stack([interpolates,real_samples,fake_samples]).requires_grad_()
    D_interpolates=critic(interpolates)
    gradients=autograd.grad(
        inputs=interpolates,
        outputs=D_interpolates,
        grad_outputs=torch.ones_like(D_interpolates),
        retain_graph=True,create_graph=True,only_inputs=True
    )[0]

    gradient_penalty= ((gradients.norm(2,dim=1)-1)**2).mean()

    return gradient_penalty






def sort_rows(matrix):
    return torch.sort(matrix, descending=True,dim=0)[0]


def discrepancy_slice_wasserstein(p1, p2,args):
    '''
    compute the sliced wasserstein distance
    :param p1: classifier result from source
    :param p2: classifier result from target
    :param device: device
    :return: sliced wasserstein
    '''
    s = p1.size(1)
    if s > 1:
        # For data more than one-dimensional, perform multiple random projection to 1-D
        theta = torch.rand(args.n_labels, 128)
        theta = (theta / torch.sum(theta,dim=0)).to(p1.device)

        p1 = torch.matmul(p1,theta)
        p2 = torch.matmul(p2,theta)
    p1 = sort_rows(p1)
    p2 = sort_rows(p2)
    wdist = (p1-p2)**2
    return torch.mean(wdist)