# author:ACoderlyy
# contact: ACoderlyy@163.com
# datetime:2021/9/19 15:08
# software: PyCharm
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD

import tqdm

class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x


def A_dis_calculate(source_feature: torch.Tensor, target_feature: torch.Tensor,
              device, progress=False, training_epochs=10):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.

    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier

    Returns:
        :math:`\mathcal{A}`-distance
    """
    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))
    feature = torch.cat([source_feature, target_feature], dim=0)
    label = torch.cat([source_label, target_label], dim=0)

    dataset = TensorDataset(feature, label)
    length = len(dataset)
    train_size = int(0.8 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    anet = ANet(feature.shape[1]).to(device)
    optimizer = SGD(anet.parameters(), lr=0.01)
    a_distance = 2.0
    for epoch in tqdm.tqdm(range(training_epochs), total=training_epochs,
                                                              desc='A-distance computing...', ncols=80,
                                                              leave=False):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward(retain_graph=True)
            optimizer.step()

        anet.eval()
        correct = 0
        allnumber=0
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.long().to(device)
                y = anet(x)
                pred = y.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(label.data.view_as(pred)).cpu().sum()
                allnumber+=x.size(0)
        error = 1 - float(correct) / allnumber
        a_distance = 2 * (1 - 2 * error)
        if progress:
            print("epoch {}  A-dist: {}".format(epoch, a_distance))

    return a_distance