# author:LENOVO
# contact: ACoderlyy@163.com
# datetime:2021/9/17 9:28
# software: PyCharm
import ot
import os
import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models
import tqdm
import math
import torch.optim as optim
import torch.nn.functional as F

from backBone import network_dict
from discrepancies.Euclidean import euclidean_dist
from discrepancies.MMD import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel
from utils import model_feature_tSNE, mini_batch_class_balanced, one_hot_label, Label_propagation


def train_process(model,source,target,sourceDataLoader, targetDataLoader,sourceTestDataLoader,taragetTestDataLoader, device,imageSize,args,method='sinkhorn', metric='deep', reg_sink=1):

    source_loader = torch.utils.data.DataLoader(dataset=source, batch_size=len(source), shuffle=True)
    target_loader = torch.utils.data.DataLoader(dataset=target, batch_size=len(target), shuffle=True)

    feature_extractor=model.feature_extractor_digits
    classifier=model.classifier_digits

    optimizer_F = optim.SGD(feature_extractor.parameters(), lr=args.lrf, momentum=args.momentum, weight_decay=args.l2_Decay,
                            nesterov=True)
    optimizer_C = optim.SGD(classifier.parameters(), lr=args.lrc, momentum=args.momentum,
                            weight_decay=args.l2_Decay, nesterov=True)
    clf_criterion = nn.CrossEntropyLoss()

    base_epoch=0
    if args.ifload:
        path = args.savePath + args.model_name
        for i in os.listdir(path):
            path2=os.path.join(path,i)
            break
        checkpoint = torch.load(path2)
        model.load_state_dict(checkpoint['net'])
        optimizer_F.load_state_dict(checkpoint['optimizer_F'])
        optimizer_C.load_state_dict(checkpoint['optimizer_C'])
        base_epoch = checkpoint['epoch']


    sourcedata = iter(source_loader)
    (allsourceData, allsourceLabel) = next(sourcedata)

    targetdata = iter(target_loader)
    (alltargetData, alltargetLabel) = next(targetdata)

    for epoch in range(1+base_epoch, base_epoch+args.epoch + 1):
        model.train()
        label_propagation_correct = 0

        lenSourceDataLoader = len(sourceDataLoader)

        correct = 0
        total_loss = 0
        for batch_idx in tqdm.tqdm(range(lenSourceDataLoader), total=lenSourceDataLoader,
                                                              desc='Train epoch = {}'.format(epoch), ncols=80,
                                                              leave=False):
            # all
            # for batch_idx, (sourceData, sourceLabel) in enumerate(sourceDataLoader):
            s_index = mini_batch_class_balanced(allsourceLabel.numpy(), args.sample_size, False)
            sourceData = allsourceData[s_index]
            sourceLabel = allsourceLabel[s_index]
            t_index = np.random.choice(len(alltargetData), args.batchSize)
            targetData = alltargetData[t_index]
            targetLabel = alltargetLabel[t_index]

            optimizer_F.zero_grad()
            optimizer_C.zero_grad()

            sourceData, sourceLabel = sourceData.expand(len(sourceData), args.n_dim, imageSize, imageSize).to(
                device), sourceLabel.to(device)

            for targetData, targetLabel in targetDataLoader:
                targetData, targetLabel = targetData.expand(len(targetData), args.n_dim, imageSize, imageSize).to(
                    device), targetLabel.to(device)
                break

            fea_source = feature_extractor(sourceData)
            fea_target = feature_extractor(targetData)

            if metric == 'original':
                C0 = euclidean_dist(sourceData, targetData, square=True)
            elif metric == 'deep':
                C0 = euclidean_dist(fea_source, fea_target, square=True)

            pre_targetlabel = classifier(fea_target)
            pre_sourcelabel = classifier(fea_source)

            source_pre = pre_sourcelabel.data.max(1, keepdim=True)[1]
            correct += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()

            one_hot_sourcelabel = one_hot_label(sourceLabel, args.n_labels).to(device)
            C1 = euclidean_dist(one_hot_sourcelabel, pre_targetlabel, square=True)

            C = args.alpha * C0 + args.alpha2 * C1

            if method == 'sinkhorn':
                gamma = ot.sinkhorn(ot.unif(fea_source.size(0)), ot.unif(fea_target.size(0)),
                                    C.detach().cpu().numpy(), reg=reg_sink)
            elif method == 'emd':
                gamma = ot.emd(ot.unif(fea_source.size(0)), ot.unif(fea_target.size(0)), C.detach().cpu().numpy())


            propagate_mat = Label_propagation(targetData.detach().cpu().numpy(),
                                                   sourceLabel.view(len(sourceLabel), ).detach().cpu().numpy(),
                                                   gamma, args.n_labels)
            propagate_label = np.argmax(propagate_mat, axis=1)
            correct_p = (propagate_label == targetLabel.detach().cpu().numpy()).sum()
            label_propagation_correct += correct_p
            # print(label_propagation_correct)

            l_c = args.train_par * torch.sum(C * torch.tensor(gamma).float().to(device))
            l_t = args.lam * clf_criterion(pre_sourcelabel, sourceLabel)
            loss = l_c + l_t
            trainingloss = loss.item()
            trainingl_c = l_c.item()
            trainingl_t = l_t.item()
            loss.backward()

            optimizer_F.step()
            optimizer_C.step()

            if batch_idx % args.logInterval == 0:
                print("training loss:{:.4f},l_c:{:.4f},l_t:{:.4f}".format(trainingloss, trainingl_c, trainingl_t))

        total_loss /= lenSourceDataLoader
        acc_train = float(correct) * 100. / (lenSourceDataLoader * args.batchSize)

        print('Source Average classification loss: {:.4f}, Train Accuracy: {}/{} ({:.2f}%)'.format(
            total_loss, correct, (lenSourceDataLoader * args.batchSize), acc_train))

        print("compute training and test acc...")
        test_process(model,sourceTestDataLoader,taragetTestDataLoader, device,args)
        # label propagation
        print("label propagation acc...")
        print(float(label_propagation_correct) / (lenSourceDataLoader * args.batchSize))

        # if epoch%args.logInterval==0:
        #     model_feature_tSNE(args, imageSize, model, sourceTestDataLoader, taragetTestDataLoader,
        #                        'epoch' + str(epoch), device)

    if args.ifsave:
        path=args.savePath+args.model_name
        if not os.path.exists(path):
            os.makedirs(path)

        if args.if_saveall:
            state = {
                'epoch': args.epoch,
                'net': model,
                'optimizer_F': optimizer_F,
                'optimizer_C': optimizer_C

            }
        else:
            state = {
                'epoch': args.epoch,
                'net': model.state_dict(),
                'optimizer_F': optimizer_F.state_dict(),
                'optimizer_C': optimizer_C.state_dict()

            }
        path+='/'+args.model_name+'_epoch'+str(args.epoch)+'.pth'
        torch.save(state, path)



def test_process(model,sourceTestDataLoader,taragetTestDataLoader, device, args):
    model.eval()

    # source Test
    correct = 0
    clsdLoss = 0
    with torch.no_grad():
        for data, sourceLabel in sourceTestDataLoader:
            if args.n_dim == 0:
                data, sourceLabel = data.to(args.device), sourceLabel.to(args.device)
            elif args.n_dim > 0:
                imgSize = torch.sqrt(
                    (torch.prod(torch.tensor(data.size())) / (data.size(1) * len(data))).float()).int()
                data = data.expand(len(data), args.n_dim, imgSize.item(), imgSize.item()).to(
                    device)
                sourceLabel = sourceLabel.to(device)
            pre_label = model(data)
            clsdLoss += F.nll_loss(F.log_softmax(pre_label, dim=1), sourceLabel, size_average=False).item()
            pred = pre_label.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(sourceLabel.data.view_as(pred)).cpu().sum()

        clsdLoss /= len(sourceTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, source Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            clsdLoss, correct, len(sourceTestDataLoader.dataset),
            100. * correct / len(sourceTestDataLoader.dataset)))

    # target Test
    correct = 0
    clsdLoss = 0
    with torch.no_grad():
        for data, targetLabel in taragetTestDataLoader:
            if args.n_dim == 0:
                data, targetLabel = data.to(args.device), targetLabel.to(args.device)
            elif args.n_dim > 0:
                imgSize = torch.sqrt(
                    (torch.prod(torch.tensor(data.size())) / (data.size(1) * len(data))).float()).int()
                data = data.expand(len(data), args.n_dim, imgSize.item(), imgSize.item()).to(
                    device)
                targetLabel = targetLabel.to(device)
            pre_label = model(data)
            clsdLoss += F.nll_loss(F.log_softmax(pre_label, dim=1), targetLabel, size_average=False).item()
            pred = pre_label.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()

        clsdLoss /= len(taragetTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, target Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            clsdLoss,correct, len(taragetTestDataLoader.dataset),
            100. * correct / len(taragetTestDataLoader.dataset)))
    return correct


class DeepJDOTModel(nn.Module):

    def __init__(self, args, device):
        super(DeepJDOTModel, self).__init__()
        self.device = device
        self.n_labels = args.n_labels

        self.feature_extractor_digits = nn.Sequential(
            nn.Conv2d(args.n_dim, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(32768, 128),
            nn.ReLU(True),
        ).to(self.device)
        self.classifier_digits = nn.Sequential(
            nn.Linear(128, self.n_labels),
            # nn.Softmax()
        ).to(self.device)

    def forward(self, Data):
        output = self.classifier_digits(self.feature_extractor_digits(Data))
        return output