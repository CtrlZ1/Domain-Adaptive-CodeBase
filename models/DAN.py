# author:LENOVO
# contact: ACoderlyy@163.com
# datetime:2021/9/16 9:30
# software: PyCharm

import os
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models
import tqdm
import math
import torch.optim as optim
import torch.nn.functional as F

from backBone import network_dict
from discrepancies.MMD import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel
from utils import model_feature_tSNE

def model_test(model,targetData):
    model.eval()
    fc6_s, fc7_s, fc8_s = model.backbone(targetData)
    targetOutput = fc8_s
    targetOutput = model.bottleneck(targetOutput)
    targetOutput = model.last_classifier(targetOutput)

    return targetOutput

def train_process(model,sourceDataLoader, targetDataLoader,sourceTestDataLoader,taragetTestDataLoader, device,imageSize,args):

    # define loss function
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        # kernels=[linear_kernel()],
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=args.linear
    )

    backbone=model.backbone
    bottleneck=model.bottleneck
    last_classifier=model.last_classifier

    optimizer = optim.SGD([
        {'params': backbone.parameters()},
        {'params': bottleneck.parameters(), 'lr': args.lr},
        {'params': last_classifier.parameters(), 'lr': args.lr}
    ], lr=args.lr / 10, momentum=args.momentum, weight_decay=args.l2_Decay)

    base_epoch=0
    if args.ifload:
        path = args.savePath + args.model_name
        for i in os.listdir(path):
            path2=os.path.join(path,i)
            break
        checkpoint = torch.load(path2)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        base_epoch = checkpoint['epoch']

    # learningRate = args.lr / math.pow((1 + 10 * (epoch - 1) / args.epoch), 0.75)
    learningRate = LambdaLR(optimizer, lambda x: (1. + args.lr_gamma * (float(x)/(base_epoch+args.epoch))) ** (-args.lr_decay))

    clf_criterion = nn.CrossEntropyLoss()
    for epoch in range(1+base_epoch, base_epoch+args.epoch + 1):
        model.train()


        lenSourceDataLoader = len(sourceDataLoader)

        correct = 0
        total_loss = 0
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                              desc='Train epoch = {}'.format(epoch), ncols=80,
                                                              leave=False):
            # print(f'Learning Rate: {learningRate.get_lr()}')
            sourceData, sourceLabel = sourceData.expand(len(sourceData), args.n_dim, imageSize, imageSize).to(
                device), sourceLabel.to(device)

            for targetData, targetLabel in targetDataLoader:
                targetData, targetLabel = targetData.expand(len(targetData), args.n_dim, imageSize, imageSize).to(
                    device), targetLabel.to(device)
                break
            optimizer.zero_grad()
            sourceOutput, mmd_loss = model(sourceData, targetData,mkmmd_loss)

            source_pre = sourceOutput.data.max(1, keepdim=True)[1]
            correct += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()
            clf_loss = clf_criterion(sourceOutput, sourceLabel)
            loss = clf_loss + args.lamb * mmd_loss
            total_loss += clf_loss.item()

            loss.backward()
            optimizer.step()

            learningRate.step(epoch)
            if batch_idx % args.logInterval == 0:
                print(
                    '\nLoss: {:.4f},  clf_Loss: {:.4f},mmd_loss:{:.4f},lamb*mmd_loss:{:.4f}'.format(
                        loss.item(), clf_loss.item(), mmd_loss.item(),args.lamb * mmd_loss.item(), mmd_loss.item()))

        total_loss /= lenSourceDataLoader
        acc_train = float(correct) * 100. / (lenSourceDataLoader * args.batchSize)

        print('Average classification loss: {:.4f}, Train Accuracy: {}/{} ({:.2f}%)'.format(
            total_loss, correct, (lenSourceDataLoader * args.batchSize), acc_train))

        test_process(model,sourceTestDataLoader,taragetTestDataLoader, device,args)
        if epoch%args.logInterval==0:
            model_feature_tSNE(args, imageSize, model, sourceTestDataLoader, taragetTestDataLoader,
                               'epoch' + str(epoch), device)

    if args.ifsave:
        path=args.savePath+args.model_name
        if not os.path.exists(path):
            os.makedirs(path)

        if args.if_saveall:
            state = {
                'epoch': args.epoch,
                'net': model,
                'optimizer': optimizer,

            }
        else:
            state = {
                'epoch': args.epoch,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),

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
            pre_label = model_test(model, data)
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
            pre_label = model_test(model,data)
            clsdLoss += F.nll_loss(F.log_softmax(pre_label, dim=1), targetLabel, size_average=False).item()
            pred = pre_label.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()

        clsdLoss /= len(taragetTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, target Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            clsdLoss,correct, len(taragetTestDataLoader.dataset),
            100. * correct / len(taragetTestDataLoader.dataset)))
    return correct


class DANModel(nn.Module):
    def __init__(self,device,args,baseNet='AlexNetFc_for_layerWiseAdaptation'):
        super(DANModel,self).__init__()
        self.backbone=network_dict[baseNet]()
        self.device=device

        self.bottleneck = nn.Sequential(
            nn.Linear(1000, args.bottleneck_dim),
            nn.ReLU(),
        )
        self.last_classifier = nn.Sequential(
            nn.Linear(args.bottleneck_dim, args.n_labels)
        )

    def forward(self,sourceData, targetData,mkmmd_loss):
        fc6_s, fc7_s, fc8_s = self.backbone(sourceData)
        sourceOutput = fc8_s
        sourceOutput = self.bottleneck(sourceOutput)

        mmd_loss = 0
        fc6_t, fc7_t, fc8_t = self.backbone(targetData)
        targetOutput = fc8_t
        targetOutput = self.bottleneck(targetOutput)
        # mmd_loss += mkmmd_loss(sourceOutput, targetOutput)
        mmd_loss += mkmmd_loss(fc8_s, fc8_t)
        mmd_loss += mkmmd_loss(sourceOutput, targetOutput)

        sourceOutput = self.last_classifier(sourceOutput)
        targetOutput = self.last_classifier(targetOutput)
        mmd_loss += mkmmd_loss(sourceOutput, targetOutput)

        return sourceOutput, mmd_loss


