# author:LENOVO
# contact: ACoderlyy@163.com
# datetime:2021/9/15 9:47
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

from discrepancies.MMD import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel
from utils import model_feature_tSNE

def model_test(model,targetData):
    model.eval()
    Output = model.backbone(targetData)
    Output = model.bottleneck(Output)
    Output = model.last_classifier(Output)

    return Output

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
    ], lr=args.lr / 10, momentum=args.momentum, weight_decay=args.l2Decay)

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
            print(f'Learning Rate: {learningRate.get_lr()}')
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
            model_feature_tSNE(model, sourceTestDataLoader, taragetTestDataLoader, 'epoch'+str(epoch), device,args.model_name)

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
        print('\nTest set: Average loss: {:.4f},Test Accuracy: {}/{} ({:.0f}%)\n'.format(
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
        print('\nTest set: Average loss: {:.4f},Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            clsdLoss,correct, len(taragetTestDataLoader.dataset),
            100. * correct / len(taragetTestDataLoader.dataset)))
    return correct



class DDCModel(nn.Module):
    def __init__(self,args,device):
        super(DDCModel,self).__init__()
        modelAlexNet=models.alexnet(pretrained=True)
        self.device=device
        self.backbone=modelAlexNet

        self.bottleneck = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True)
        )
        self.last_classifier = nn.Sequential(
            nn.Linear(256, args.n_labels)
        )

    def forward(self,sourceData, targetData,mmd):

        sourcefeature = self.backbone(sourceData)
        sourceOutput = self.bottleneck(sourcefeature)

        targetfeature = self.backbone(targetData)
        targetOutput = self.bottleneck(targetfeature)
        mmd_loss = mmd(sourceOutput, targetOutput)

        sourceOutput = self.last_classifier(sourceOutput)

        return sourceOutput, mmd_loss
