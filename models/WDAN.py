import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR
import tqdm
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
from backBone import network_dict
from discrepancies.MMD import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel,linear_kernel
from utils import model_feature_tSNE, one_hot_label

def get_Omega(Label,n_labels):
    Omega=np.zeros((n_labels,))
    label=Label.detach().cpu().numpy()
    for i in label:
        Omega[i]+=1
    Omega=Omega/np.sum(Omega)

    return torch.tensor(Omega)
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
    optimizer = optim.SGD([
        {'params': model.backbone.parameters()},
        {'params': model.bottleneck.parameters(), 'lr': args.lr},
        {'params': model.last_classifier.parameters(), 'lr': args.lr}
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
    learningRate = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    clf_criterion = nn.CrossEntropyLoss()
    for epoch in range(1+base_epoch, base_epoch+args.epoch + 1):
        model.train()


        lenSourceDataLoader = len(sourceDataLoader)

        correct = 0
        total_loss = 0
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                              desc='Train epoch = {}'.format(epoch), ncols=80,
                                                              leave=False):

            sourceData, sourceLabel = sourceData.expand(len(sourceData), args.n_dim, imageSize, imageSize).to(
                device), sourceLabel.to(device)

            for targetData, targetLabel in targetDataLoader:
                targetData, targetLabel = targetData.expand(len(targetData), args.n_dim, imageSize, imageSize).to(
                    device), targetLabel.to(device)
                break
            optimizer.zero_grad()
            sourceOutput, targetOutput, pseudo_label, mmd_loss = model.forward(sourceData, targetData, mkmmd_loss,
                                                                               sourceLabel)
            source_pre = sourceOutput.data.max(1, keepdim=True)[1]
            correct += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()
            clf_loss = clf_criterion(sourceOutput, sourceLabel)
            clf_t_loss = clf_criterion(targetOutput, pseudo_label)
            loss = args.lam_cls * clf_loss + args.lam_mmd * mmd_loss + args.gamma * clf_t_loss
            total_loss += clf_loss.item()

            loss.backward(retain_graph=True)
            optimizer.step()
            learningRate.step(epoch)
            if batch_idx % args.logInterval == 0:
                print(
                    '\nLoss: {:.4f},  clf_Loss: {:.4f},lam_clf*clf_Loss:{:.4f}, clf_t_loss:{:.4f},gamma*clf_t_loss:{:.4f}, mmd_loss: {:.4f}, lam_mmd*mmd_loss:{:.4f}'.format(
                        loss.item(), clf_loss.item(), args.lam_cls * clf_loss.item(), clf_t_loss.item(),
                                                      args.gamma * clf_t_loss.item(), mmd_loss.item(),
                                                      args.lam_mmd * mmd_loss.item()))

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
    testLoss = 0
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
            testLoss += F.nll_loss(F.log_softmax(pre_label, dim=1), sourceLabel, size_average=False).item()
            pred = pre_label.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(sourceLabel.data.view_as(pred)).cpu().sum()

        testLoss /= len(sourceTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f},Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, len(sourceTestDataLoader.dataset),
            100. * correct / len(sourceTestDataLoader.dataset)))

    # target Test
    correct = 0
    testLoss = 0
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
            testLoss += F.nll_loss(F.log_softmax(pre_label, dim=1), targetLabel, size_average=False).item()
            pred = pre_label.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()

        testLoss /= len(taragetTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f},Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss,correct, len(taragetTestDataLoader.dataset),
            100. * correct / len(taragetTestDataLoader.dataset)))
    return correct



class WDANModel(nn.Module):
    def __init__(self,device,args,baseNet='AlexNetFc_for_layerWiseAdaptation'):
        super(WDANModel,self).__init__()
        self.backbone_name=baseNet
        self.backbone=network_dict[baseNet]()
        self.device=device

        self.bottleneck = nn.Sequential(
            nn.Linear(1000, args.bottleneck_dim),
            nn.ReLU(),
        )
        self.last_classifier = nn.Sequential(
            nn.Linear(args.bottleneck_dim, args.n_labels)
        )
        self.args=args

    def forward(self,sourceData, targetData,mkmmd_loss,sourceLabel):
        fc6_s, fc7_s, fc8_s = self.backbone(sourceData)
        sourceOutput_neck = self.bottleneck(fc8_s)

        mmd_loss = 0
        fc6_t, fc7_t, fc8_t = self.backbone(targetData)
        targetOutput_neck = self.bottleneck(fc8_t)

        sourceOutput = self.last_classifier(sourceOutput_neck)
        targetOutput = self.last_classifier(targetOutput_neck)

        pseudo_label = (targetOutput.detach().data.max(1)[1]).view_as(sourceLabel)
        Omega_s = get_Omega(sourceLabel, self.args.n_labels)
        Omega_t = get_Omega(pseudo_label, self.args.n_labels)
        Omega = Omega_t / Omega_s
        # Omega/=torch.sum(Omega)
        Omega = Omega.view(len(Omega), 1).float().to(self.device)
        Omega = torch.autograd.Variable(Omega, requires_grad=True)
        sourceLabel_onehot = one_hot_label(sourceLabel, self.args.n_labels).to(self.device)
        source_Omega = torch.matmul(sourceLabel_onehot, Omega).float()
        # print(Omega_t)
        # print(Omega_s)
        # mmd_loss += mkmmd_loss(sourceOutput, targetOutput)
        mmd_loss += mkmmd_loss(fc8_s, fc8_t, source_Omega)
        # mmd_loss += mkmmd_loss(sourceOutput_neck, targetOutput_neck,source_Omega)
        # mmd_loss += mkmmd_loss(sourceOutput, targetOutput,source_Omega)
        mmd_loss += mkmmd_loss(fc6_s, fc6_t, source_Omega)
        mmd_loss += mkmmd_loss(fc7_s, fc7_t, source_Omega)

        return sourceOutput, targetOutput,  pseudo_label, mmd_loss

