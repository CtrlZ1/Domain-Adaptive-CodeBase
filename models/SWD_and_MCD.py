# author:ACoderlyy
# contact: ACoderlyy@163.com
# datetime:2021/9/19 10:20
# software: PyCharm

import torch
import tqdm
import torch.nn as nn
import os
import torch.nn.functional as F
from discrepancies.Wasserstein import getGradientPenalty, discrepancy_slice_wasserstein
from utils import one_hot_label, model_feature_tSNE, set_requires_grad
import torch.optim as optim
def discrepancy_mcd(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))


def train_process(model, sourceDataLoader, targetDataLoader,sourceTestDataLoader,taragetTestDataLoader,DEVICE,imageSize,mode,args):
    model.train()

    parameters = [
        {'params': model.classifier1.parameters()},
        {'params': model.classifier2.parameters()},
    ]
    optimizer_all = optim.Adam(model.parameters(), args.lr)
    optimizer_cls_dis = optim.Adam(parameters, args.lr)
    optimizer_dist = optim.Adam(model.feature_extractor.parameters(), args.lr)

    clf_criterion = nn.CrossEntropyLoss()

    lenSourceDataLoader = len(sourceDataLoader)

    base_epoch = 0
    if args.ifload:
        path = args.savePath + args.model_name
        for i in os.listdir(path):
            path2 = os.path.join(path, i)
            break
        checkpoint = torch.load(path2)
        model.load_state_dict(checkpoint['net'])
        optimizer_all.load_state_dict(checkpoint['optimizer_all'])
        optimizer_cls_dis.load_state_dict(checkpoint['optimizer_cls_dis'])
        optimizer_dist.load_state_dict(checkpoint['optimizer_dist'])
        base_epoch = checkpoint['epoch']
    t_correct=0

    for epoch in range(1 + base_epoch, base_epoch + args.epoch + 1):
        model.train()
        correct1 = 0
        correct2 = 0
        total_loss = 0
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                              desc='Train epoch = {}'.format(epoch), ncols=80,
                                                              leave=False):

            sourceData, sourceLabel = sourceData.expand(len(sourceData), args.n_dim, imageSize, imageSize).to(
                DEVICE), sourceLabel.to(DEVICE)

            for targetData, targetLabel in targetDataLoader:
                targetData, targetLabel = targetData.expand(len(targetData), args.n_dim, imageSize, imageSize).to(
                    DEVICE), targetLabel.to(DEVICE)
                break

            optimizer_all.zero_grad()

            clf_loss = 0
            sourceOutput = model.classifier1(model.feature_extractor(sourceData))
            source_pre = sourceOutput.data.max(1, keepdim=True)[1]
            correct1 += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()
            clf_loss += clf_criterion(sourceOutput, sourceLabel)

            sourceOutput = model.classifier2(model.feature_extractor(sourceData))
            source_pre = sourceOutput.data.max(1, keepdim=True)[1]
            correct2 += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()
            clf_loss += clf_criterion(sourceOutput, sourceLabel)
            total_loss += clf_loss.item()
            clf_loss.backward()
            optimizer_all.step()

            optimizer_cls_dis.zero_grad()
            p1 = model.classifier1(model.feature_extractor(targetData))
            p2 = model.classifier2(model.feature_extractor(targetData))

            if mode == 'adapt_swd':
                loss_dist = discrepancy_slice_wasserstein(p1, p2, args)
            elif mode == 'MCD':
                loss_dist = discrepancy_mcd(p1, p2)

            loss_clf_dist = 0
            sourceOutput = model.classifier1(model.feature_extractor(sourceData))
            loss_clf_dist += clf_criterion(sourceOutput, sourceLabel)

            sourceOutput = model.classifier2(model.feature_extractor(sourceData))
            loss_clf_dist += clf_criterion(sourceOutput, sourceLabel)
            loss_clf_dist -= loss_dist
            total_loss += loss_clf_dist.item()
            loss_clf_dist.backward()

            optimizer_cls_dis.step()

            if mode == 'adapt_swd':  # adapt_swd
                optimizer_dist.zero_grad()
                p1 = model.classifier1(model.feature_extractor(targetData))
                p2 = model.classifier2(model.feature_extractor(targetData))
                loss_dist = discrepancy_slice_wasserstein(p1, p2, args)
                total_loss += loss_dist.item()
                loss_dist.backward()
                optimizer_dist.step()
                # for i in range(args.num_k):
                #     optimizer_dist.zero_grad()
                #     p1 = model.classifier1(model.feature_extractor(targetData))
                #     p2 = model.classifier2(model.feature_extractor(targetData))
                #     loss_dist = discrepancy_slice_wasserstein(p1, p2,args)
                #     loss_dist.backward()
                #     optimizer_dist.step()
                # total_loss += loss_dist.item()
            elif mode == 'MCD':
                for i in range(args.num_k):
                    optimizer_dist.zero_grad()
                    p1 = model.classifier1(model.feature_extractor(targetData))
                    p2 = model.classifier2(model.feature_extractor(targetData))
                    loss_dist = discrepancy_mcd(p1, p2)
                    loss_dist.backward()
                    optimizer_dist.step()
                total_loss += loss_dist.item()

            if batch_idx % args.logInterval == 0:
                print(
                    '\nclf_Loss: {:.4f},  loss_clf_dist: {:.4f}, loss_dist:{:.4f}'.format(
                        clf_loss.item(), loss_clf_dist.item(), loss_dist.item()))

        total_loss /= lenSourceDataLoader
        acc_train = float(correct1) * 100. / (lenSourceDataLoader * args.batchSize)

        print('Average classification loss: {:.4f}, Train Accuracy: {}/{} ({:.2f}%)'.format(
             total_loss, correct1, (lenSourceDataLoader * args.batchSize), acc_train))

        test_correct=test_process(model, sourceTestDataLoader,taragetTestDataLoader, DEVICE, args)
        if test_correct > t_correct:
            t_correct = test_correct
        print("max correct:" , t_correct)
        if epoch % args.logInterval == 0:
            model_feature_tSNE(args,imageSize,model, sourceTestDataLoader, taragetTestDataLoader, 'epoch' + str(epoch), DEVICE)

    if args.ifsave:
        path=args.savePath+args.model_name
        if not os.path.exists(path):
            os.makedirs(path)

        if args.if_saveall:
            state = {
                'epoch': args.epoch,
                'net': model,
                'optimizer_all': optimizer_all,
                'optimizer_dist': optimizer_dist,

            }
        else:
            state = {
                'epoch': args.epoch,
                'net': model.state_dict(),
                'optimizer_all': optimizer_all.state_dict(),
                'optimizer_cls_dis': optimizer_cls_dis.state_dict(),
                'optimizer_dist': optimizer_dist.state_dict(),

            }
        path+='/'+args.model_name+'_epoch'+str(args.epoch)+'.pth'
        torch.save(state, path)


def test_process(model,sourceTestDataLoader,taragetTestDataLoader, device, args):
    model.eval()


    # source Test
    correct = 0
    testLoss = 0
    with torch.no_grad():
        for data, suorceLabel in sourceTestDataLoader:
            if args.n_dim == 0:
                data, suorceLabel = data.to(args.device), suorceLabel.to(args.device)
            elif args.n_dim > 0:
                imgSize = torch.sqrt(
                    (torch.prod(torch.tensor(data.size())) / (data.size(1) * len(data))).float()).int()
                data = data.expand(len(data), args.n_dim, imgSize.item(), imgSize.item()).to(
                    device)
                suorceLabel = suorceLabel.to(device)
            Output = model.classifier1(model.feature_extractor(data))
            testLoss += F.nll_loss(F.log_softmax(Output, dim=1), suorceLabel,
                                   size_average=False).item()  # sum up batch loss
            pred = Output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(suorceLabel.data.view_as(pred)).cpu().sum()
        testLoss /= len(sourceTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, source Test Accuracy: {}/{} ({:.0f}%)\n'.format(
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
            Output = model.classifier1(model.feature_extractor(data))
            testLoss += F.nll_loss(F.log_softmax(Output, dim=1), targetLabel,
                                   size_average=False).item()  # sum up batch loss
            pred = Output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()
        testLoss /= len(taragetTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, target Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, len(taragetTestDataLoader.dataset),
            100. * correct / len(taragetTestDataLoader.dataset)))
    return correct




class SWD_MCDModel(nn.Module):
    def __init__(self,device,args):
        super(SWD_MCDModel,self).__init__()
        self.device=device
        self.args=args
        self.feature_extractor=nn.Sequential(
            nn.Conv2d(args.n_dim, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(768, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, args.n_labels),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(768, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, args.n_labels),
        )