# author:LENOVO
# contact: ACoderlyy@163.com
# datetime:2021/9/12 9:24
# software: PyCharm

import torch
import tqdm
import torch.nn as nn
import os
import torch.nn.functional as F

from backBone import network_dict
from discrepancies.Wasserstein import getGradientPenalty
from utils import one_hot_label, model_feature_tSNE, set_requires_grad


def train_process(model, sourceDataLoader, targetDataLoader,sourceTestDataLoader,taragetTestDataLoader,DEVICE,imageSize,args):
    model.train()
    critic = model.critic
    classifier = model.classifier
    feature_extractor=model.feature_extractor
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr)
    clf_optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    wd_optim = torch.optim.Adam(feature_extractor.parameters(), lr=args.lr)
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
        critic_optim.load_state_dict(checkpoint['critic_optim'])
        clf_optim.load_state_dict(checkpoint['clf_optim'])
        wd_optim.load_state_dict(checkpoint['wd_optim'])
        base_epoch = checkpoint['epoch']
    t_correct=0
    for epoch in range(1 + base_epoch, base_epoch + args.epoch + 1):
        model.train()
        correct = 0
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                              desc='Train epoch = {}'.format(epoch), ncols=80,
                                                              leave=False):

            sourceData, sourceLabel = sourceData.expand(len(sourceData), args.n_dim, imageSize, imageSize).to(
                DEVICE), sourceLabel.to(DEVICE)

            for targetData, targetLabel in targetDataLoader:
                targetData, targetLabel = targetData.expand(len(targetData), args.n_dim, imageSize, imageSize).to(
                    DEVICE), targetLabel.to(DEVICE)
                break

            outPut = model(sourceData, targetData, True)
            h_s, h_t = outPut[0], outPut[1]




            # Training critic
            set_requires_grad(critic, True)
            for _ in range(args.n_critic):
                gp = getGradientPenalty(critic, h_s.detach(), h_t.detach(), args)

                Critic_loss = -critic(h_s.detach()).mean() + critic(h_t.detach()).mean() + args.lambda_gp * gp
                # Critic_loss*=10
                critic_optim.zero_grad()
                Critic_loss.backward(retain_graph=True)

                critic_optim.step()
            # Training classifier
            set_requires_grad(critic, False)

            for _ in range(args.n_clf):

                source_preds = classifier(h_s)
                source_pre = source_preds.data.max(1, keepdim=True)[1]
                correct += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()
                clf_loss = clf_criterion(source_preds, sourceLabel)

                clf_optim.zero_grad()
                clf_loss.backward()
                clf_optim.step()


                source_features, target_features = model(sourceData, targetData, True)

                wasserstein_distance = critic(source_features).mean() - critic(target_features).mean()
                wd_loss = args.n_clf * wasserstein_distance

                wd_optim.zero_grad()
                wd_loss.backward()
                wd_optim.step()
                # for par in model.feature_extractor.parameters():
                #    print(par.grad)

            if batch_idx % args.logInterval == 0:
                print(
                    '\ncritic_loss: {:.4f},  classifer_loss: {:.4f},  wd_Loss: {:.6f}'.format(
                        Critic_loss.item(), clf_loss.item(), wd_loss.item()))


        acc_train = float(correct) * 100. / (lenSourceDataLoader * args.batchSize)

        print('Train Accuracy: {}/{} ({:.2f}%)'.format(
            correct, (lenSourceDataLoader * args.batchSize), acc_train))

        test_correct=test_process(model, sourceTestDataLoader,taragetTestDataLoader, DEVICE, args)
        if test_correct > t_correct:
            t_correct = test_correct
        print("max correct:" , t_correct)
        if epoch % args.logInterval == 0:
            model_feature_tSNE(args, imageSize, model, sourceTestDataLoader, taragetTestDataLoader,
                               'epoch' + str(epoch), DEVICE)

    if args.ifsave:
        path=args.savePath+args.model_name
        if not os.path.exists(path):
            os.makedirs(path)

        if args.if_saveall:
            state = {
                'epoch': args.epoch,
                'net': model,
                'critic_optim': critic_optim,
                'clf_optim': clf_optim,
                'wd_optim':wd_optim
            }
        else:
            state = {
                'epoch': args.epoch,
                'net': model.state_dict(),
                'critic_optim': critic_optim.state_dict(),
                'clf_optim': clf_optim.state_dict(),
                'wd_optim':wd_optim.state_dict()
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
            Output = model.classifier(model(data, data, False))
            testLoss += F.nll_loss(F.log_softmax(Output, dim=1), suorceLabel,
                                   size_average=False).item()  # sum up batch loss
            pred = Output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(suorceLabel.data.view_as(pred)).cpu().sum()
        testLoss /= len(sourceTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, source Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss,correct, len(sourceTestDataLoader.dataset),
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
            Output = model.classifier(model(data, data, False))
            testLoss += F.nll_loss(F.log_softmax(Output, dim=1), targetLabel,
                                   size_average=False).item()  # sum up batch loss
            pred = Output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()

        testLoss /= len(taragetTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, target Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, len(taragetTestDataLoader.dataset),
            100. * correct / len(taragetTestDataLoader.dataset)))
    return correct






class WDGRLModel(nn.Module):

    def __init__(self, args):
        super(WDGRLModel, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.args=args
        self.feature_extractor = network_dict['LeNet'](args.n_dim)


        # 标签分类器Gy
        # 源数据全连接层
        self.classifier = nn.Sequential()
        # 划分十类
        self.classifier.add_module('c_fc3', nn.Linear(256, self.args.n_labels))

        # WD
        self.critic=nn.Sequential(
            nn.Linear(256,500),
            nn.LeakyReLU(),
            nn.Linear(500,500),
            nn.LeakyReLU(),
            nn.Linear(500,1)
        )


    def forward(self,sourceData,targetData,training):
        # 提取数据特征
        sourceFeature = self.feature_extractor(sourceData)
        targetFeature = self.feature_extractor(targetData)

        h_s=sourceFeature.view(sourceFeature.size(0),-1)
        h_t=targetFeature.view(targetFeature.size(0),-1)

        if training:

            return h_s,h_t
        else:
            return h_t
