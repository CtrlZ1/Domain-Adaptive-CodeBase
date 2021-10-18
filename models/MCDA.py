# author:LENOVO
# contact: ACoderlyy@163.com
# datetime:2021/9/11 10:01
# software: PyCharm
import torch
import tqdm
import torch.nn as nn
import os
import torch.nn.functional as F
from discrepancies.Wasserstein import getGradientPenalty
from utils import one_hot_label, model_feature_tSNE, set_requires_grad


def train_process(model, sourceDataLoader, targetDataLoader,sourceTestDataLoader,taragetTestDataLoader,DEVICE,imageSize,args):
    model.train()

    critics = [i.to(DEVICE).train() for i in model.critics]

    classifier = model.classifier
    par_critic = [
        {'params': c.parameters()} for c in critics
    ]
    critic_optim = torch.optim.Adam(par_critic, lr=args.lr)
    par_clf_wd = [
        {'params': model.feature_extractor.parameters()},
        {'params': model.classifier.parameters()},
    ]
    clf_optim = torch.optim.Adam(par_clf_wd, lr=args.lr)
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
            sourceFeature, targetFeature, sourceLabel_pre, targeteLabel_pre = outPut[0], outPut[1], outPut[2], outPut[3]
            source_pre = sourceLabel_pre.data.max(1, keepdim=True)[1]
            correct += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()

            # Training critic
            for cc in critics:
                set_requires_grad(cc, True)
            Critic_loss = torch.zeros(1).to(DEVICE)
            T = torch.ones(args.n_labels + 1).to(DEVICE)
            ones = torch.ones((args.batchSize, 1)).to(DEVICE)
            T = T / torch.sum(T)
            T[args.n_labels] = 1
            sourceLabel_onehot = one_hot_label(sourceLabel, args.n_labels).to(DEVICE)
            ys_weight = torch.cat([sourceLabel_onehot, ones], dim=1).to(DEVICE)
            yt_weight = torch.cat([targeteLabel_pre, ones], dim=1).to(DEVICE)
            ys_weight = ys_weight / (torch.mean(ys_weight, dim=0) + 1e-6)
            yt_weight = yt_weight / (torch.mean(yt_weight, dim=0) + 1e-6)
            gradient_weights = ys_weight * yt_weight * T
            ys_weight = ys_weight * T
            yt_weight = yt_weight * T

            for _ in range(args.n_critic):
                for i in range(args.n_labels + 1):
                    gp = getGradientPenalty(critics[i], sourceFeature, targetFeature, args)
                    Critic_loss += (-ys_weight[:, i].view(-1, 1) * critics[i](sourceFeature.detach())).mean() + \
                                   (yt_weight[:, i].view(-1, 1) * critics[i](targetFeature.detach())).mean() + \
                                   (gradient_weights[:, i].view(-1, 1) * args.lambda_gp * gp).mean()

                critic_optim.zero_grad()
                Critic_loss.backward(retain_graph=True)

                critic_optim.step()
            # Training classifier
            for cc in critics:
                set_requires_grad(cc, False)

            for _ in range(args.n_clf):
                clf_trust_target_loss = torch.zeros(1).to(DEVICE)
                # I think trust target datas will bring errors to model,so i stop compute clf_trust_target_loss as following.
                # num = 0
                # for i in targeteLabel_pre:
                #     if i.data.max() > args.theta:
                #         num += 1
                #         label = i.view(1, -1).data.max(1)[1]
                #         clf_trust_target_loss += clf_criterion(i.view(1, -1), label)
                clf_loss = clf_criterion(sourceLabel_pre, sourceLabel)
                wd_loss = torch.zeros(1).to(DEVICE)
                for i in range(args.n_labels + 1):
                    if i != args.n_labels:
                        alpha = args.alpha
                    else:
                        alpha = alpha * args.n_labels
                    wd_loss += alpha * ((ys_weight[:, i] * critics[i](sourceFeature)).mean() - (
                            yt_weight[:, i] * critics[i](targetFeature)).mean())

                # classifer_and_wd_loss = (clf_loss * (len(sourceLabel) / (num + len(sourceLabel))) + (
                #             clf_trust_target_loss / num) * (num / (num + len(sourceLabel)))) + args.n_clf * wd_loss
                classifer_and_wd_loss = clf_loss+args.n_clf * wd_loss
                clf_optim.zero_grad()
                classifer_and_wd_loss.backward()
                clf_optim.step()
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

            }
        else:
            state = {
                'epoch': args.epoch,
                'net': model.state_dict(),
                'critic_optim': critic_optim.state_dict(),
                'clf_optim': clf_optim.state_dict(),

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
            Output = model.classifier(model(data, data, False)[0])
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
            Output = model.classifier(model(data, data, False)[0])
            testLoss += F.nll_loss(F.log_softmax(Output, dim=1), targetLabel,
                                   size_average=False).item()  # sum up batch loss
            pred = Output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()
        testLoss /= len(taragetTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, target Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, len(taragetTestDataLoader.dataset),
            100. * correct / len(taragetTestDataLoader.dataset)))
    return correct





class MCDAModel(nn.Module):

    def __init__(self, args):
        super(MCDAModel, self).__init__()
        self.args=args
        self.softmax = nn.Softmax(dim=1)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(args.n_dim, 32, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(1024,1024),
            nn.LeakyReLU(),
            nn.Dropout2d(0)
        )


        self.classifier = nn.Sequential(
            nn.Linear(1024, args.n_labels),
        )

        # D
        self.critics=[nn.Sequential(
            nn.Linear(1024,args.critic_dim),
            nn.LeakyReLU(),
            nn.Linear(args.critic_dim,1)
        ) for i in range(args.n_labels+1)]



    def forward(self,sourceData,targetData,training):


        # 提取数据特征
        sourceFeature = self.feature_extractor(sourceData)
        targetFeature = self.feature_extractor(targetData)

        sourceLabel=self.classifier(sourceFeature)
        targeteLabel=self.classifier(targetFeature)


        if training:

            return sourceFeature,targetFeature,sourceLabel,targeteLabel
        else:
            return targetFeature,targeteLabel
