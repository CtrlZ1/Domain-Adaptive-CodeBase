# author:LENOVO
# contact: ACoderlyy@163.com
# datetime:2021/9/14 9:33
# software: PyCharm
import torch
import torch.nn as nn
import numpy as np
import os
import tqdm
from torchvision.utils import save_image

from utils import compute_gradient_penalty


def train_process(model, sourceDataLoader,DEVICE,imageSize,args):
    model.train()
    feature_extractor=model.feature_extractor
    critic = model.critic

    # Optimizers
    optimizer_G = torch.optim.RMSprop(feature_extractor.parameters(), lr=args.lr)
    optimizer_D = torch.optim.RMSprop(critic.parameters(), lr=args.lr)

    lenSourceDataLoader = len(sourceDataLoader)

    base_epoch = 0
    if args.ifload:
        path = args.savePath + args.model_name
        for i in os.listdir(path):
            path2 = os.path.join(path, i)
            break
        checkpoint = torch.load(path2)
        model.load_state_dict(checkpoint['net'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        base_epoch = checkpoint['epoch']


    for epoch in range(1 + base_epoch, base_epoch + args.epoch + 1):
        model.train()
        Tensor = torch.FloatTensor
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                              desc='Train epoch = {}'.format(epoch), ncols=80,
                                                              leave=False):

            sourceData, sourceLabel = sourceData.expand(len(sourceData), args.n_dim, imageSize, imageSize).to(
                DEVICE), sourceLabel.to(DEVICE)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Sample noise as generator input
            z = Tensor(np.random.normal(0, 1, (sourceData.shape[0], args.latent_dim))).to(DEVICE)

            # Measure discriminator's ability to classify real from generated samples
            fake_img = feature_extractor(z).detach()

            d_loss = -torch.mean(critic(sourceData)) + torch.mean(critic(fake_img))+args.lambda_gp*compute_gradient_penalty(critic,sourceData.data,fake_img.data)

            d_loss.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            num_parameters = 0
            num1 = 0
            num2 = 0
            for p in critic.parameters():
                num_parameters += (p.data.size(0) * p.data.size(-1))
                p.data.clamp_(-args.clip_value, args.clip_value)
                # print(num_parameters)
                num1 += float(p.data[p.data > (args.clip_value * 0.9)].size(0))
                num2 += float(p.data[p.data < (-args.clip_value * 0.9)].size(0))

            if batch_idx % args.n_critic == 0:
                print(num_parameters, '%.2f' % (num1 / num_parameters), '%.2f' % (num2 / num_parameters),
                      '%.2f' % ((num1 + num2) / num_parameters))
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()
                fake_img = feature_extractor(z)
                # Loss measures generator's ability to fool the discriminator
                g_loss = torch.mean(critic(sourceData))-torch.mean(critic(fake_img))

                g_loss.backward()
                optimizer_G.step()



                if batch_idx % args.logInterval == 0:
                    print(
                        '\ncritic_loss: {:.4f}, wd_Loss: {:.6f}'.format(
                            d_loss.item(), g_loss.item()))

            batches_done = (epoch-1) * len(sourceDataLoader) + batch_idx
            if batches_done % args.save_interval == 0:
                path="../images/"+args.model_name
                if not os.path.exists(path):
                    os.makedirs(path)
                save_image(fake_img.data[:25], path+"/%d.png" % batches_done, nrow=5, normalize=True)


    if args.ifsave:
        path=args.savePath+args.model_name
        if not os.path.exists(path):
            os.makedirs(path)

        if args.if_saveall:
            state = {
                'epoch': args.epoch,
                'net': model,
                'optimizer_G': optimizer_G,
                'optimizer_D': optimizer_D,

            }
        else:
            state = {
                'epoch': args.epoch,
                'net': model.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),

            }
        path+='/'+args.model_name+'_epoch'+str(args.epoch)+'.pth'
        torch.save(state, path)


class Generator(nn.Module):
    def __init__(self,args,img_size):
        super(Generator, self).__init__()
        self.args=args
        self.img_shape=(self.args.n_dim,img_size,img_size)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.args.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self,args,img_size):
        super(Discriminator, self).__init__()
        self.args=args
        self.img_shape = (self.args.n_dim, img_size, img_size)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class WGAN_GPModel(nn.Module):
    def __init__(self,args,img_size):
        super(WGAN_GPModel, self).__init__()
        self.args=args
        self.feature_extractor=Generator(args,img_size)
        self.critic=Discriminator(args,img_size)