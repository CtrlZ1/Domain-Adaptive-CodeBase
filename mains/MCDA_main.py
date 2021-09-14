# author:LENOVO
# contact: ACoderlyy@163.com
# datetime:2021/9/11 10:00
# software: PyCharm


import torch
from models.MCDA import train_process, test_process, MCDAModel
import dataLoader
from utils import model_feature_tSNE,datasetRootAndImageSize
import argparse
import matplotlib.pyplot as plt
import os

# parameter setting
parser=argparse.ArgumentParser()

# model parameter
parser.add_argument('--batchSize',type=int,default=128,metavar='batchSize',help='input the batch size of training process.(default=64)')
parser.add_argument('--epoch',type=int, default=1000,metavar='epoch',help='the number of epochs for training.(default=100)')
parser.add_argument('--lr',type=float,default=1e-3,metavar='LR',help='the learning rate for training.(default=1e-2)')
parser.add_argument('--critic_dim',type=int,default=100,help='the Number of neurons in critic.(default=100)')
parser.add_argument('--n_critic',type=int,default=5,help='the number of training critic before training others one time.(default=5)')
parser.add_argument('--n_clf',type=int,default=1,help='the number of training classifier after training critic one time.(default=1)')
parser.add_argument('--n_labels',type=int,default=10,help='the number of sources and target labels')
parser.add_argument('--n_dim',type=int,default=1,help='the channels of images.(default=1)')

# hyperparameter
parser.add_argument('--alpha',type=float,default=1e-4)
parser.add_argument('--momentum',type=float,default=0.9,metavar='M',help='SGD momentum.(default=0.9)')
parser.add_argument('--lambda_wd_clf',type=float,default=1.0,help='the Hyperparameter to weight the Wasserstein loss and classifier loss in total loss.(default=1.0)')
parser.add_argument('--theta', default=0.9, type=float,help='the threshold of softmax to use target data to train.(default=0.9)')
parser.add_argument('--lambda_gp',type=float,default=10.0,help='the Hyperparameter to weight the Gradient Penalty loss in total loss.(default=10.0)')
parser.add_argument('--l2Decay',type=float,default=5e-4,help='the L2 weight decay.(default=5e-4')

# setting parameter
parser.add_argument('--noCuda',action='store_true',default=False,help='use cude in training process or do not.(default=Flase)')
parser.add_argument('--seed',type=int,default=1999,metavar='S',help='random seed.(default:1)')
parser.add_argument('--logInterval', type=int,default=50,metavar='log',help='the interval to log for one time.(default=10)')
parser.add_argument('--gpu', default=0, type=int,help='the index of GPU to use.(default=0)')
parser.add_argument('--model_name',type=str,default='MCDA',help='the model')
parser.add_argument('--savePath',type=str,default='../checkpoints/',help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifsave',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--if_saveall',default=False,type=bool,help='if save all or just state.(default=False)')
parser.add_argument('--loadPath',default='../checkpoints/',type=str,help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifload',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--usecheckpoints', default=False, type=bool,help='use the pretrained model.(default=False)')
parser.add_argument('--datasetIndex',type=int,default=4,help='chose the index of dataset by it')



args=parser.parse_args()
imageSize=[224,32,28]
datasetRoot=[
    # 源域，目标域 office-31目录 a-w
    [r"E:\transferlearning\data\office-31\Original_images\amazon",r"E:\transferlearning\data\office-31\Original_images\webcam"],
    # svhn->mnist
    [r'E:\transferlearning\data\SVHN',r'E:\transferlearning\data\MNIST'],
    #mnist-mnist-m
    [r'E:\transferlearning\data\MNIST',r'E:\transferlearning\data\MNIST-M\mnist_m'],
    #ImageCLEF 2014
    [r'E:\transferlearning\data\ImageCLEF 2014\b',r'E:\transferlearning\data\ImageCLEF 2014\c'],
    # usps-mnist
    [r'E:\transferlearning\data\usps',r'E:\transferlearning\data\MNIST'],
    # 源域，目标域 office_caltech_10目录 a-w
    [r"E:\transferlearning\data\office_caltech_10\amazon",r"E:\transferlearning\data\office_caltech_10\webcam"],
    # mnist-usps
    [r'E:\李沂洋毕设\data\MNIST', r'E:\李沂洋毕设\data\usps'],

]
# 默认CPU
DEVICE=torch.device('cpu')
kwargs={}

# 如果Cuda可用就用Cuda
if not args.noCuda and torch.cuda.is_available():
    DEVICE=torch.device('cuda:'+str(args.gpu))
    torch.cuda.manual_seed(args.seed)
    kwargs={'num_workers': 0, 'pin_memory': True}

print(DEVICE,torch.cuda.is_available())




if __name__ == '__main__':

    # 准备数据
    sourceTrainLoader, targetTrainLoader = dataLoader.loadTrainData(datasetRoot[args.datasetIndex], args.batchSize,
                                                                   args.datasetIndex, datasetRootAndImageSize[args.datasetIndex][2],
                                                                   kwargs)
    sourceTestLoader,targetTestLoader = dataLoader.loadTestData(datasetRoot[args.datasetIndex], args.batchSize, args.datasetIndex,
                                                                datasetRootAndImageSize[args.datasetIndex][2], kwargs)

    mcdamodel=MCDAModel(args).to(DEVICE)

    train_process(mcdamodel, sourceTrainLoader, targetTrainLoader,sourceTestLoader,targetTestLoader,DEVICE,datasetRootAndImageSize[args.datasetIndex][2],args)

    test_process(mcdamodel, sourceTestLoader,targetTestLoader,DEVICE,args)

