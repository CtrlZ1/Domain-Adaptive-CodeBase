# author:ACoderlyy
# contact: ACoderlyy@163.com
# datetime:2021/9/18 10:36
# software: PyCharm

import dataLoader
import torch
from models.JAN import train_process, test_process, JANModel
from utils import model_feature_tSNE,datasetRootAndImageSize
import argparse
import matplotlib.pyplot as plt
import os
import torch
# parameter setting
parser=argparse.ArgumentParser()

# model parameter
parser.add_argument('--batchSize',type=int,default=32,metavar='batchSize',help='input the batch size of training process.(default=64)')
parser.add_argument('--epoch',type=int, default=1000,metavar='epoch',help='the number of epochs for training.(default=100)')
parser.add_argument('--lr',type=float,default=1e-3,metavar='LR',help='the learning rate for training.(default=1e-2)')
parser.add_argument('--n_labels',type=int,default=31,help='the number of sources and target labels')
parser.add_argument('--n_dim',type=int,default=3,help='the channels of images.(default=1)')
parser.add_argument('--linear', default=False, action='store_true',help='whether not use the linear kernel')
parser.add_argument('--bottleneck_dim',type=int,default=256,help='the Number of neurons in bottleneck.(default=256)')
parser.add_argument('--adversarial', default=False, action='store_true',help='whether use adversarial theta')
# hyperparameter
parser.add_argument('--momentum',type=float,default=0.9,metavar='M',help='SGD momentum.(default=0.9)')
parser.add_argument('--l2_Decay',type=float,default=5e-4,help='the L2 weight decay.(default=5e-4')
parser.add_argument('--lamb',type=float,default=0.25,help='the Hyperparameter to weight the mmd_loss loss in total loss.(default=0.25)')
parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler.(default=3e-4)')
parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler.(default=0.75)')
# setting parameter
parser.add_argument('--backbone_name',type=str,default='ResNet50')
parser.add_argument('--noCuda',action='store_true',default=False,help='use cude in training process or do not.(default=Flase)')
parser.add_argument('--seed',type=int,default=1999,metavar='S',help='random seed.(default:1)')
parser.add_argument('--logInterval', type=int,default=10,metavar='log',help='the interval to log for one time.(default=10)')
parser.add_argument('--gpu', default=0, type=int,help='the index of GPU to use.(default=0)')
parser.add_argument('--model_name',type=str,default='MCDA',help='the model')
parser.add_argument('--savePath',type=str,default='../checkpoints/',help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifsave',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--if_saveall',default=False,type=bool,help='if save all or just state.(default=False)')
parser.add_argument('--loadPath',default='../checkpoints/',type=str,help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifload',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--usecheckpoints', default=False, type=bool,help='use the pretrained model.(default=False)')
parser.add_argument('--datasetIndex',type=int,default=5,help='chose the index of dataset by it')

args=parser.parse_args()
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


    sourceTrainLoader, targetTrainLoader = dataLoader.loadTrainData(datasetRootAndImageSize[args.datasetIndex], args.batchSize,
                                                                   args.datasetIndex, datasetRootAndImageSize[args.datasetIndex][2],
                                                                   kwargs)
    sourceTestLoader,targetTestLoader = dataLoader.loadTestData(datasetRootAndImageSize[args.datasetIndex], args.batchSize, args.datasetIndex,
                                                                datasetRootAndImageSize[args.datasetIndex][2], kwargs)

    mcdamodel=JANModel(DEVICE,args).to(DEVICE)

    train_process(mcdamodel, sourceTrainLoader, targetTrainLoader,sourceTestLoader,targetTestLoader,DEVICE,datasetRootAndImageSize[args.datasetIndex][2],args)

    test_process(mcdamodel, sourceTestLoader,targetTestLoader,DEVICE,args)