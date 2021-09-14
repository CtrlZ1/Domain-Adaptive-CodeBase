# author:LENOVO
# contact: ACoderlyy@163.com
# datetime:2021/9/12 9:24
# software: PyCharm

import torch
from models.WDGRL import train_process, test_process, WDGRLModel
import dataLoader
from utils import model_feature_tSNE,datasetRootAndImageSize
import argparse



# parameter setting
parser=argparse.ArgumentParser()
# model parameter
parser.add_argument('--batchSize',type=int,default=128,help='input the batch size of training process.(default=128)')
parser.add_argument('--epoch',type=int, default=1000,help='the number of epochs for training.(default=1000)')
parser.add_argument('--lr',type=float,default=1e-3,metavar='LR',help='the learning rate for training.(default=5e-3)')
parser.add_argument('--n_labels',type=int,default=10,help='the number of sources and target labels')
parser.add_argument('--n_critic',type=int,default=5,help='the number of training critic before training others one time.(default=5)')
parser.add_argument('--n_clf',type=int,default=1,help='the number of training classifier after training critic one time.(default=1)')
parser.add_argument('--n_dim',type=int,default=3,help='the channels of images.(default=3)')

# hyperparameter
parser.add_argument('--lambda_wd_clf',type=float,default=1.0,help='the Hyperparameter to weight the Wasserstein loss and classifier loss in total loss.(default=1.0)')
parser.add_argument('--lambda_gp',type=float,default=10.0,help='the Hyperparameter to weight the Gradient Penalty loss in total loss.(default=10.0)')
parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler.(default=3e-4)')
parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler.(default=0.75)')
parser.add_argument('--l2_Decay',type=float,default=5e-4,help='parameter for SGD.(default=5e-4)')
parser.add_argument('--momentum',type=float,default=0.9,metavar='M',help='SGD momentum.(default=0.5)')

# setting parameter
parser.add_argument('--noCuda',action='store_true',default=False,help='use cude in training process or do not.(default=Flase)')
parser.add_argument('--seed',type=int,default=1999,metavar='S',help='random seed.(default:1)')
parser.add_argument('--logInterval', type=int,default=50,metavar='log',help='the interval to log for one time.(default=10)')
parser.add_argument('--gpu', default=0, type=int,help='the index of GPU to use.(default=0)')
parser.add_argument('--model_name',type=str,default='WDGRL',help='the model')
parser.add_argument('--savePath',type=str,default='../checkpoints/',help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifsave',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--if_saveall',default=False,type=bool,help='if save all or just state.(default=False)')
parser.add_argument('--loadPath',default='../checkpoints/',type=str,help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifload',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--usecheckpoints', default=False, type=bool,help='use the pretrained model.(default=False)')
parser.add_argument('--datasetIndex',type=int,default=1,help='chose the index of dataset by it.(default=5)')


args=parser.parse_args()


DEVICE=torch.device('cpu')
kwargs={}

# if use cuda
if not args.noCuda and torch.cuda.is_available():
    DEVICE=torch.device('cuda:'+str(args.gpu))
    torch.cuda.manual_seed(args.seed)
    kwargs={'num_workers': 0, 'pin_memory': True}

print(DEVICE,torch.cuda.is_available())




if __name__ == '__main__':

    sourceTrainLoader, targetTrainLoader = dataLoader.loadTrainData(datasetRootAndImageSize[args.datasetIndex], args.batchSize,
                                                                   args.datasetIndex, datasetRootAndImageSize[args.datasetIndex][2],kwargs)
    sourceTestLoader, targetTestLoader = dataLoader.loadTestData(datasetRootAndImageSize[args.datasetIndex], args.batchSize, args.datasetIndex,
                                               datasetRootAndImageSize[args.datasetIndex][2], kwargs)
    model = WDGRLModel(args).to(DEVICE)
    train_process(model,sourceTrainLoader, targetTrainLoader,sourceTestLoader,targetTestLoader ,DEVICE,datasetRootAndImageSize[args.datasetIndex][2],args)
    test_process(model,sourceTestLoader,targetTestLoader,DEVICE,args)
    model_feature_tSNE(model,sourceTrainLoader,targetTestLoader,'last',DEVICE,args.model_name)

