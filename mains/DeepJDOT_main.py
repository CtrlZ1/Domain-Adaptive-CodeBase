# author:LENOVO
# contact: ACoderlyy@163.com
# datetime:2021/9/17 10:07
# software: PyCharm

import dataLoader
from models.DeepJDOT import train_process, test_process, DeepJDOTModel
from utils import datasetRootAndImageSize
import argparse
import torch


# parameter setting
parser=argparse.ArgumentParser()

# model parameter
parser.add_argument('--batchSize',type=int,default=500,metavar='batchSize',help='input the batch size of training process.(default=500)')
parser.add_argument('--sample_size',type=int,default=50,metavar='sample_size',help='the number of every class to chose in samples in every source for one batch.(default=50)')
parser.add_argument('--epoch',type=int, default=1000,metavar='epoch',help='the number of epochs for training.(default=100)')
parser.add_argument('--lr',type=float,default=1e-5,metavar='LR',help='the learning rate for training.(default=1e-5)')
parser.add_argument('--n_labels',type=int,default=10,help='the number of sources and target labels')
parser.add_argument('--n_dim',type=int,default=1,help='the channels of images.(default=1)')
parser.add_argument('--linear', default=False, action='store_true',help='whether not use the linear kernel')
parser.add_argument('--bottleneck_dim',type=int,default=256,help='the Number of neurons in bottleneck.(default=256)')


# hyperparameter
parser.add_argument('--lrf',type=float,default=0.0002,help='the lr of feature_extractor')
parser.add_argument('--lrc',type=float,default=0.0002,help='the lr of classifier')
parser.add_argument('--alpha',type=float,default=0.001,help='the Hyperparameter to weight C0 and C1.(default=0.001)')
parser.add_argument('--alpha2',type=float,default=0.0001,help='the Hyperparameter to weight C0 and C1.(default=0.0001)')
parser.add_argument('--train_par',type=float,default=1.0,help='the Hyperparameter to weight loss_ot and loss_clf.(default=1.0)')
parser.add_argument('--lam',type=float,default=1.0,help='the Hyperparameter to weight loss_ot and loss_clf.(default=1.0)')
parser.add_argument('--momentum',type=float,default=0.9,metavar='M',help='SGD momentum.(default=0.9)')
parser.add_argument('--l2_Decay',type=float,default=5e-4,help='the L2 weight decay.(default=5e-4')
parser.add_argument('--lamb',type=float,default=0.25,help='the Hyperparameter to weight the mmd_loss loss in total loss.(default=0.25)')
parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler.(default=3e-4)')
parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler.(default=0.75)')


# setting parameter
parser.add_argument('--noCuda',action='store_true',default=False,help='use cude in training process or do not.(default=Flase)')
parser.add_argument('--seed',type=int,default=1999,metavar='S',help='random seed.(default:1)')
parser.add_argument('--logInterval', type=int,default=10,metavar='log',help='the interval to log for one time.(default=10)')
parser.add_argument('--gpu', default=0, type=int,help='the index of GPU to use.(default=0)')
parser.add_argument('--model_name',type=str,default='DeepJDOT',help='the model')
parser.add_argument('--savePath',type=str,default='../checkpoints/',help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifsave',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--if_saveall',default=False,type=bool,help='if save all or just state.(default=False)')
parser.add_argument('--loadPath',default='../checkpoints/',type=str,help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifload',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--usecheckpoints', default=False, type=bool,help='use the pretrained model.(default=False)')
parser.add_argument('--datasetIndex',type=int,default=4,help='chose the index of dataset by it')

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


    sourceTrainLoader, sourceTestLoader,targetTrainLoader,targetTestLoader = \
        dataLoader.singleSourceDataLoader(args.batchSize,datasetRootAndImageSize[args.datasetIndex][0],
                                          datasetRootAndImageSize[args.datasetIndex][1],kwargs)


    mcdamodel=DeepJDOTModel(args,DEVICE).to(DEVICE)

    # use usps->mnist as demo
    # source= dataLoader.loadDigitsDataset(datasetRootAndImageSize[args.datasetIndex][0], datasetRootAndImageSize[args.datasetIndex][2], 'usps', 'train')
    # target= dataLoader.loadDigitsDataset(datasetRootAndImageSize[args.datasetIndex][1], datasetRootAndImageSize[args.datasetIndex][2], 'mnist', 'train')
    source=sourceTrainLoader.dataset
    target=targetTrainLoader.dataset
    train_process(mcdamodel,  source, target, sourceTrainLoader, targetTrainLoader,sourceTestLoader,targetTestLoader,DEVICE,datasetRootAndImageSize[args.datasetIndex][2],args,method='emd',
                          metric='deep', reg_sink=1)

    test_process(mcdamodel, sourceTestLoader,targetTestLoader,DEVICE,args)