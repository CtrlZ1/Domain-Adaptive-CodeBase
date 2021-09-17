# author:LENOVO
# contact: ACoderlyy@163.com
# datetime:2021/9/13 9:44
# software: PyCharm

import torch
from models.WGAN import train_process, WGANModel
import dataLoader
import argparse
from utils import datasetRootAndImageSize
# parameter setting
parser=argparse.ArgumentParser()

# model parameter
parser.add_argument('--batchSize',type=int,default=8,metavar='batchSize',help='input the batch size of training process.(default=8)')
parser.add_argument('--epoch',type=int, default=1000,metavar='epoch',help='the number of epochs for training.(default=1000)')
parser.add_argument('--lr',type=float,default=0.00005,metavar='LR',help='the learning rate for training.(default=5e-5)')
parser.add_argument('--latent_dim',type=int,default=100,help='the Number of neurons in feature extractor.(default=100)')
parser.add_argument('--n_critic',type=int,default=5,help='the number of training critic before training others one time.(default=5)')
parser.add_argument('--n_dim',type=int,default=1,help='the channels of images.(default=1)')

# hyperparameter
parser.add_argument('--clip_value',type=float,default=1e-2,help='the hyperparameter to clip the grad')
# setting parameter
parser.add_argument('--noCuda',action='store_true',default=False,help='use cude in training process or do not.(default=Flase)')
parser.add_argument('--seed',type=int,default=1999,metavar='S',help='random seed.(default:1)')
parser.add_argument('--logInterval', type=int,default=50,metavar='log',help='the interval to log for one time.(default=10)')
parser.add_argument('--save_interval', type=int,default=500,metavar='log',help='the interval to save images.(default=10)')
parser.add_argument('--gpu', default=0, type=int,help='the index of GPU to use.(default=0)')
parser.add_argument('--model_name',type=str,default='WGAN',help='the model')
parser.add_argument('--savePath',type=str,default='../checkpoints/',help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifsave',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--if_saveall',default=False,type=bool,help='if save all or just state.(default=False)')
parser.add_argument('--loadPath',default='../checkpoints/',type=str,help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifload',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--usecheckpoints', default=False, type=bool,help='use the pretrained model.(default=False)')
parser.add_argument('--datasetIndex',type=int,default=4,help='chose the index of dataset by it')

args=parser.parse_args()




DEVICE=torch.device('cpu')
kwargs={}


if not args.noCuda and torch.cuda.is_available():
    DEVICE=torch.device('cuda:'+str(args.gpu))
    torch.cuda.manual_seed(args.seed)
    kwargs={'num_workers': 0, 'pin_memory': True}

print(DEVICE,torch.cuda.is_available())




if __name__ == '__main__':

    sourceTrainLoader, targetTrainLoader = dataLoader.loadTrainData(datasetRootAndImageSize[args.datasetIndex], args.batchSize,
                                                                   args.datasetIndex, datasetRootAndImageSize[args.datasetIndex][2],
                                                                   kwargs)

    mcdamodel=WGANModel(args,datasetRootAndImageSize[args.datasetIndex][2]).to(DEVICE)

    train_process(mcdamodel, sourceTrainLoader,DEVICE,datasetRootAndImageSize[args.datasetIndex][2],args)

