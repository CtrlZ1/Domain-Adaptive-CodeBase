import torch.nn as nn
from torchvision import models
import torch
class LeNet(nn.Module):
    def __init__(self,n_dim):
        super(LeNet,self).__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv1',nn.Conv2d(n_dim,6,5))
        layer1.add_module('pool1',nn.MaxPool2d(2,2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2',nn.Conv2d(6,16,5))
        layer2.add_module('pool2',nn.MaxPool2d(2,2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1',nn.Linear(256,120))
        layer3.add_module('fc2',nn.Linear(120,84))
        layer3.add_module('fc3',nn.Linear(84,10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0),-1)
        # x = self.layer3(x)
        return x

class AlexNetFc(nn.Module):
    def __init__(self):
        super(AlexNetFc,self).__init__()
        self.modelAlexNet=models.alexnet(pretrained=True)
        # self.features=nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
        # )
        # for i in range(len(self.modelAlexNet.features)):
        #     if i!=0:
        #         self.features.add_module("features"+str(i),self.modelAlexNet.features[i])
        self.features =self.modelAlexNet.features
        self.classifier=self.modelAlexNet.classifier
        self.avgpool = self.modelAlexNet.avgpool
        # self.classifier=nn.Sequential()
        # for i in range(6):
        #     self.classifier.add_module("classifier"+str(i),modelAlexNet.classifier[i])
    def forward(self,x):
        x=self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x=self.classifier(x)
        return x


class AlexNetFc_for_layerWiseAdaptation(nn.Module):
    def __init__(self):
        super(AlexNetFc_for_layerWiseAdaptation,self).__init__()
        self.modelAlexNet=models.alexnet(pretrained=True)
        # self.features=nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
        # )
        # for i in range(len(self.modelAlexNet.features)):
        #     if i!=0:
        #         self.features.add_module("features"+str(i),self.modelAlexNet.features[i])
        self.features =self.modelAlexNet.features
        self.classifier=self.modelAlexNet.classifier
        self.avgpool = self.modelAlexNet.avgpool
        # self.classifier=nn.Sequential()
        # for i in range(6):
        #     self.classifier.add_module("classifier"+str(i),modelAlexNet.classifier[i])
    def forward(self,x):
        x=self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        dropout1=self.classifier[0](x)
        fc6=self.classifier[1](dropout1)
        relu1=self.classifier[2](fc6)
        dropout2=self.classifier[3](relu1)
        fc7 = self.classifier[4](dropout2)
        relu2 = self.classifier[5](fc7)
        fc8 = self.classifier[6](relu2)
        #x=self.classifier(x)
        return fc6,fc7,fc8

class ResNet18Fc(nn.Module):
    def __init__(self):
        super(ResNet18Fc, self).__init__()
        model_resnet18 = models.resnet18(pretrained=True)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResNet34Fc(nn.Module):
    def __init__(self):
        super(ResNet34Fc, self).__init__()
        model_resnet34 = models.resnet34(pretrained=True)
        self.conv1 = model_resnet34.conv1
        self.bn1 = model_resnet34.bn1
        self.relu = model_resnet34.relu
        self.maxpool = model_resnet34.maxpool
        self.layer1 = model_resnet34.layer1
        self.layer2 = model_resnet34.layer2
        self.layer3 = model_resnet34.layer3
        self.layer4 = model_resnet34.layer4
        self.avgpool = model_resnet34.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResNet101Fc(nn.Module):
    def __init__(self):
        super(ResNet101Fc, self).__init__()
        model_resnet101 = models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResNet152Fc(nn.Module):
    def __init__(self):
        super(ResNet152Fc, self).__init__()
        model_resnet152 = models.resnet152(pretrained=True)
        self.conv1 = model_resnet152.conv1
        self.bn1 = model_resnet152.bn1
        self.relu = model_resnet152.relu
        self.maxpool = model_resnet152.maxpool
        self.layer1 = model_resnet152.layer1
        self.layer2 = model_resnet152.layer2
        self.layer3 = model_resnet152.layer3
        self.layer4 = model_resnet152.layer4
        self.avgpool = model_resnet152.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


network_dict = {
    "LeNet":LeNet,
    "AlexNet": AlexNetFc,
    "AlexNetFc_for_layerWiseAdaptation": AlexNetFc_for_layerWiseAdaptation,
    "ResNet18": ResNet18Fc,
    "ResNet34": ResNet34Fc,
    "ResNet50": ResNet50Fc,
    "ResNet101": ResNet101Fc,
    "ResNet152": ResNet152Fc}