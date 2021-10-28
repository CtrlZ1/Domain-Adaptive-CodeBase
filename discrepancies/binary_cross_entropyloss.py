# author:ACoderlyy
# contact: ACoderlyy@163.com
# datetime:2021/10/28 20:30
# software: PyCharm
import torch
def binary_cross_entropyloss(prob, target, weight=None):
    if not weight is None:
        loss = -weight * (target * torch.log(prob) + (1 - target) * (torch.log(1 - prob)))
    else:
        loss = -(target * torch.log(prob) + (1 - target) * (torch.log(1 - prob)))
    loss = torch.sum(loss) / torch.numel(target)
    return loss