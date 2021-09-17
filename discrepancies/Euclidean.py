# author:LENOVO
# contact: ACoderlyy@163.com
# datetime:2021/9/17 9:51
# software: PyCharm
import torch
def euclidean_dist(x, y, square=False):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m = x.size(0)
    n = y.size(0)

    # 方式1
    a1 = torch.sum((x ** 2), 1, keepdim=True).expand(m, n)
    b2 = (y ** 2).sum(1).expand(m, n)
    if square:
        dist = (a1 + b2 - 2 * (x @ y.T)).float()
    else:
        dist = (a1 + b2 - 2 * (x @ y.T)).float().sqrt()
    return dist
