import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import accuracy

class proto(nn.Module):
    def __init__(self, **kwargs):
        super(proto, self).__init__()
        self.criterion  = torch.nn.CrossEntropyLoss()
        print('Initialised Prototypical Loss')

    def forward(self, x, label=None):
        assert x.size()[1] >= 2
        out_anchor = torch.mean(x[:,1:,:],1)
        out_positive = x[:,0,:]
        stepsize = out_anchor.size()[0]
        output = -1 * (F.pairwise_distance(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))**2)
        label = torch.from_numpy(np.asarray(range(0,stepsize))).cuda()
        loss = self.criterion(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1

