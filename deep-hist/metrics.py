import torch
from torch import nn

class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()
    
    def forward(self, pred, target):
        bins = pred.size(1)
        rang = torch.arange(bins) # [0, 1, 2, ..., bins-1]
        s, t = torch.meshgrid(rang, rang) # s, t: [bins, bins]
        
        target = t >= s 

        cdf_pred = torch.matmul(pred, target.float()) # [batch, bins]
        cdf_target = torch.matmul(target.float(), target.float()) # [batch, bins]

        loss = torch.sum(torch.square(cdf_pred - cdf_target), dim=1) # [batch]
        return loss

class MILoss(nn.Module):
    def __init__(self):
        super(MILoss, self).__init__()
    
    def forward(self, i1, i2, target):
        # prod_i is the product of the marginals
        prod_i = torch.matmul(torch.transpose(i1.unsqueeze(1), 1, 2), i2.unsqueeze(1)) + torch.finfo(i1.dtype).eps

        # mi is the mutual information
        mi = torch.sum(target * torch.log(target / prod_i + torch.finfo(i1.dtype).eps), dim=(1, 2))

        # h is the entropy
        h = -torch.sum(target * torch.log(target + torch.finfo(i1.dtype).eps), dim=(1, 2))

        loss = 1 - (mi / h)
        return loss
