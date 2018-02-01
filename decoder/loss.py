import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from IPython import embed

class EuclideanLoss(nn.Module):
  def __init__(self):
    super(EuclideanLoss, self).__init__()

  def forward_loss(self, clip_remap, clip_emb):
    # compute image-sentence score matrix
#    embed()
    score = clip_remap - clip_emb

    score = torch.sqrt((score**2).sum(dim=1)).sum(0)

    return score

  def forward(self, clip_remap, clip_emb):
      return self.forward_loss(clip_remap, clip_emb)

