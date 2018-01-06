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

def cosine_sim(im, s):
  return im.mm(s.t())

class ContrastiveLoss_no_correspond(nn.Module):
  def __init__(self, margin=0, measure=False, max_violation=False):
    super(ContrastiveLoss_no_correspond, self).__init__()
    self.margin = margin
    if measure == 'order':
      NotImplemented
    else:
      self.sim = cosine_sim

    self.max_violation = max_violation

  def forward(self, im, s, seg_num):
    # compute image-sentence score matrix
    scores = self.sim(im, s)
    diagonal = scores.diag().view(im.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (self.margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (self.margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.zeros(scores.shape)
    for i in range(len(seg_num)):
        mask[sum(seg_num[0:i]):sum(seg_num[0:i+1]), sum(seg_num[0:i]):sum(seg_num[0:i+1])] = 1

    mask = mask > 0.5
    I = Variable(mask)
    if torch.cuda.is_available():
      I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    if self.max_violation:
      cost_s = cost_s.max(1)[0]
      cost_im = cost_im.max(0)[0]

    return cost_s.sum() + cost_im.sum()
    # return cost_s.sum()



class ContrastiveLoss(nn.Module):
  def __init__(self, margin=0, measure=False, max_violation=False):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin
    if measure == 'order':
      NotImplemented
    else:
      self.sim = cosine_sim

    self.max_violation = max_violation

  def forward(self, im, s):
    # compute image-sentence score matrix
    scores = self.sim(im, s)
    diagonal = scores.diag().view(im.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (self.margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (self.margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
      I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    if self.max_violation:
      cost_s = cost_s.max(1)[0]
      cost_im = cost_im.max(0)[0]

    return cost_s.sum() + cost_im.sum()
    # return cost_s.sum()


class CenterLoss(nn.Module):
  def __init__(self, margin=0, measure=False, max_violation=False, tune_center=False):
    super(CenterLoss, self).__init__()
    self.margin = margin
    self.sim = cosine_sim
    self.max_violation = max_violation
    self.tune_center=tune_center

  def forward_loss(self, im, vid, seg_num):
    # compute image-sentence score matrix
    if self.tune_center:
        pass
    else:
        vid = vid.detach()
    scores = self.sim(im, vid)

    middle_block = Variable(torch.zeros(scores.shape[0])).cuda()

    mask = torch.zeros(scores.shape)

    for i in range(len(seg_num)):
        cur_block = scores[sum(seg_num[0:i]):sum(seg_num[0:i+1]), i]
        middle_block[sum(seg_num[0:i]):sum(seg_num[0:i+1])] = cur_block
        mask[sum(seg_num[0:i]):sum(seg_num[0:i+1]), i] = 1
    middle_block_reshape = middle_block.view(middle_block.shape[0],1).expand_as(scores)


    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (self.margin + scores - middle_block_reshape).clamp(min=0)

    # clear diagonals
    mask = mask > .5
    I = Variable(mask)
    if torch.cuda.is_available():
      I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    if self.max_violation:
      cost_s = cost_s.max(1)[0]

    return cost_s.sum()

  def forward(self, im, vid, sent, para, seg_num):
      return self.forward_loss(im, vid, seg_num) + self.forward_loss(sent, para, seg_num)

