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

class GroupWiseContrastiveLoss(nn.Module):
  def __init__(self, margin=0, measure=False, max_violation=False):
    super(GroupWiseContrastiveLoss, self).__init__()
    self.margin = margin
    if measure == 'order':
      NotImplemented
    else:
      self.sim = cosine_sim

    self.max_violation = max_violation

  def forward(self, im, s, num_clips, num_caps):
    # compute image-sentence score matrix
    scores = self.sim(im, s)

    # generate mask
    N_ = len(num_clips)
    scores_reduced = Variable(torch.zeros(N_, N_).cuda())
    assert N_ == len(num_caps)
    for i in range(N_):
      clip_start, clip_end = sum(num_clips[0:i]), sum(num_clips[0:i+1])
      for j in range(N_):
        cap_start, cap_end   = sum(num_caps[0:j]), sum(num_caps[0:j+1])
        if self.max_violation:
          scores_reduced[i, j] = scores[clip_start:clip_end, cap_start:cap_end].max()
        else:
          scores_reduced[i, j] = scores[clip_start:clip_end, cap_start:cap_end].mean()

    diagonal = scores_reduced.diag().view(N_, 1)
    d1 = diagonal.expand_as(scores_reduced)
    d2 = diagonal.t().expand_as(scores_reduced)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s  = (self.margin + scores_reduced - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (self.margin + scores_reduced - d2).clamp(min=0)

    mask = torch.eye(scores_reduced.size(0)) > .5
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

  def forward(self, clips, videos, caps, paragraphs, num_clips, num_caps):
      return self.forward_loss(clips, videos, num_clips) + self.forward_loss(caps, paragraphs, num_caps)

class ReconstructLoss(nn.Module):
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

    return cost_s.sum() + cost_im.sum() + (d1-1).abs().sum() + (d2-1).abs().sum()
    # return cost_s.sum()


