from __future__ import print_function
import os
import pickle

import numpy
from activity_net.data import get_test_loader
import time
import numpy as np
from anet_vocab import Vocabulary  # NOQA
import torch
from model import VSE
from collections import OrderedDict
from IPython import embed
import torch.nn.functional as F



class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=0):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / (.0001 + self.count)

  def __str__(self):
    """String representation for logging
    """
    # for values that should be recorded exactly e.g. iteration number
    if self.count == 0:
      return str(self.val)
    # for stats
    return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
  """A collection of logging objects that can change from train to val"""

  def __init__(self):
    # to keep the order of logged variables deterministic
    self.meters = OrderedDict()

  def update(self, k, v, n=0):
    # create a new meter if previously not recorded
    if k not in self.meters:
      self.meters[k] = AverageMeter()
    self.meters[k].update(v, n)

  def __str__(self):
    """Concatenate the meters in one log line
    """
    s = ''
    for i, (k, v) in enumerate(self.meters.items()):
      if i > 0:
        s += '  '
      s += k + ' ' + str(v)
    return s

  def tb_log(self, tb_logger, prefix='', step=None):
    """Log using tensorboard
    """
    for k, v in self.meters.items():
      tb_logger.log_value(prefix + k, v.val, step=step)

def LogReporter(tb_logger, result, epoch, name):
    for key in result:
        tb_logger.log_value(name+key, result[key], step=epoch)
    return

def encode_data(opt, model, data_loader, log_step=10, logging=print, contextual_model=True):
  """Encode all images and captions loadable by `data_loader`
  """
  batch_time = AverageMeter()
  val_logger = LogCollector()

  # switch to evaluate mode
  model.val_start(opt)

  end = time.time()

  # numpy array to keep all the embeddings
  clip_embs, cap_embs = [], []
  vid_embs, para_embs = [], []
  vid_contexts, para_contexts = [], []
  num_clips_total = []
  cur_vid_total = []
  for i, (clips, captions, videos, paragraphs, lengths_clip, lengths_cap, lengths_video, lengths_paragraph, num_clips, num_caps, ind, cur_vid) in enumerate(data_loader):
    # make sure val logger is used
    model.logger = val_logger
    num_clips_total.extend(num_clips)

    # compute the embeddings
    clip_emb, cap_emb = model.forward_emb(clips, captions, lengths_clip, lengths_cap)
    vid_context, para_context = model.forward_emb(videos, paragraphs, lengths_video, lengths_paragraph)
    if contextual_model:
      vid_emb, para_emb = model.structure_emb(clip_emb, cap_emb, num_clips, num_caps, vid_context, para_context)
    else:
      vid_emb, para_emb = model.structure_emb(clip_emb, cap_emb, num_clips, num_caps)


    clip_emb = F.normalize(clip_emb)
    cap_emb = F.normalize(cap_emb)
    vid_emb = F.normalize(vid_emb)
    para_emb = F.normalize(para_emb)
    vid_context = F.normalize(vid_context)
    para_context = F.normalize(para_context)


    # initialize the numpy arrays given the size of the embeddings
    clip_embs.extend(clip_emb.data.cpu())
    cap_embs.extend(cap_emb.data.cpu())
    vid_embs.extend(vid_emb.data.cpu())
    para_embs.extend(para_emb.data.cpu())
    vid_contexts.extend(vid_context.data.cpu())
    para_contexts.extend(para_context.data.cpu())
    cur_vid_total.extend(cur_vid)

    # measure accuracy and record loss
    model.forward_loss(vid_emb, para_emb, 'test')

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % log_step == 0:
      logging('Test: [{0}/{1}]\t'
          '{e_log}\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          .format(
            i, len(data_loader), batch_time=batch_time,
            e_log=str(model.logger)))

  vid_embs  = torch.stack(vid_embs, 0)
  para_embs = torch.stack(para_embs, 0)
  vid_embs  = vid_embs.numpy()
  para_embs = para_embs.numpy()

  clip_embs = torch.stack(clip_embs, 0)
  cap_embs = torch.stack(cap_embs, 0)
  clip_embs = clip_embs.numpy()
  cap_embs = cap_embs.numpy()

  vid_contexts  = torch.stack(vid_contexts, 0)
  para_contexts = torch.stack(para_contexts, 0)
  vid_contexts  = vid_contexts.numpy()
  para_contexts = para_contexts.numpy()

  return vid_embs, para_embs, clip_embs, cap_embs, vid_contexts, para_contexts, num_clips_total, cur_vid_total

def i2t(images, captions, npts=None, measure='cosine'):
  npts = images.shape[0]
  ranks = numpy.zeros(npts)
  top1 = numpy.zeros(npts)
  d = numpy.dot(images, captions.T)

  for index in range(npts):
    inds = numpy.argsort(d[index])[::-1]

    rank = numpy.where(inds == index)[0][0]
    ranks[index] = rank
    top1[index] = inds[0]

  r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
  r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
  r10 = 100.0 * len(numpy.where(ranks < 50)[0]) / len(ranks)
  medr = numpy.floor(numpy.median(ranks)) + 1
  meanr = ranks.mean() + 1
  report_dict = dict()
  report_dict['r1'] = r1
  report_dict['r5'] = r5
  report_dict['r10'] = r10
  report_dict['medr'] = medr
  report_dict['meanr'] = meanr
  report_dict['sum'] = r1+r5+r10
  return report_dict, top1, ranks


def t2i(images, captions, npts=None, measure='cosine'):
  npts = captions.shape[0]
  ranks = numpy.zeros(npts)
  top1 = numpy.zeros(npts)
  d = numpy.dot(captions, images.T)

  for index in range(npts):
    inds = numpy.argsort(d[index])[::-1]

    rank = numpy.where(inds == index)[0][0]
    ranks[index] = rank
    top1[index] = inds[0]

  r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
  r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
  r10 = 100.0 * len(numpy.where(ranks < 50)[0]) / len(ranks)
  medr = numpy.floor(numpy.median(ranks)) + 1
  meanr = ranks.mean() + 1
  report_dict = dict()
  report_dict['r1'] = r1
  report_dict['r5'] = r5
  report_dict['r10'] = r10
  report_dict['medr'] = medr
  report_dict['meanr'] = meanr
  report_dict['sum'] = r1+r5+r10
  return report_dict, top1, ranks
