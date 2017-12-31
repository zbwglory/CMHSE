from __future__ import print_function
import os
import pickle

import numpy
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary  # NOQA
import torch
from model import VSE
from collections import OrderedDict
from IPython import embed


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
    for i, (k, v) in enumerate(self.meters.iteritems()):
      if i > 0:
        s += '  '
      s += k + ' ' + str(v)
    return s

  def tb_log(self, tb_logger, prefix='', step=None):
    """Log using tensorboard
    """
    for k, v in self.meters.iteritems():
      tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
  """Encode all images and captions loadable by `data_loader`
  """
  batch_time = AverageMeter()
  val_logger = LogCollector()

  # switch to evaluate mode
  model.val_start()

  end = time.time()

  # numpy array to keep all the embeddings
  img_embs, cap_embs = [], []
  img_seq_embs, cap_seq_embs = [], []
  img_whole_embs, cap_whole_embs = [], []
  for i, (images, captions, video_whole, captions_whole, lengths_img, lengths_cap, lengths_whole_vid, lengths_whole_cap, ind, seg_num) in enumerate(data_loader):
    # make sure val logger is used
    model.logger = val_logger

    # compute the embeddings
    img_seq_emb, cap_seq_emb, img_emb, cap_emb, img_whole_emb, cap_whole_emb = model.structure_emb(images, captions, video_whole, captions_whole, lengths_img, lengths_cap, lengths_whole_vid, lengths_whole_cap, ind, seg_num)

    # initialize the numpy arrays given the size of the embeddings
    img_embs.append(img_emb.data.cpu())
    cap_embs.append(cap_emb.data.cpu())
    img_seq_embs.append(img_seq_emb.data.cpu())
    cap_seq_embs.append(cap_seq_emb.data.cpu())
    img_whole_embs.append(img_whole_emb.data.cpu())
    cap_whole_embs.append(cap_whole_embs.data.cpu())

    # measure accuracy and record loss
    model.forward_loss(img_emb, cap_emb, 'test')

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
    del images, captions

  img_seq_embs = torch.cat(img_seq_embs, 0)
  cap_seq_embs = torch.cat(cap_seq_embs, 0)
  img_seq_embs = img_seq_embs.numpy()
  cap_seq_embs = cap_seq_embs.numpy()

  img_whole_embs = torch.cat(img_whole_embs, 0)
  cap_whole_embs = torch.cat(cap_whole_embs, 0)
  img_whole_embs = img_whole_embs.numpy()
  cap_whole_embs = cap_whole_embs.numpy()

  img_embs = torch.cat(img_embs, 0)
  cap_embs = torch.cat(cap_embs, 0)
  img_embs = img_embs.numpy()
  cap_embs = cap_embs.numpy()
  return img_seq_embs, cap_seq_embs, img_embs, cap_embs, img_whole_embs, cap_whole_embs

def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
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
  if return_ranks:
    return (r1, r5, r10, medr, meanr), (ranks, top1)
  else:
    return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
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
  if return_ranks:
    return (r1, r5, r10, medr, meanr), (ranks, top1)
  else:
    return (r1, r5, r10, medr, meanr)

def encode_eval_data(model, data_loader, log_step=10, logging=print, num_offsets=3):
  """Encode all images and captions loadable by `data_loader`
  """
  batch_time = AverageMeter()
  val_logger = LogCollector()

  # switch to evaluate mode
  model.val_start()

  end = time.time()

  # numpy array to keep all the embeddings
  img_seq_embs = [ [] for _ in xrange(num_offsets)]
  cap_seq_embs = [ [] for _ in xrange(num_offsets)]
  for i, (images, captions, _, _, lengths_img, lengths_cap, _, _, ind, seg_nums) in enumerate(data_loader):
    # make sure val logger is used
    model.logger = val_logger

    # compute the embeddings
    for _offset in xrange(num_offsets):
      img_seq_emb, cap_seq_emb = model.test_emb(images, captions, lengths_img, lengths_cap, ind, seg_nums, offset=_offset)

      img_seq_embs[_offset].append(img_seq_emb.data.cpu())
      cap_seq_embs[_offset].append(cap_seq_emb.data.cpu())

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % log_step == 0:
      logging('Test: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          .format(
            i, len(data_loader), batch_time=batch_time))
    del images, captions

  img_seq_embs = [ torch.cat(img_seq_embs[_offset], 0).numpy() for _offset in xrange(num_offsets) ]
  cap_seq_embs = [ torch.cat(cap_seq_embs[_offset], 0).numpy() for _offset in xrange(num_offsets) ]

  return img_seq_embs, cap_seq_embs
