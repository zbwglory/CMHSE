import pickle
import os
import time
import shutil

import torch

import activity_net.data as data
# import didemo.data as data
from anet_vocab import Vocabulary
from model import VSE
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, LogReporter, t2v, i2p

import logging
import tensorboard_logger as tb_logger

import argparse

from IPython import embed

torch.cuda.manual_seed_all(1)
torch.manual_seed(1)

def main():
  # Hyper Parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', default='/data2/bwzhang/anet_img/captions/',
            help='path to datasets')
  parser.add_argument('--data_name', default='anet_precomp',
            help='anet_precomp')
  parser.add_argument('--vocab_path', default='./vocab/',
            help='Path to saved vocabulary pickle files.')
  parser.add_argument('--margin', default=0.2, type=float,
            help='Rank loss margin.')
  parser.add_argument('--num_epochs', default=50, type=int,
            help='Number of training epochs.')
  parser.add_argument('--batch_size', default=64, type=int,
            help='Size of a training mini-batch.')
  parser.add_argument('--word_dim', default=300, type=int,
            help='Dimensionality of the word embedding.')
  parser.add_argument('--embed_size', default=1024, type=int,
            help='Dimensionality of the joint embedding.')
  parser.add_argument('--grad_clip', default=0., type=float,
            help='Gradient clipping threshold.')
  parser.add_argument('--num_layers', default=1, type=int,
            help='Number of GRU layers.')
  parser.add_argument('--learning_rate', default=.001, type=float,
            help='Initial learning rate.')
  parser.add_argument('--lr_update', default=10, type=int,
            help='Number of epochs to update the learning rate.')
  parser.add_argument('--workers', default=10, type=int,
            help='Number of data loader workers.')
  parser.add_argument('--log_step', default=10, type=int,
            help='Number of steps to print and record the log.')
  parser.add_argument('--val_step', default=500, type=int,
            help='Number of steps to run validation.')
  parser.add_argument('--logger_name', default='runs/runX',
            help='Path to save the model and Tensorboard log.')
  parser.add_argument('--resume', default='', type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)')
  parser.add_argument('--max_violation', action='store_true',
            help='Use max instead of sum in the rank loss.')
  parser.add_argument('--img_dim', default=500, choices=[500, 2048],
            help='Dimensionality of the image embedding.')
  parser.add_argument('--measure', default='cosine',
            help='Similarity measure used (cosine|order)')
  parser.add_argument('--use_abs', action='store_true',
            help='Take the absolute value of embedding vectors.')
  parser.add_argument('--no_imgnorm', action='store_true',
            help='Do not normalize the image embeddings.')
  parser.add_argument('--gpu_id', default=0, type=int,
            help='GPU to use.')
  parser.add_argument('--rnn_type', default='maxout', choices=['maxout', 'seq2seq', 'attention'],
            help='Type of recurrent model.')
  parser.add_argument('--img_first_size', default=1024, type=int,
            help='first img layer emb size')
  parser.add_argument('--cap_first_size', default=1024, type=int,
            help='first cap layer emb size')
  parser.add_argument('--img_first_dropout', default=0, type=float,
            help='first img layer emb size')
  parser.add_argument('--cap_first_dropout', default=0, type=float,
            help='first cap layer emb size')
  parser.add_argument('--center_loss_weight', default=1, type=float,
            help='weight of center loss')

  parser.add_argument('--data_switch', default=0, type=int)
  parser.add_argument('--center_loss', action='store_true')
  parser.add_argument('--identity', action='store_true')
  parser.add_argument('--tune_seq', action='store_true')
  parser.add_argument('--no_correspond', action='store_true')
  parser.add_argument('--other_loss_weight', default=1, type=float)

  opt = parser.parse_args()
  print (opt)

  torch.cuda.set_device(opt.gpu_id)

  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  tb_logger.configure(opt.logger_name, flush_secs=5)

  # Load Vocabulary Wrapper
  vocab_path = os.path.join(opt.vocab_path, '%s_vocab_no_emb.pkl' % opt.data_name)
  print (vocab_path)
  vocab = pickle.load(open(vocab_path, 'rb'))
  opt.vocab_size = len(vocab)

  # Load data loaders
  train_loader, val_loader = data.get_loaders(
    opt.data_name, vocab, opt.batch_size, opt.workers, opt)

  # Construct the model
  model = VSE(opt)

  print('Print out models:')
  print(model.clip_enc)
  print(model.txt_enc)
  print(model.vid_seq_enc)
  print(model.txt_seq_enc)

  start_epoch = 0
  # optionally resume from a checkpoint
  if opt.resume:
    if os.path.isfile(opt.resume):
      print("=> loading checkpoint '{}'".format(opt.resume))
      checkpoint = torch.load(opt.resume)
      start_epoch = checkpoint['epoch']
      best_rsum = checkpoint['best_rsum']
      model.load_state_dict(checkpoint['model'])
      # Eiters is used to show logs as the continuation of another
      # training
      model.Eiters = checkpoint['Eiters']
      print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
          .format(opt.resume, start_epoch, best_rsum))
      validate(opt, val_loader, model)
    else:
      print("=> no checkpoint found at '{}'".format(opt.resume))

  # Train the Model
  best_rsum = 0
  for epoch in range(start_epoch, opt.num_epochs):
    adjust_learning_rate(opt, model.optimizer, epoch)

    # train for one epoch
    train(opt, train_loader, model, epoch, val_loader)

    # evaluate on validation set
    rsum = validate(opt, val_loader, model)

    # remember best R@ sum and save checkpoint
    is_best = rsum > best_rsum
    best_rsum = max(rsum, best_rsum)
    save_checkpoint({
      'epoch': epoch + 1,
      'model': model.state_dict(),
      'best_rsum': best_rsum,
      'opt': opt,
      'Eiters': model.Eiters,
    }, is_best, prefix=opt.logger_name + '/')


def train(opt, train_loader, model, epoch, val_loader):
  # average meters to record the training statistics
  batch_time = AverageMeter()
  data_time = AverageMeter()
  train_logger = LogCollector()

  # switch to train mode
  model.train_start()

  end = time.time()
  for i, train_data in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    # make sure train logger is used
    model.logger = train_logger

    # Update the model
    model.train_emb(opt, *train_data)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    # Print log info
    if model.Eiters % opt.log_step == 0:
      logging.info(
        'Epoch: [{0}][{1}/{2}]\t'
        '{e_log}\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        .format(
          epoch, i, len(train_loader), batch_time=batch_time,
          data_time=data_time, e_log=str(model.logger)))

    # Record logs in tensorboard
    tb_logger.log_value('epoch', epoch, step=model.Eiters)
    tb_logger.log_value('step', i, step=model.Eiters)
    tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
    tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
    model.logger.tb_log(tb_logger, step=model.Eiters)

    # validate at every val_step
    if model.Eiters % opt.val_step == 0:
      validate(opt, val_loader, model)


def validate(opt, val_loader, model):
  # compute the encoding for all the validation images and captions
  vid_seq_embs, para_seq_embs, clip_embs, cap_embs, _, _ = encode_data(
    model, val_loader, opt.log_step, logging.info)

  # caption retrieval
  vid_clip_rep, _, _ = i2t(clip_embs, cap_embs, measure=opt.measure)
  # image retrieval
  cap_clip_rep, _, _ = t2i(clip_embs, cap_embs, measure=opt.measure)

  # caption retrieval
  vid_seq_rep, _, _  = i2t(vid_seq_embs, para_seq_embs, measure=opt.measure)
  # image retrieval
  para_seq_rep, _, _ = t2i(vid_seq_embs, para_seq_embs, measure=opt.measure)

  # sum of recalls to be used for early stopping
  currscore = vid_seq_rep['sum'] + para_seq_rep['sum']

  logging.info("Clip to Sent: %.1f, %.1f, %.1f, %.1f, %.1f" %
         (vid_clip_rep['r1'], vid_clip_rep['r5'], vid_clip_rep['r10'], vid_clip_rep['medr'], vid_clip_rep['meanr']))
  logging.info("Sent to Clip: %.1f, %.1f, %.1f, %.1f, %.1f" %
         (cap_clip_rep['r1'], cap_clip_rep['r5'], cap_clip_rep['r10'], cap_clip_rep['medr'], cap_clip_rep['meanr']))
  logging.info("Video to Paragraph: %.1f, %.1f, %.1f, %.1f, %.1f" %
         (vid_seq_rep['r1'], vid_seq_rep['r5'], vid_seq_rep['r10'], vid_seq_rep['medr'], vid_seq_rep['meanr']))
  logging.info("Paragraph to Video: %.1f, %.1f, %.1f, %.1f, %.1f" %
         (para_seq_rep['r1'], para_seq_rep['r5'], para_seq_rep['r10'], para_seq_rep['medr'], para_seq_rep['meanr']))
  logging.info("Currscore: %.1f" % (currscore))

  # record metrics in tensorboard
  LogReporter(tb_logger, vid_clip_rep, model.Eiters, 'clip')
  LogReporter(tb_logger, cap_clip_rep, model.Eiters, 'clipi')
  LogReporter(tb_logger, vid_seq_rep, model.Eiters, 'seq')
  LogReporter(tb_logger, para_seq_rep, model.Eiters, 'seqi')
  tb_logger.log_value('rsum', currscore, step=model.Eiters)

  return currscore

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
  torch.save(state, prefix + filename)
  if is_best:
    shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
  """Sets the learning rate to the initial LR
     decayed by 10 every 30 epochs"""
  lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

if __name__ == '__main__':
  main()
