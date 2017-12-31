import pickle
import os
import time
import shutil

import torch

import data
from vocab import Vocabulary  # NOQA
from model import VSE
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_eval_data

import logging
import tensorboard_logger as tb_logger

import argparse

from IPython import embed

torch.cuda.manual_seed_all(1)
torch.manual_seed(1)

def main():
  # Hyper Parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', default='/data1/hexianghu/activitynet/captions/',
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
  parser.add_argument('--resume', default='', type=str, metavar='PATH', required=True,
            help='path to latest checkpoint (default: none)')
  parser.add_argument('--max_violation', action='store_true',
            help='Use max instead of sum in the rank loss.')
  parser.add_argument('--img_dim', default=500, type=int,
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
 
  opt = parser.parse_args()
  print(opt)

  torch.cuda.set_device(opt.gpu_id)

  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  tb_logger.configure(opt.logger_name, flush_secs=5)

  # Load Vocabulary Wrapper
  vocab = pickle.load(open(os.path.join(
    opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
  opt.vocab_size = len(vocab)

  # Load data loaders
  train_loader, val_loader = data.get_loaders(
    opt.data_name, vocab, opt.batch_size, opt.workers, opt)

  # Construct the model
  model = VSE(opt)

  print('Print out models:')
  print(model.img_enc)
  print(model.txt_enc)
  print(model.img_seq_enc)
  print(model.txt_seq_enc)

  # optionally resume from a checkpoint
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

def validate(opt, val_loader, model, num_offsets=10):
  # compute the encoding for all the validation images and captions
  img_seq_embs, cap_seq_embs = encode_eval_data(
    model, val_loader, opt.log_step, logging.info, num_offsets=num_offsets)

  for _offset in xrange(num_offsets):
    logging.info("Offset: %.1f" % _offset )

    # caption retrieval
    (seq_r1, seq_r5, seq_r10, seq_medr, seq_meanr) = i2t(
        img_seq_embs[_offset], cap_seq_embs[_offset], measure=opt.measure)
    logging.info("seq_Image to seq_text: %.1f, %.1f, %.1f, %.1f, %.1f" %
          (seq_r1, seq_r5, seq_r10, seq_medr, seq_meanr))
    # image retrieval
    (seq_r1i, seq_r5i, seq_r10i, seq_medri, seq_meanr) = t2i(
        img_seq_embs[_offset], cap_seq_embs[_offset], measure=opt.measure)
    logging.info("seq_Text to seq_image: %.1f, %.1f, %.1f, %.1f, %.1f" %
          (seq_r1i, seq_r5i, seq_r10i, seq_medri, seq_meanr))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
  torch.save(state, prefix + filename)
  if is_best:
    shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')

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
