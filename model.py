import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from IPython import embed

from layers import *
from loss import *
from decoder.layers_v2 import *
from decoder.model_v2 import *
from decoder.loss import *

def EncoderImage(data_name, img_dim, embed_size,
                 finetune=False, dropout=0.5,
                 no_imgnorm=False, rnn_type='maxout', bidirectional=False):
  """A wrapper to image encoders. Chooses between an encoder that uses
  precomputed image features, `EncoderImagePrecomp`, or an encoder that
  computes image features on the fly `EncoderImageFull`.
  """
  if data_name.endswith('_precomp'):
    clip_enc = EncoderImagePrecomp(img_dim, embed_size, dropout=dropout, no_imgnorm=no_imgnorm, bidirectional=bidirectional, rnn_type=rnn_type)

  return clip_enc

class EncoderSequence(nn.Module):
  def __init__(self, img_dim, embed_size, dropout=0,
                     no_imgnorm=False, bidirectional=False, rnn_type='maxout'):
    super(EncoderSequence, self).__init__()
    self.embed_size = embed_size
    self.no_imgnorm = no_imgnorm
    self.bidirectional = bidirectional

    num_layers = 1
    if dropout > 0:
        self.dropout = nn.Dropout(dropout)
    if rnn_type == 'attention':
      self.rnn = Attention(img_dim, embed_size, rnn_bidirectional=bidirectional)
    elif rnn_type == 'seq2seq':
      self.rnn = Seq2Seq(img_dim, embed_size, rnn_bidirectional=bidirectional)
    elif rnn_type == 'maxout':
      self.rnn = Maxout(img_dim, embed_size, rnn_bidirectional=bidirectional)
    else:
      raise ValueError('Unsupported RNN type')

  def forward(self, x, lengths, hidden=None):
    """Extract image feature vectors."""
    outputs = self.rnn(x, lengths, hidden)

    # normalization in the joint embedding space
    return F.normalize(outputs)


class EncoderImagePrecomp(nn.Module):
  def __init__(self, img_dim, embed_size, dropout=0,
      no_imgnorm=False, bidirectional=False, rnn_type='maxout'):
    super(EncoderImagePrecomp, self).__init__()
    self.embed_size = embed_size
    self.no_imgnorm = no_imgnorm
    self.bidirectional = bidirectional
    self.dropout = dropout

    num_layers = 1
    if dropout > 0:
        self.dropout = nn.Dropout(dropout)
    if rnn_type == 'attention':
      self.rnn = Attention(img_dim, embed_size, rnn_bidirectional=bidirectional)
    elif rnn_type == 'seq2seq':
      self.rnn = Seq2Seq(img_dim, embed_size, rnn_bidirectional=bidirectional)
    elif rnn_type == 'maxout':
      self.rnn = Maxout(img_dim, embed_size, rnn_bidirectional=bidirectional)
    else:
      raise ValueError('Unsupported RNN type')

  def forward(self, x, lengths):
    """Extract image feature vectors."""
    if self.dropout > 0:
        x = self.dropout(x)
    else:
        pass
    outputs = self.rnn(x, lengths)

    # normalization in the joint embedding space
    return F.normalize(outputs)

class EncoderText(nn.Module):
  def __init__(self, vocab_size, word_dim, embed_size,
      num_layers, dropout=0, bidirectional=False, rnn_type='maxout', data_name='anet_precomp'):
    super(EncoderText, self).__init__()
    self.embed_size = embed_size
    self.bidirectional = bidirectional
    self.dropout = dropout

    # word embedding
    self.embed   = nn.Embedding(vocab_size, word_dim)

    # caption embedding
    if dropout > 0:
        self.dropout = nn.Dropout(dropout)
    if rnn_type == 'attention':
      self.rnn = Attention(word_dim, embed_size, rnn_bidirectional=bidirectional)
    elif rnn_type == 'seq2seq':
      self.rnn = Seq2Seq(word_dim, embed_size, rnn_bidirectional=bidirectional)
    elif rnn_type == 'maxout':
      self.rnn = Maxout(word_dim, embed_size, rnn_bidirectional=bidirectional)
    else:
      raise ValueError('Unsupported RNN type')

    self.init_weights(data_name)

  def init_weights(self, data_name):
#    self.embed.weight.data.uniform_(-0.1, 0.1)
    self.embed.weight.data = torch.from_numpy(np.load('vocab/{}_w2v.npz'.format(data_name))['arr_0'].astype(float)).float()

  def forward(self, x, lengths):
    # Embed word ids to vectors
    if self.dropout > 0:
        cap_emb = self.dropout(self.embed(x))
    else:
        cap_emb = self.embed(x)
    outputs = self.rnn(cap_emb, lengths)

    # normalization in the joint embedding space
    return F.normalize(outputs)

class FC(nn.Module):
  def __init__(self, input_size, output_size, identity):
    super(FC, self).__init__()
    self.output_size = output_size
    self.fc = nn.Sequential(nn.Linear(input_size, output_size, bias=False), nn.ReLU(), nn.Linear(output_size, input_size, bias=False))
    if identity: self.init_param()

  def init_param(self):
    self.fc[0].weight.data.copy_(torch.eye(self.output_size))
    self.fc[2].weight.data.copy_(torch.eye(self.output_size))

  def forward(self, clip_emb):
    img_out = self.fc(clip_emb)
    return F.normalize(img_out)

class VSE(object):
  def __init__(self, opt):
    self.grad_clip = opt.grad_clip
    self.clip_enc = EncoderImage(opt.data_name, opt.img_dim, opt.img_first_size, dropout=opt.img_first_dropout,
                  no_imgnorm=opt.no_imgnorm, rnn_type=opt.rnn_type)
    self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim, opt.cap_first_size, opt.num_layers, dropout=opt.cap_first_dropout,
                  rnn_type=opt.rnn_type, data_name = opt.data_name)
    self.vid_seq_enc = EncoderSequence(opt.img_first_size, opt.embed_size,
                  rnn_type=opt.rnn_type)
    self.txt_seq_enc = EncoderSequence(opt.cap_first_size, opt.embed_size,
                  rnn_type=opt.rnn_type)
    self.vid_seq_dec = DecoderSequence(opt.embed_size, opt.img_first_size,
                  rnn_type=opt.decode_rnn_type)
    self.txt_seq_dec = DecoderSequence(opt.embed_size, opt.cap_first_size,
                  rnn_type=opt.decode_rnn_type)

    self.fc_visual = FC(opt.embed_size, opt.img_first_size, opt.identity)
    self.fc_language = FC(opt.embed_size, opt.cap_first_size, opt.identity)
    if torch.cuda.is_available():
      self.clip_enc.cuda()
      self.txt_enc.cuda()
      self.vid_seq_enc.cuda()
      self.txt_seq_enc.cuda()
      self.fc_visual.cuda()
      self.fc_language.cuda()
      self.vid_seq_dec.cuda()
      self.txt_seq_dec.cuda()
      cudnn.benchmark = True

    # Loss and Optimizer
    self.criterion = ContrastiveLoss(margin=opt.margin,
                     measure=opt.measure,
                     max_violation=opt.max_violation)
    self.criterion_group = GroupWiseContrastiveLoss(margin=opt.margin,
                     measure=opt.measure,
                     max_violation=opt.max_violation)
    self.criterion_center = CenterLoss(margin=opt.margin,
                     measure=opt.measure,
                     max_violation=opt.max_violation, tune_center=opt.tune_seq)
    self.criterion_Euclid_Distance = EuclideanLoss()

    params = list(self.txt_enc.parameters())
    params += list(self.clip_enc.parameters())
    params += list(self.vid_seq_enc.parameters())
    params += list(self.txt_seq_enc.parameters())
    params += list(self.fc_visual.parameters())
    params += list(self.fc_language.parameters())
    params += list(self.vid_seq_dec.parameters())
    params += list(self.txt_seq_dec.parameters())
    self.params = params

    self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

    self.Eiters = 0

  def state_dict(self):
    state_dict = [self.clip_enc.state_dict(), self.txt_enc.state_dict(), \
                  self.vid_seq_enc.state_dict(), self.txt_seq_enc.state_dict(), \
                  self.fc_visual.state_dict(), self.fc_language.state_dict(), \
                  self.vid_seq_dec.state_dict(), self.txt_seq_dec.state_dict()]
    return state_dict

  def load_state_dict(self, state_dict):
    self.clip_enc.load_state_dict(state_dict[0])
    self.txt_enc.load_state_dict(state_dict[1])
    self.vid_seq_enc.load_state_dict(state_dict[2])
    self.txt_seq_enc.load_state_dict(state_dict[3])
    self.fc_visual.load_state_dict(state_dict[4])
    self.fc_language.load_state_dict(state_dict[5])
    self.vid_seq_dec.load_state_dict(state_dict[6])
    self.txt_seq_dec.load_state_dict(state_dict[7])

  def train_start(self):
    """switch to train mode
    """
    self.clip_enc.train()
    self.txt_enc.train()
    self.vid_seq_enc.train()
    self.txt_seq_enc.train()
    self.fc_visual.train()
    self.fc_language.train()
    self.vid_seq_dec.train()
    self.txt_seq_dec.train()

  def val_start(self):
    """switch to evaluate mode
    """
    self.clip_enc.eval()
    self.txt_enc.eval()
    self.vid_seq_enc.eval()
    self.txt_seq_enc.eval()
    self.fc_visual.eval()
    self.fc_language.eval()
    self.vid_seq_dec.eval()
    self.txt_seq_dec.eval()

  def forward_emb(self, clips, captions, lengths_clip, lengths_cap):
    clips   = Variable(clips)
    captions = Variable(captions)
    if torch.cuda.is_available():
      clips = clips.cuda()
      captions = captions.cuda()

    # Forward
    clip_emb = self.clip_enc(clips, Variable(lengths_clip))
    cap_emb = self.txt_enc(captions, Variable(lengths_cap))
    return clip_emb, cap_emb

  def structure_emb(self, clip_emb, cap_emb, num_clips, num_caps, vid_context=None, para_context=None):
    img_reshape_emb = Variable(torch.zeros(len(num_clips), max(num_clips), clip_emb.shape[1])).cuda()
    cap_reshape_emb = Variable(torch.zeros(len(num_caps),  max(num_caps),  cap_emb.shape[1])).cuda()

    cur_displace = 0
    for i, end_place in enumerate(num_clips):
      img_reshape_emb[i, 0:end_place, :] = clip_emb[cur_displace : cur_displace + end_place, :]
      cur_displace = cur_displace + end_place

    cur_displace = 0
    for i, end_place in enumerate(num_caps):
      cap_reshape_emb[i, 0:end_place, :] = cap_emb[cur_displace : cur_displace + end_place, :]
      cur_displace = cur_displace + end_place

    vid_emb  = self.vid_seq_enc(img_reshape_emb, Variable(torch.Tensor(num_clips)), vid_context)
    para_emb = self.txt_seq_enc(cap_reshape_emb, Variable(torch.Tensor(num_caps)), para_context)

    return vid_emb, para_emb

  def remap_emb(self, vid_emb, para_emb, num_clips, num_caps, clip_emb=None, cap_emb=None):
    vid_reshape_emb = Variable(torch.zeros(len(num_clips), max(num_clips), vid_emb.shape[1])).cuda()
    para_reshape_emb = Variable(torch.zeros(len(num_caps),  max(num_caps),  para_emb.shape[1])).cuda()

    if clip_emb == None and cap_emb == None:
        for i, end_place in enumerate(num_clips):
            vid_reshape_emb[i,0,:] = vid_emb[i]
            for k in range(1, end_place):
                vid_reshape_emb[i, k, :] = clip_emb[i,k-1,:]

        for i, end_place in enumerate(num_caps):
          para_reshape_emb[i,0,:] = para_emb[i]
          for k in range(1, end_place):
              para_reshape_emb[i, k, :] = cap_emb[i,k-1,:]
    else:
        for i, end_place in enumerate(num_clips):
            for k in range(end_place):
                vid_reshape_emb[i, k, :] = vid_emb[i]

        for i, end_place in enumerate(num_caps):
          for k in range(end_place):
              para_reshape_emb[i, k, :] = para_emb[i,:]

    vid_emb  = self.vid_seq_dec(vid_reshape_emb, Variable(torch.Tensor(num_clips)))
    para_emb = self.txt_seq_dec(para_reshape_emb, Variable(torch.Tensor(num_caps)))

    return vid_emb, para_emb



  def forward_loss(self, clip_emb, cap_emb, name, **kwargs):
    """Compute the loss given pairs of image and caption embeddings
    """
    loss = self.criterion(clip_emb, cap_emb)
    self.logger.update('Le'+name, loss.data[0], clip_emb.size(0))
    return loss

  def forward_loss_group(self, clip_emb, cap_emb, num_clips, num_caps, name, **kwargs):
    """Compute the loss given pairs of image and caption embeddings
    """
    loss = self.criterion_group(clip_emb, cap_emb, num_clips, num_caps)
    self.logger.update('Le'+name, loss.data[0], clip_emb.size(0))
    return loss

  def forward_center_loss(self, clip_emb, vid_emb, cap_emb, para_emb, num_clips, num_caps, name, **kwargs):
    """Compute the loss given pairs of image and caption embeddings
    """
    loss = self.criterion_center(clip_emb, vid_emb, cap_emb, para_emb, num_clips, num_caps)
    self.logger.update('Le'+name, loss.data[0], clip_emb.size(0))
    return loss

  def forward_remap_loss(self, vid_emb, clip_emb, name, **kwargs):
    """Compute the loss given pairs of image and caption embeddings
    """
    loss = self.criterion_Euclid_Distance(vid_emb, clip_emb)
    self.logger.update('Le'+name, loss.data[0], clip_emb.size(0))
    return loss

  def train_emb(self, opts, clips, captions, videos, paragraphs,
      lengths_clip, lengths_cap, lengths_video, lengths_paragraph,
      num_clips, num_caps, ind, *args):
    """One training step given clips and captions.
    """
    self.Eiters += 1
    self.logger.update('Eit', self.Eiters)
    self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

    # compute the embeddings
    clip_emb, cap_emb = self.forward_emb(clips, captions, lengths_clip, lengths_cap)
    vid_context, para_context = self.forward_emb(videos, paragraphs, lengths_video, lengths_paragraph)
    vid_emb, para_emb = self.structure_emb(clip_emb, cap_emb, num_clips, num_caps, vid_context, para_context)
    if opts.remap_term:
        clip_remap, cap_remap = self.remap_emb(vid_emb, para_emb, num_clips, num_caps)

    if opts.center_loss:
      vid_reproject  = self.fc_visual(vid_emb)
      para_reproject = self.fc_language(para_emb)

    # measure accuracy and record loss
    self.optimizer.zero_grad()

    loss_1 = self.forward_loss(vid_emb, para_emb, '_vid')
    if opts.no_correspond:
      loss_2 = self.forward_loss_group(clip_emb, cap_emb, num_clips, num_caps, '_clip')
    else:
      loss_2 = self.forward_loss(clip_emb, cap_emb, '_clip')

    loss_3 = self.forward_loss(vid_context, para_context, '_context')
    if opts.center_loss:
      loss_4 = self.forward_center_loss(clip_emb, vid_reproject, cap_emb, para_reproject, num_clips, num_caps, '_center')
    else:
      loss_4 = 0

    loss_5 = self.forward_loss(vid_emb, vid_emb, '_ex_vid') + self.forward_loss(para_emb, para_emb, '_ex_para')

    if opts.remap_term:
        loss_remap = self.forward_remap_loss(clip_remap, clip_emb, '_remap_clip') + self.forward_remap_loss(cap_remap, cap_emb, '_remap_cap')
    else:
        loss_remap = 0

    loss = loss_1 + (loss_2 + loss_3 + opts.center_loss_weight * loss_4 + loss_5 + loss_remap) * opts.other_loss_weight

    # compute gradient and do SGD step
    loss.backward()
    if self.grad_clip > 0: clip_grad_norm(self.params, self.grad_clip)
    self.optimizer.step()

  def test_emb(self, clips, captions, lengths_clip, lengths_cap, ind, num_clips, num_caps, offset=0):
    clip_emb, cap_emb = self.forward_emb(clips, captions, lengths_clip, lengths_cap)
    img_reshape_emb = Variable(torch.zeros(len(num_clips), max(num_clips), clip_emb.shape[1])).cuda()
    cap_reshape_emb = Variable(torch.zeros(len(num_caps),  max(num_caps),  cap_emb.shape[1])).cuda()

    cur_displace = 0
    for i, seg_num in enumerate(num_clips):
      seg_len = max(seg_num - offset, 1)
      img_reshape_emb[i, 0:seg_len, :] = clip_emb[cur_displace : cur_displace + seg_len, :]
      cur_displace = cur_displace + seg_num

    cur_displace = 0
    for i, seg_num in enumerate(num_caps):
      cap_reshape_emb[i, 0:seg_num, :] = cap_emb[cur_displace : cur_displace + seg_num, :]
      cur_displace = cur_displace + seg_num

    vid_emb  = self.vid_seq_enc(img_reshape_emb, Variable(torch.Tensor(num_clips)))
    para_emb = self.txt_seq_enc(cap_reshape_emb, Variable(torch.Tensor(num_caps)))

    return vid_emb, para_emb
