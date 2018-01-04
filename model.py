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

def EncoderImage(data_name, img_dim, embed_size, finetune=False, dropout=0.5, 
        no_imgnorm=False, rnn_type='maxout', bidirectional=False):
  """A wrapper to image encoders. Chooses between an encoder that uses
  precomputed image features, `EncoderImagePrecomp`, or an encoder that
  computes image features on the fly `EncoderImageFull`.
  """
  if data_name.endswith('_precomp'):
    img_enc = EncoderImagePrecomp(img_dim, embed_size, dropout=dropout, no_imgnorm=no_imgnorm, bidirectional=bidirectional, rnn_type=rnn_type)

  return img_enc

class EncoderSequence(nn.Module):
  def __init__(self, img_dim, embed_size, dropout=0, no_imgnorm=False, bidirectional=False, rnn_type='maxout'):
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

    # self.mlp = GroupMLP(embed_size, 2048, embed_size, drop=0.5, groups=64)

  def forward(self, x, lengths):
    """Extract image feature vectors."""
    #img_emb = self.dropout(x)
    # outputs = self.rnn(img_emb, lengths)
    outputs = self.rnn(x, lengths)

    # normalization in the joint embedding space
    return F.normalize(outputs)

class EncoderImagePrecomp(nn.Module):
  def __init__(self, img_dim, embed_size, dropout=0, no_imgnorm=False, bidirectional=False, rnn_type='maxout'):
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

    # self.mlp = GroupMLP(embed_size, 2048, embed_size, drop=0.5, groups=64)

  def forward(self, x, lengths):
    """Extract image feature vectors."""
    if self.dropout > 0:
        x = self.dropout(x)
    else:
        pass
    # outputs = self.rnn(img_emb, lengths)
    outputs = self.rnn(x, lengths)

    # normalization in the joint embedding space
    return F.normalize(outputs)
    # return F.normalize(self.mlp(outputs))

class EncoderText(nn.Module):
  def __init__(self, vocab_size, word_dim, embed_size, num_layers, dropout=0, bidirectional=False, rnn_type='maxout'):
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

    # self.mlp = GroupMLP(embed_size, 2048, embed_size, drop=0.5, groups=64)

    self.init_weights()

  def init_weights(self):
    self.embed.weight.data = torch.from_numpy(np.load('vocab/anet_precomp_w2v.npz')['arr_0'].astype(float)).float()

  def forward(self, x, lengths):
    """Handles variable size captions
    """
    # Embed word ids to vectors
    if self.dropout > 0:
        cap_emb = self.dropout(self.embed(x))
    else:
        cap_emb = self.embed(x)
    outputs = self.rnn(cap_emb, lengths)

    # normalization in the joint embedding space
    # return F.normalize(self.mlp(outputs))
    return F.normalize(outputs)

def cosine_sim(im, s):
  """Cosine similarity between all the image and sentence pairs
  """
  return im.mm(s.t())

class ContrastiveLoss_no_correspond(nn.Module):
  """
  Compute contrastive loss
  """

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
  """
  Compute contrastive loss
  """

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
  """
  Compute contrastive loss
  """

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

class FC(nn.Module):
    def __init__(self, input_size, output_size, identity):
        super(FC, self).__init__()
        self.output_size = output_size
        self.fc = nn.Sequential(nn.Linear(input_size, output_size, bias=False), nn.ReLU(), nn.Linear(output_size, input_size, bias=False))
        if identity:
            self.init_param()

    def init_param(self):
        self.fc[0].weight.data.copy_(torch.eye(self.output_size))
        self.fc[2].weight.data.copy_(torch.eye(self.output_size))

        
    def forward(self, img_emb):
        img_out = self.fc(img_emb)
        return F.normalize(img_out)


class VSE(object):
  """
  rkiros/uvs model
  """

  def __init__(self, opt):
    # tutorials/09 - Image Captioning
    # Build Models
    self.grad_clip = opt.grad_clip
    self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.img_first_size, dropout=opt.img_first_dropout,
                  no_imgnorm=opt.no_imgnorm, rnn_type=opt.rnn_type)
    self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim, opt.cap_first_size, opt.num_layers, dropout=opt.cap_first_dropout,
                  rnn_type=opt.rnn_type)
    self.img_seq_enc = EncoderSequence(opt.img_first_size, opt.embed_size,
                  rnn_type=opt.rnn_type)
    self.txt_seq_enc = EncoderSequence(opt.cap_first_size, opt.embed_size,
                  rnn_type=opt.rnn_type)
    self.fc_visual = FC(opt.embed_size, opt.img_first_size, opt.identity)
    self.fc_language = FC(opt.embed_size, opt.cap_first_size, opt.identity)
    if torch.cuda.is_available():
      self.img_enc.cuda()
      self.txt_enc.cuda()
      self.img_seq_enc.cuda()
      self.txt_seq_enc.cuda()
      self.fc_visual.cuda()
      self.fc_language.cuda()
      cudnn.benchmark = True

    # Loss and Optimizer
    self.criterion = ContrastiveLoss(margin=opt.margin,
                     measure=opt.measure,
                     max_violation=opt.max_violation)
    self.criterion_no_correspond = ContrastiveLoss_no_correspond(margin=opt.margin,
                     measure=opt.measure,
                     max_violation=opt.max_violation)
 
    self.criterion_center = CenterLoss(margin=opt.margin,
                     measure=opt.measure,
                     max_violation=opt.max_violation, tune_center=opt.tune_seq)

    #self.criterion_no_align = ContrastiveLoss_NoAlign(margin = opt.margin, measure = opt.measure, max_violation = opt.max_violation)
 
    params = list(self.txt_enc.parameters())
    params += list(self.img_enc.parameters())
    params += list(self.img_seq_enc.parameters())
    params += list(self.txt_seq_enc.parameters())
    params += list(self.fc_visual.parameters())
    params += list(self.fc_language.parameters())
    self.params = params

    self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

    self.Eiters = 0

  def state_dict(self):
    state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), \
                  self.img_seq_enc.state_dict(), self.txt_seq_enc.state_dict(), \
                  self.fc_visual.state_dict(), self.fc_language.state_dict()]
    return state_dict

  def load_state_dict(self, state_dict):
    self.img_enc.load_state_dict(state_dict[0])
    self.txt_enc.load_state_dict(state_dict[1])
    self.img_seq_enc.load_state_dict(state_dict[2])
    self.txt_seq_enc.load_state_dict(state_dict[3])
    self.fc_visual.load_state_dict(state_dict[4])
    self.fc_language.load_state_dict(state_dict[5])

  def train_start(self):
    """switch to train mode
    """
    self.img_enc.train()
    self.txt_enc.train()
    self.img_seq_enc.train()
    self.txt_seq_enc.train()
    self.fc_visual.train()
    self.fc_language.train()

  def val_start(self):
    """switch to evaluate mode
    """
    self.img_enc.eval()
    self.txt_enc.eval()
    self.img_seq_enc.eval()
    self.txt_seq_enc.eval()
    self.fc_visual.eval()
    self.fc_language.eval()

  def forward_emb(self, images, captions, lengths_img, lengths_cap):
    """Compute the image and caption embeddings
    """
    # Set mini-batch dataset
    images   = Variable(images)
    captions = Variable(captions)
    if torch.cuda.is_available():
      images = images.cuda()
      captions = captions.cuda()

    # Forward
    img_emb = self.img_enc(images, Variable(lengths_img))
    cap_emb = self.txt_enc(captions, Variable(lengths_cap))
    return img_emb, cap_emb

  def structure_emb(self, img_emb, cap_emb, seg_num):
    img_reshape_emb = Variable(torch.zeros(len(seg_num), max(seg_num), img_emb.shape[1])).cuda()
    cap_reshape_emb = Variable(torch.zeros(len(seg_num), max(seg_num), cap_emb.shape[1])).cuda()

    cur_displace = 0
    for i, end_place in enumerate(seg_num):
      img_reshape_emb[i, 0:end_place, :] = img_emb[cur_displace : cur_displace + end_place, :]
      cap_reshape_emb[i, 0:end_place, :] = cap_emb[cur_displace : cur_displace + end_place, :]
      cur_displace = cur_displace + end_place

    img_seq_emb = self.img_seq_enc(img_reshape_emb, Variable(torch.Tensor(seg_num)))
    cap_seq_emb = self.txt_seq_enc(cap_reshape_emb, Variable(torch.Tensor(seg_num)))

    return img_seq_emb, cap_seq_emb


  def forward_loss(self, img_emb, cap_emb, name, **kwargs):
    """Compute the loss given pairs of image and caption embeddings
    """
    loss = self.criterion(img_emb, cap_emb)
    self.logger.update('Le'+name, loss.data[0], img_emb.size(0))
    return loss

  def forward_loss_no_correspond(self, img_emb, cap_emb, seg_num, name, **kwargs):
    """Compute the loss given pairs of image and caption embeddings
    """
    loss = self.criterion_no_correspond(img_emb, cap_emb, seg_num)
    self.logger.update('Le'+name, loss.data[0], img_emb.size(0))
    return loss



  def forward_center_loss(self, clip_emb, vid_emb, txt_emb, para_emb, seg_num, name, **kwargs):
    """Compute the loss given pairs of image and caption embeddings
    """
    loss = self.criterion_center(clip_emb, vid_emb, txt_emb, para_emb, seg_num)
    self.logger.update('Le'+name, loss.data[0], clip_emb.size(0))
    return loss


  def train_emb(self, opts, images, captions, video_whole, captions_whole, lengths_img, lengths_cap, lengths_whole_vid, lengths_whole_cap, ind, seg_num, *args):
    """One training step given images and captions.
    """
    self.Eiters += 1
    self.logger.update('Eit', self.Eiters)
    self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

    # compute the embeddings
    img_emb, cap_emb = self.forward_emb(images, captions, lengths_img, lengths_cap)
    if opts.seq_loss:
        img_seq_emb, cap_seq_emb = self.structure_emb(img_emb, cap_emb, seg_num)
    if opts.whole_loss:
        vid_whole_emb, cap_whole_emb = self.forward_emb(video_whole, captions_whole, lengths_whole_vid, lengths_whole_cap)
    if opts.center_loss:
        img_seq_emb_down = self.fc_visual(img_seq_emb)
        cap_seq_emb_down = self.fc_language(cap_seq_emb)

    # measure accuracy and record loss
    self.optimizer.zero_grad()
    if opts.no_correspond:
        loss_2 = self.forward_loss_no_correspond(img_emb, cap_emb, seg_num, 'vid')
    else:
        loss_2 = self.forward_loss(img_emb, cap_emb, 'vid')
    if opts.seq_loss:
        loss_1 = self.forward_loss(img_seq_emb, cap_seq_emb, 'seq')
    else:
        loss_1 = 0
    if opts.whole_loss:
        loss_3 = self.forward_loss(vid_whole_emb, cap_whole_emb, 'whole')
    else:
        loss_3 = 0
    if opts.center_loss:
        loss_4 = self.forward_center_loss(img_emb, img_seq_emb_down, cap_emb, cap_seq_emb_down, seg_num, 'sim_loss')
#    loss_5 = self.forward_center_loss(img_emb, cap_seq_emb, img_emb, cap_seq_emb, seg_num, 'cross_loss')
        loss = loss_1 + (loss_2 + loss_3 + opts.center_loss_weight * loss_4 ) * opts.other_loss_weight
    else:
#        loss = loss_3
        loss = loss_1 + (loss_2 + loss_3) * opts.other_loss_weight

    # compute gradient and do SGD step
    loss.backward()
    if self.grad_clip > 0:
      clip_grad_norm(self.params, self.grad_clip)
    self.optimizer.step()

  def test_emb(self, images, captions, lengths_img, lengths_cap, ind, seg_nums, offset=0):
    img_emb, cap_emb = self.forward_emb(images, captions, lengths_img, lengths_cap)
    img_reshape_emb = Variable(torch.zeros(len(seg_nums), max(seg_nums), img_emb.shape[1])).cuda()
    cap_reshape_emb = Variable(torch.zeros(len(seg_nums), max(seg_nums), cap_emb.shape[1])).cuda()

    cur_displace = 0
    for i, seg_num in enumerate(seg_nums):
      seg_len = max(seg_num - offset, 1)
      img_reshape_emb[i, 0:seg_len, :] = img_emb[cur_displace : cur_displace + seg_len, :]
      cap_reshape_emb[i, 0:seg_num, :] = cap_emb[cur_displace : cur_displace + seg_num, :]
      cur_displace = cur_displace + seg_num

    img_seq_emb = self.img_seq_enc(img_reshape_emb, Variable(torch.Tensor(seg_nums)))
    cap_seq_emb = self.txt_seq_enc(cap_reshape_emb, Variable(torch.Tensor(seg_nums)))

    return img_seq_emb, cap_seq_emb
