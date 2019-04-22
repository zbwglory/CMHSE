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
from decoder.loss import *
from decoder.model import *
from decoder.layers import *
import time

class EncoderImage(nn.Module):
  def __init__(self, img_dim, embed_size, bidirectional=False, rnn_type='maxout'):
    super(EncoderImage, self).__init__()
    self.embed_size = embed_size
    self.bidirectional = bidirectional

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
    outputs = self.rnn(x, lengths)

    # normalization in the joint embedding space
    # return F.normalize(outputs)
    return outputs

class EncoderSequence(nn.Module):
  def __init__(self, img_dim, embed_size, bidirectional=False, rnn_type='maxout'):
    super(EncoderSequence, self).__init__()
    self.embed_size = embed_size
    self.bidirectional = bidirectional

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
    # return F.normalize(outputs)
    return outputs

class EncoderText(nn.Module):
  def __init__(self, vocab_size, word_dim, embed_size,
      bidirectional=False, rnn_type='maxout', data_name='anet_precomp'):
    super(EncoderText, self).__init__()
    self.embed_size = embed_size
    self.bidirectional = bidirectional

    # word embedding
    self.embed   = nn.Embedding(vocab_size, word_dim)

    # caption embedding
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
    self.embed.weight.data = torch.from_numpy(np.load('vocab/{}_w2v_total.npz'.format(data_name))['arr_0'].astype(float)).float()

  def forward(self, x, lengths):
    # Embed word ids to vectors
    cap_emb = self.embed(x)
    outputs = self.rnn(cap_emb, lengths)

    # normalization in the joint embedding space
    # return F.normalize(outputs), cap_emb
    return outputs, cap_emb


class VSE(object):
  def __init__(self, opt):
    self.norm = opt.norm
    self.grad_clip = opt.grad_clip
    self.clip_enc = EncoderImage(opt.img_dim, opt.img_first_size,
                  rnn_type=opt.rnn_type)
    self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim, opt.cap_first_size,
                  rnn_type=opt.rnn_type, data_name = opt.data_name)
    self.vid_seq_enc = EncoderSequence(opt.img_first_size, opt.embed_size,
                  rnn_type=opt.rnn_type)
    self.txt_seq_enc = EncoderSequence(opt.cap_first_size, opt.embed_size,
                  rnn_type=opt.rnn_type)

    if torch.cuda.is_available():
      self.clip_enc.cuda()
      self.txt_enc.cuda()
      self.vid_seq_enc.cuda()
      self.txt_seq_enc.cuda()
      cudnn.benchmark = True

    # Loss and Optimizer
    self.criterion = ContrastiveLoss(margin=opt.margin,
                     measure=opt.measure,
                     max_violation=opt.max_violation, norm=self.norm)

    self.weak_criterion = GroupWiseContrastiveLoss(margin=opt.margin,
                     measure=opt.measure,
                     max_violation=opt.max_violation, norm=self.norm)


    params = list(self.txt_enc.parameters())
    params += list(self.clip_enc.parameters())
    params += list(self.vid_seq_enc.parameters())
    params += list(self.txt_seq_enc.parameters())


    if opt.reconstruct_loss:
        self.vid_seq_dec = DecoderSequence(opt.embed_size, opt.img_first_size,
                           rnn_type=opt.decode_rnn_type)
        self.txt_seq_dec = DecoderSequence(opt.embed_size, opt.cap_first_size,
                           rnn_type=opt.decode_rnn_type)
        self.vid_seq_dec.cuda()
        self.txt_seq_dec.cuda()

        self.criterion_Euclid_Distance = EuclideanLoss(norm=self.norm)

        params += list(self.vid_seq_dec.parameters())
        params += list(self.txt_seq_dec.parameters())

    if opt.lowest_reconstruct_loss:
        self.clip_seq_dec = DecoderSequence(opt.embed_size, opt.img_dim, rnn_type=opt.decode_rnn_type)
        self.sent_seq_dec = DecoderSequence(opt.embed_size, opt.word_dim, rnn_type=opt.decode_rnn_type)
        self.clip_seq_dec.cuda()
        self.sent_seq_dec.cuda()

        params += list(self.clip_seq_dec.parameters())
        params += list(self.sent_seq_dec.parameters())

    self.params = params

    self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

    self.Eiters = 0

  def state_dict(self, opt):
    state_dict = [self.clip_enc.state_dict(), self.txt_enc.state_dict(), \
                  self.vid_seq_enc.state_dict(), self.txt_seq_enc.state_dict()]
    if opt.reconstruct_loss:
        state_dict = [self.clip_enc.state_dict(), self.txt_enc.state_dict(), \
                      self.vid_seq_enc.state_dict(), self.txt_seq_enc.state_dict(), \
                      self.vid_seq_dec.state_dict(), self.txt_seq_dec.state_dict()]
    if opt.lowest_reconstruct_loss:
        state_dict = [self.clip_enc.state_dict(), self.txt_enc.state_dict(), \
                      self.vid_seq_enc.state_dict(), self.txt_seq_enc.state_dict(), \
                      self.vid_seq_dec.state_dict(), self.txt_seq_dec.state_dict(), \
                      self.clip_seq_dec.state_dict(), self.sent_seq_dec.state_dict()]

    return state_dict

  def load_state_dict(self, state_dict, opt):
    self.clip_enc.load_state_dict(state_dict[0])
    self.txt_enc.load_state_dict(state_dict[1])
    self.vid_seq_enc.load_state_dict(state_dict[2])
    self.txt_seq_enc.load_state_dict(state_dict[3])
    if opt.reconstruct_loss:
        self.vid_seq_dec.load_state_dict(state_dict[4])
        self.txt_seq_dec.load_state_dict(state_dict[5])
    if opt.lowest_reconstruct_loss:
        self.clip_seq_dec.load_state_dict(state_dict[6])
        self.sent_seq_dec.load_state_dict(state_dict[7])

  def train_start(self, opt):
    """switch to train mode
    """
    self.clip_enc.train()
    self.txt_enc.train()
    self.vid_seq_enc.train()
    self.txt_seq_enc.train()
    if opt.reconstruct_loss:
        self.vid_seq_dec.train()
        self.txt_seq_dec.train()
    if opt.lowest_reconstruct_loss:
        self.clip_seq_dec.train()
        self.sent_seq_dec.train()

  def val_start(self, opt):
    """switch to evaluate mode
    """
    self.clip_enc.eval()
    self.txt_enc.eval()
    self.vid_seq_enc.eval()
    self.txt_seq_enc.eval()
    if opt.reconstruct_loss:
        self.vid_seq_dec.eval()
        self.txt_seq_dec.eval()
    if opt.lowest_reconstruct_loss:
        self.clip_seq_dec.eval()
        self.sent_seq_dec.eval()


  def forward_emb(self, clips, captions, lengths_clip, lengths_cap, return_word=False):
    clips    = Variable(clips)
    captions = Variable(captions)
    if torch.cuda.is_available():
      clips = clips.cuda()
      captions = captions.cuda()

    # Forward
    clip_emb = self.clip_enc(clips, Variable(lengths_clip))
    cap_emb, word = self.txt_enc(captions, Variable(lengths_cap))

    if return_word:
        return clip_emb, cap_emb, word
    else:
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

    vid_emb  = self.vid_seq_enc(img_reshape_emb, Variable(torch.Tensor(num_clips).long()), vid_context)
    para_emb = self.txt_seq_enc(cap_reshape_emb, Variable(torch.Tensor(num_caps).long()), para_context)

    return vid_emb, para_emb

  def reconstruct_emb(self, vid_emb, para_emb, num_clips, num_caps):
    vid_reshape_emb = Variable(torch.zeros(len(num_clips), max(num_clips), vid_emb.shape[1])).cuda()
    para_reshape_emb = Variable(torch.zeros(len(num_caps),  max(num_caps),  para_emb.shape[1])).cuda()

    for i, end_place in enumerate(num_clips):
        vid_reshape_emb[i, :end_place, :] = vid_emb[i].expand(1, end_place, vid_emb.shape[1])

    for i, end_place in enumerate(num_caps):
        para_reshape_emb[i, :end_place, :] = para_emb[i,:].expand(1, end_place, para_emb.shape[1])

    clip_emb  = self.vid_seq_dec(vid_reshape_emb, Variable(torch.Tensor(num_clips)))
    sent_emb  = self.txt_seq_dec(para_reshape_emb, Variable(torch.Tensor(num_caps)))

    return clip_emb, sent_emb

  def lowest_reconstruct_emb(self, vid_emb, para_emb, num_clips, num_caps):
    vid_reshape_emb = Variable(torch.zeros(len(num_clips), max(num_clips), vid_emb.shape[1])).cuda()
    para_reshape_emb = Variable(torch.zeros(len(num_caps),  max(num_caps),  para_emb.shape[1])).cuda()

    for i, end_place in enumerate(num_clips):
        vid_reshape_emb[i, :end_place, :] = vid_emb[i].view(1,1,-1).expand(1, end_place, vid_emb.shape[1])

    for i, end_place in enumerate(num_caps):
        para_reshape_emb[i, :end_place, :] = para_emb[i,:].view(1,1,-1).expand(1, end_place, para_emb.shape[1])

    frame_emb  = self.clip_seq_dec(vid_reshape_emb, Variable(torch.Tensor(num_clips)))
    word_emb = self.sent_seq_dec(para_reshape_emb, Variable(torch.Tensor(num_caps)))

    return frame_emb, word_emb

  def forward_loss(self, clip_emb, cap_emb, name, **kwargs):
    """Compute the loss given pairs of image and caption embeddings
    """
    loss = self.criterion(clip_emb, cap_emb)
    self.logger.update('Le'+name, loss.item(), clip_emb.size(0))
    return loss

  def forward_weak_loss(self, clip_emb, cap_emb, num_clips, num_caps, name, **kwargs):
    """Compute the loss given pairs of image and caption embeddings
    """
    loss = self.weak_criterion(clip_emb, cap_emb, num_clips, num_caps)
    self.logger.update('Le'+name, loss.item(), clip_emb.size(0))
    return loss

  def forward_reconstruct_loss(self, clip_recon, clip_emb, name, **kwargs):
    """Compute the loss given pairs of image and caption embeddings
    """
    loss = self.criterion_Euclid_Distance(clip_recon, clip_emb)
    self.logger.update('Le'+name, loss.item(), clip_emb.size(0))
    return loss


  def train_emb(self, opts, clips, captions, videos, paragraphs,
      lengths_clip, lengths_cap, lengths_video, lengths_paragraph,
      num_clips, num_caps, ind, cur_vid, *args):
    """One training step given clips and captions.
    """
    self.Eiters += 1
    self.logger.update('Eit', self.Eiters)
    self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

    # compute the embeddings
    clip_emb, cap_emb, word = self.forward_emb(clips, captions, lengths_clip, lengths_cap, return_word=True)
    vid_context, para_context = self.forward_emb(videos, paragraphs, lengths_video, lengths_paragraph)
    vid_emb, para_emb = self.structure_emb(clip_emb, cap_emb, num_clips, num_caps, vid_context, para_context)

    if opts.reconstruct_loss:
        clip_recon, cap_recon = self.reconstruct_emb(vid_emb, para_emb, num_clips, num_caps)
    if opts.lowest_reconstruct_loss:
        frame_recon, sent_recon = self.lowest_reconstruct_emb(clip_recon, cap_recon, lengths_clip.numpy(), lengths_cap.numpy())

    # measure accuracy and record loss
    self.optimizer.zero_grad()

    loss = 0

    loss_1 = self.forward_loss(F.normalize(vid_emb), F.normalize(para_emb), '_vid')
    loss_3 = self.forward_loss(F.normalize(vid_context), F.normalize(para_context), '_ctx_low_lvel')
    loss_5 = (self.forward_loss(F.normalize(vid_emb), F.normalize(vid_emb), '_vid_inloss') + self.forward_loss(F.normalize(para_emb), F.normalize(para_emb), '_para_inloss'))/2
    loss = loss + loss_1 + loss_3 + loss_5

    if opts.low_level_loss:
        if opts.weak_low_level_loss:
            loss_2 = self.forward_weak_loss(F.normalize(clip_emb), F.normalize(cap_emb), num_clips, num_caps, '_wlow_lvel')
        else:
            loss_2 = self.forward_loss(F.normalize(clip_emb), F.normalize(cap_emb), '_low_lvel')
        loss_6 = (self.forward_loss(F.normalize(clip_emb), F.normalize(clip_emb), '_clip_inloss') + self.forward_loss(F.normalize(cap_emb), F.normalize(cap_emb), '_cap_inloss'))/2
        loss = loss + loss_2 + loss_6

    if opts.reconstruct_loss:
        loss_recon = (self.forward_reconstruct_loss(clip_recon, clip_emb.detach(), '_clip_recon') + self.forward_reconstruct_loss(cap_recon, cap_emb.detach(), '_cap_recon'))
        loss = loss + loss_recon * opts.weight_recon

    if opts.lowest_reconstruct_loss:
        clips_var = torch.zeros(lengths_clip.sum().item(), opts.img_dim)
        curpos = 0
        for i in range(clips.shape[0]):
            clips_var[curpos: curpos+lengths_clip[i],:] = clips[i,0:lengths_clip[i],:]
            curpos = curpos + lengths_clip[i]

        words_var = Variable(torch.zeros(lengths_cap.sum().item(), 300)).cuda()
        curpos = 0
        for i in range(captions.shape[0]):
            words_var[curpos: curpos+lengths_cap[i],:] = word[i,0:lengths_cap[i],:]
            curpos = curpos + lengths_cap[i]

        loss_lowest_recon = self.forward_reconstruct_loss(frame_recon, Variable(clips_var).cuda().detach(), '_reconstruct_frame_hier') + self.forward_reconstruct_loss(sent_recon, words_var.detach(), '_reconstruct_word_hier')
        loss = loss + loss_lowest_recon * opts.lowest_weight_recon

    # compute gradient and do SGD step
    loss.backward()
    if self.grad_clip > 0: clip_grad_norm(self.params, self.grad_clip)
    self.optimizer.step()
