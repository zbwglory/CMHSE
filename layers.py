import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn.functional as F
from IPython import embed

class GroupMLP(nn.Module):
  def __init__(self, in_features, mid_features, out_features, drop=0.5, groups=1):
    super(GroupMLP, self).__init__()

    self.conv1 = nn.Conv1d(in_features, mid_features, 1)
    self.drop  = nn.Dropout(p=drop)
    self.relu  = nn.ReLU()
    self.conv2 = nn.Conv1d(mid_features, out_features, 1, groups=groups)

  def forward(self, a):
    N, C = a.size()
    h = self.relu(self.conv1(a.view(N, C, 1)))
    return self.conv2(self.drop(h)).view(N, -1)

class Seq2Seq(nn.Module):
  def __init__(self, embedding_features, rnn_features, rnn_bidirectional=False):
    super(Seq2Seq, self).__init__()
    self.bidirectional = rnn_bidirectional

    self.rnn = nn.GRU(input_size=embedding_features,
              hidden_size=rnn_features,
              num_layers=1, batch_first=True,
              bidirectional=rnn_bidirectional)

    self.features = rnn_features

    self._init_rnn(self.rnn.weight_ih_l0)
    self._init_rnn(self.rnn.weight_hh_l0)
    self.rnn.bias_ih_l0.data.zero_()
    self.rnn.bias_hh_l0.data.zero_()

  def _init_rnn(self, weight):
    for w in weight.chunk(3, 0):
      init.xavier_uniform(w)

  def forward(self, q_emb, q_len):
    lengths = q_len.long()
    lens, indices = torch.sort(lengths, 0, True)

    packed = pack_padded_sequence(q_emb[indices.cuda()], lens.tolist(), batch_first=True)
    _, outputs = self.rnn(packed)

    if self.bidirectional:
      outputs = torch.cat([ outputs[0, :, :], outputs[1, :, :] ], dim=1)
    else:
      outputs = outputs.squeeze(0)

    _, _indices = torch.sort(indices, 0)
    outputs = outputs[_indices.cuda()]

    return outputs

class Attention(nn.Module):
  def __init__(self, embedding_features, rnn_features, rnn_bidirectional=False):
    super(Attention, self).__init__()
    self.register_buffer("mask",torch.FloatTensor())
    hid_size = rnn_features
    natt = rnn_features

    self.rnn = nn.GRU(input_size=embedding_features,
                    hidden_size=rnn_features,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=rnn_bidirectional)
    self.lin = nn.Linear(hid_size,natt)
    self.att_w = nn.Linear(natt,1,bias=False)
    self.tanh = nn.Tanh()

    self._init_rnn(self.rnn.weight_ih_l0)
    self._init_rnn(self.rnn.weight_hh_l0)
    self.rnn.bias_ih_l0.data.zero_()
    self.rnn.bias_hh_l0.data.zero_()

  def _init_rnn(self, weight):
    for w in weight.chunk(3, 0):
      init.xavier_uniform(w)

  def forward(self, q_emb, q_len):
    lengths = q_len.long()
    lens, indices = torch.sort(lengths, 0, True)

    packed_batch = pack_padded_sequence(q_emb[indices.cuda()], lens.tolist(), batch_first=True)

    hs, _ = self.rnn(packed_batch)
    enc_sents, len_s = pad_packed_sequence(hs, batch_first=True)

    emb_h = self.tanh(self.lin(enc_sents.contiguous().view(enc_sents.size(0)*enc_sents.size(1),-1)))  # Nwords * Emb
    attend = self.att_w(emb_h).view(enc_sents.size(0),
                                    enc_sents.size(1))
    mask = self._list_to_bytemask(list(len_s))
    all_att = self._masked_softmax(attend, mask)
    try:
      attended = all_att.unsqueeze(2).expand_as(enc_sents) * enc_sents
    except:
      embed()
      raise

    _, _indices = torch.sort(indices, 0)
    return attended.sum(1,True).squeeze(1)[_indices.cuda()]

  def forward_att(self, q_emb, q_len):
    lengths = q_len.long()
    lens, indices = torch.sort(lengths, 0, True)

    packed_batch = pack_padded_sequence(q_emb[indices.cuda()], lens.tolist(), batch_first=True)

    hs, _ = self.rnn(packed_batch)
    enc_sents, len_s = pad_packed_sequence(hs, batch_first=True)

    _, _indices = torch.sort(indices, 0)
    enc_sents = rnn_sents[_indices.cuda()]

    emb_h = self.tanh(self.lin(enc_sents.contiguous().view(enc_sents.size(0)*enc_sents.size(1),-1)))  # Nwords * Emb
    attend = self.att_w(emb_h).view(enc_sents.size(0),
                                    enc_sents.size(1))
    mask = self._list_to_bytemask(list(lens.tolist))
    all_att = self._masked_softmax(attend, mask)
    try:
      attended = all_att.unsqueeze(2).expand_as(enc_sents) * enc_sents
    except:
      embed()
      raise

    _, _indices = torch.sort(indices, 0)
    return attended.sum(1,True).squeeze(1)[_indicies.cuda()], all_att

  def _list_to_bytemask(self,l):
    mask = self._buffers['mask'].resize_(len(l),l[0]).fill_(1)

    for i,j in enumerate(l):
      if j != l[0]:
        mask[i,j:l[0]] = 0

    return mask

  def _masked_softmax(self,mat,mask):
    exp = torch.exp(mat) * Variable(mask,requires_grad=False)
    sum_exp = exp.sum(1,True)+0.0001

    return exp/sum_exp.expand_as(exp)

class Maxout(nn.Module):
  def __init__(self, embedding_features, rnn_features, rnn_bidirectional=False):
    super(Maxout, self).__init__()
    self.bidirectional = rnn_bidirectional

    self.rnn = nn.GRU(input_size=embedding_features,
              hidden_size=rnn_features,
              num_layers=1, batch_first=True,
              bidirectional=False)

    self.features = rnn_features

    self._init_rnn(self.rnn.weight_ih_l0)
    self._init_rnn(self.rnn.weight_hh_l0)
    self.rnn.bias_ih_l0.data.zero_()
    self.rnn.bias_hh_l0.data.zero_()

  def _init_rnn(self, weight):
    for w in weight.chunk(3, 0):
      init.xavier_uniform(w)

  def forward(self, q_emb, q_len):
    lengths = q_len.long()
    lens, indices = torch.sort(lengths, 0, True)

    packed = pack_padded_sequence(q_emb[indices.cuda()], lens.tolist(), batch_first=True)
    hs, _ = self.rnn(packed)
    outputs, _ = pad_packed_sequence(hs, batch_first=True)

    _, _indices = torch.sort(indices, 0)
    outputs = outputs[_indices.cuda()]
    N, L, H = outputs.size()
    maxout = []
    for batch_ind, length in enumerate(lengths.tolist()):
      maxout.append( F.max_pool1d(outputs[batch_ind, :length, :].view(1, length, H).transpose(1, 2), length).squeeze().view(1, -1) )

    return torch.cat(maxout)
