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

class Seq2Seq_Decode(nn.Module):
  def __init__(self, embedding_features, rnn_features, rnn_bidirectional=False):
    super(Seq2Seq_Decode, self).__init__()
    self.bidirectional = rnn_bidirectional
    print embedding_features, rnn_features

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

  def forward(self, q_emb, q_len, hidden=None):
    lengths = q_len.long()
    lens, indices = torch.sort(lengths, 0, True)

    packed_batch = pack_padded_sequence(q_emb[indices.cuda()], lens.tolist(), batch_first=True)
    if hidden is not None:
      N_, H_ = hidden.size()
      hidden, _ = self.rnn(packed_batch, hidden[indices.cuda()].view(1, N_, H_))
    else:
      hidden, _ = self.rnn(packed_batch)
    hidden_depack = pad_packed_sequence(hidden, True)[0]

    if self.bidirectional:
        NotImplemented

    _, _indices = torch.sort(indices, 0)
    hidden_depack = hidden_depack[_indices.cuda()]

    return hidden_depack


