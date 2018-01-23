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

class DecoderSequence(nn.Module):
  def __init__(self, img_dim, embed_size, dropout=0,
                     no_imgnorm=False, bidirectional=False, rnn_type='seq2seq'):
    super(DecoderSequence, self).__init__()
    self.embed_size = embed_size
    self.no_imgnorm = no_imgnorm
    self.bidirectional = bidirectional

    num_layers = 1
    if dropout > 0:
        self.dropout = nn.Dropout(dropout)
    if rnn_type == 'seq2seq':
      self.rnn = Seq2Seq_Decode(img_dim, embed_size, rnn_bidirectional=bidirectional)
    else:
      raise ValueError('Unsupported RNN type')

  def forward(self, x, lengths):
    """Extract image feature vectors."""
    outputs = self.rnn(x, lengths)
    lengths = lengths.numpy().astype(int)
    sum_total = sum(lengths)
#    print (x.shape, sum_total)
    outputs_reshape = Variable(torch.zeros(sum_total,x.shape[2])).cuda()
#    print outputs_reshape.shape
    pos = 0
    for i,leng in enumerate(lengths):
        outputs_reshape[pos:pos+leng,:] = outputs_reshape[pos:pos+leng,:]+outputs[i,0:leng,:]
        pos = pos + leng

    # normalization in the joint embedding space
    return F.normalize(outputs_reshape)

