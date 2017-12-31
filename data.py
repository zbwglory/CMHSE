import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import h5py
import copy
from IPython import embed

class PrecompDataset(data.Dataset):
  """
  Load precomputed captions and image features
  Possible options: f8k, f30k, coco, 10crop
  """

  def __init__(self, data_path, data_split, vocab):
    self.vocab = vocab
    loc = './data/activitynet/captions/'

    # Captions
    self.jsondict = jsonmod.load(open(loc+'{}.json'.format(data_split), 'r'))
    self.ann_id = {}
    for i, keys in enumerate(self.jsondict.keys()):
      self.ann_id[i] = keys

    # Image features
    self.video_emb = h5py.File('./data/sub_activitynet_v1-3.c3d.hdf5')

    self.length = len(self.ann_id)

  def img_cap_feat_combine(self, c3d_feat, captions, cur_vid):

    #### Videos ####
    frame_duration = self.jsondict[cur_vid]['duration']
    segment_number = len(self.jsondict[cur_vid]['timestamps'])
    c3d_feat_len = c3d_feat.shape[0]

    video_segment_c3d = []

    for seg_id in range(segment_number):
        picked_segment = self.jsondict[cur_vid]['timestamps'][seg_id]
        start_frame = int(np.floor(picked_segment[0] / frame_duration * c3d_feat_len))
        end_frame = int(np.ceil(picked_segment[1] / frame_duration * c3d_feat_len))

        if start_frame > end_frame:
            #one video is misaligned
            end_frame = start_frame

        c3d_cur_feat = c3d_feat[start_frame: end_frame+1, :]
        max_frames = 140.0
        if c3d_cur_feat.shape[0] > max_frames:
          ind = np.arange(0, c3d_cur_feat.shape[0], c3d_cur_feat.shape[0]/max_frames).astype(int).tolist()
          c3d_cur_feat = c3d_cur_feat[ind,:]

        video_segment_c3d.append(c3d_cur_feat)

    lengths_vid = [len(vid) for vid in video_segment_c3d]
#    video_segment_torch = torch.zeros(len(video_segment_c3d), max(lengths_vid), 500)
#    for i, vid in enumerate(video_segment_c3d):
#      end = lengths_vid[i]
#      video_segment_torch[i, :end, :] = vid[:end, :]

    c3d_whole_feat = c3d_feat
    if c3d_whole_feat.shape[0] > max_frames:
      ind = np.arange(0, c3d_whole_feat.shape[0], c3d_whole_feat.shape[0]/max_frames).astype(int).tolist()
      c3d_whole_feat = c3d_whole_feat[ind,:]



    #### Captions ####

    vocab = self.vocab

    caption_segment = []
    length_cap = []

    for cap in captions:
        tokens_cap = nltk.tokenize.word_tokenize(cap.lower()) 
        caption = []
#        caption.append(vocab('BOS'))
        caption.extend([vocab(token) for token in tokens_cap])
#        caption.append(vocab('EOS'))
        target_cap = torch.Tensor(caption)
        caption_segment.append(target_cap)

    lengths_cap = [len(cap) for cap in caption_segment]
#    caption_segment_torch = torch.zeros(len(caption_segment), max(lengths_cap)).long()
#    for i, cap in enumerate(caption_segment):
#      end = lengths_cap[i]
#      caption_segment_torch[i, :end] = cap[:end]

    seg_num = len(video_segment_c3d)

    lengths_vid = torch.Tensor(lengths_vid).long()
    lengths_cap = torch.Tensor(lengths_cap).long()

    cap_whole_feat = torch.cat(caption_segment, 0)

    return video_segment_c3d, caption_segment, c3d_whole_feat, cap_whole_feat, lengths_vid, lengths_cap, seg_num



  def __getitem__(self, index):
    # handle the image redundancy
    cur_vid = self.ann_id[index]
    image_data = self.video_emb[cur_vid]['c3d_features'].value
    image = torch.Tensor(image_data)
    captions = self.jsondict[cur_vid]['sentences']

    video_segment_c3d, caption_segment, c3d_whole_feat, cap_whole_feat, lengths_vid, lengths_cap, seg_num = self.img_cap_feat_combine(image, captions, cur_vid)

    return video_segment_c3d, caption_segment, c3d_whole_feat, cap_whole_feat, lengths_vid, lengths_cap, index, seg_num

  def __len__(self):
    return self.length

def collate_fn(data_batch):
  _videos, _captions, _c3d_whole_feat, _cap_whole_feat, _lengths_small_vid, _lengths_small_cap, _ind, _seg_num = zip(*data_batch)

  # Merge images
  lengths_vid = torch.cat(_lengths_small_vid, 0)
  videos = torch.zeros(len(lengths_vid), lengths_vid.max(), 500)
  _cur_ind = 0
  for i, _vid_seg in enumerate(_videos):
      for j, vid in enumerate(_vid_seg):
          end = lengths_vid[_cur_ind]
          videos[_cur_ind, :end, :] = vid[:end, :]
          _cur_ind = _cur_ind + 1

  lengths_whole_vid = torch.Tensor([len(x) for x in _c3d_whole_feat]).long()
  videos_whole = torch.zeros(len(_c3d_whole_feat), lengths_whole_vid.max(), 500)
  for i, vid in enumerate(_c3d_whole_feat):
      end = lengths_whole_vid[i]
      videos_whole[i, :end, :] = vid[:end, : ]

  # Merget captions
  lengths_cap = torch.cat(_lengths_small_cap, 0)
  captions = torch.zeros(len(lengths_cap), lengths_cap.max()).long()
  _cur_ind = 0
  for i, _cap_seg in enumerate(_captions):
      for j, cap in enumerate(_cap_seg):
          end = lengths_cap[_cur_ind]
          captions[_cur_ind, :end] = cap[:end]
          _cur_ind = _cur_ind + 1

  lengths_whole_cap = torch.Tensor([len(x) for x in _cap_whole_feat]).long()
  captions_whole = torch.zeros(len(_cap_whole_feat), lengths_whole_cap.max()).long()
  for i, cap in enumerate(_cap_whole_feat):
      end = lengths_whole_cap[i]
      captions_whole[i, :end] = cap[:end ]


  return videos, captions, videos_whole, captions_whole, lengths_vid, lengths_cap, lengths_whole_vid, lengths_whole_cap, _ind, _seg_num

def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
             shuffle=True, num_workers=2):
  """Returns torch.utils.data.DataLoader for custom coco dataset."""
  dset = PrecompDataset(data_path, data_split, vocab)

  data_loader = torch.utils.data.DataLoader(dataset=dset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True,
                        collate_fn=collate_fn)
  return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
  dpath = os.path.join(opt.data_path, data_name)
  if opt.data_name.endswith('_precomp'):
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                      batch_size, True, workers)
    val_loader   = get_precomp_loader(dpath, 'val_1', vocab, opt,
                    batch_size, False, workers)
  return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
          workers, opt):
  dpath = os.path.join(opt.data_path, data_name)
  if opt.data_name.endswith('_precomp'):
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                     batch_size, False, workers)
  return test_loader
