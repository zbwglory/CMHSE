import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import time
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import h5py
import copy
from IPython import embed
from json_data import json_data
import os.path as osp

this_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))
unit_test = jsonmod.load(open('didemo_dev/train_data_bwzhang.json','r'))
def prepare_videos(vidinfo, seg_num, sample_rate=5):
  cur_featname = vidinfo.feat_filename.split('/')[-1]
  slot = []
  for i in range(seg_num):
    start_frame = vidinfo.img_length / vidinfo.vid_length * sample_rate * i
    end_frame = min(vidinfo.img_length / vidinfo.vid_length * sample_rate * (i+1), vidinfo.img_length)
    start_feat_frame = int(start_frame )
    end_feat_frame = int(end_frame )
    slot.append((start_feat_frame,end_feat_frame))

  return slot, cur_featname

class PrecompDataset(data.Dataset):
  def __init__(self, data_path, data_split, vocab, opt):
    self.vocab = vocab
    self.json_compiled = json_data(data_split)
    self.ann_id = {}
    for i, keys in enumerate(self.json_compiled.keys()):
      self.ann_id[i] = str(keys)
    self.length = len(self.ann_id)
    self.h5file  = h5py.File(osp.join(this_dir, 'data', 'didemo_incep_v3.h5'),'r', swmr=True)
    self.vidinfo = np.load(osp.join(this_dir, 'data', 'didemo', '{}_vidinfo_didemo.npz'.format(data_split)))['vidinfo'].item()

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    # handle the image redundancy
    cur_vid = self.ann_id[index]
    cur_vidinfo = self.vidinfo[cur_vid]
    cur_jsoninfo = self.json_compiled[cur_vid]
    slot, cur_featname = prepare_videos(cur_vidinfo, cur_jsoninfo['num_segments'])
    embed()
    feat_data = self.h5file[cur_featname].value
    feat_data = torch.Tensor(feat_data)
    captions = self.json_compiled[cur_vid]['description']
    # start_t = time.time()
    clips, captions, videos, paragraphs, lengths_clip, lengths_cap, num_clip, num_caption = self.img_cap_feat_combine(feat_data, slot, captions, cur_vid)
    # print("combined feature time: %.2f" % (time.time() - start_t) )
    return clips, captions, videos, paragraphs, lengths_clip, lengths_cap, num_clip, num_caption, index

  def img_cap_feat_combine(self, feats, slot, captions, cur_vid, max_frames=140.0):
    #### Videos ####
    #start_t = time.time()
    clips = []
    len_feat = feats.shape[0]
    for i, seg_id in enumerate(self.json_compiled[cur_vid]['times']):
      start_time = seg_id[0]
      end_time = seg_id[1]
      slot_start = slot[start_time][0]
      slot_end = slot[end_time][1]
      if slot_start <= len_feat and slot_end <= len_feat:
        cur_feat = feats[slot_start: slot_end]
      else:
        print (cur_vid, slot_start, slot_end, len_feat)
        cur_feat = feats
      cur_length = cur_feat.shape[0]
      if cur_length > max_frames:
        ind = np.arange(0, cur_feat.shape[0], cur_feat.shape[0] / max_frames).astype(int).tolist()
        cur_feat = cur_feat[ind, :]
      clips.append(cur_feat)
    # for feat in feats:
    #   cur_feat = torch.Tensor(feat)
    #   cur_length = cur_feat.shape[0]
    #   if cur_length > max_frames:
    #     ind = np.arange(0, cur_feat.shape[0], cur_feat.shape[0] / max_frames).astype(int).tolist()
    #     cur_feat = cur_feat[ind, :]
    #   clips.append(cur_feat)

    lengths_clip = [len(vid) for vid in clips]
    if len_feat > max_frames:
      ind = np.arange(0, len_feat, len_feat / max_frames).astype(int).tolist()
      videos = feats[ind, :]
    else:
      videos = feat
    #print("video feature time: %.2f" % (time.time() - start_t))

    #### Captions ####
    vocab = self.vocab
    captions = []
    # start_t = time.time()
    for cap in self.json_compiled[cur_vid]['description']:
      tokens_cap = nltk.tokenize.word_tokenize(cap.lower())
      caption = []
      caption.extend([vocab(token) for token in tokens_cap])
      target_cap = torch.Tensor(caption)
      captions.append(target_cap)

    lengths_cap = [len(cap) for cap in captions]
    lengths_clip = [len(vid) for vid in clips]

    num_clip    = len(clips)
    num_caption = len(captions)
    assert num_clip == num_caption, 'something is wrong: {} vs. {}'.format(num_clip, num_caption)

    lengths_clip = torch.Tensor(lengths_clip).long()
    lengths_cap = torch.Tensor(lengths_cap).long()
    paragraphs = torch.cat(captions, 0)
    # print("caption feature time: %.2f" % (time.time() - start_t))
    return clips, captions, videos, paragraphs, lengths_clip, lengths_cap, num_clip, num_caption

def collate_fn(data_batch):
  _clips, _captions, _videos, _paragraphs, _lengths_clip, _lengths_cap, num_clips, num_captions, index = zip(*data_batch)

  #start_t = time.time()
  lengths_clip = torch.cat(_lengths_clip, 0)
  clips = torch.zeros(len(lengths_clip), lengths_clip.max(), 2048)
  _cur_ind = 0
  for i, _vid_seg in enumerate(_clips):
    for j, vid in enumerate(_vid_seg):
      end = lengths_clip[_cur_ind]
      clips[_cur_ind, :end, :] = vid[:end, :]
      _cur_ind = _cur_ind + 1

  lengths_video = torch.Tensor([len(x) for x in _videos]).long()
  videos = torch.zeros(len(_videos), lengths_video.max(), 2048)
  for i, vid in enumerate(_videos):
    end = lengths_video[i]
    videos[i, :end, :] = vid[:end, : ]

  #print("video collate time: %.2f" % (time.time() - start_t))

  lengths_cap = torch.cat(_lengths_cap, 0)
  captions = torch.zeros(len(lengths_cap), lengths_cap.max()).long()
  _cur_ind = 0
  for i, _cap_seg in enumerate(_captions):
    for j, cap in enumerate(_cap_seg):
      end = lengths_cap[_cur_ind]
      captions[_cur_ind, :end] = cap[:end]
      _cur_ind = _cur_ind + 1

  lengths_paragraph = torch.Tensor([len(x) for x in _paragraphs]).long()
  paragraphs = torch.zeros(len(_paragraphs), lengths_paragraph.max()).long()
  for i, cap in enumerate(_paragraphs):
    end = lengths_paragraph[i]
    paragraphs[i, :end] = cap[:end ]

  return clips, captions, videos, paragraphs, lengths_clip, lengths_cap, lengths_video, lengths_paragraph, num_clips, num_captions, index

def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
       shuffle=True, num_workers=2):
  """Returns torch.utils.data.DataLoader for custom coco dataset."""
  dset = PrecompDataset(data_path, data_split, vocab, opt)

  data_loader = torch.utils.data.DataLoader(dataset=dset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            collate_fn=collate_fn)
  return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
  dpath = os.path.join(opt.data_path, data_name)
  if opt.data_name.endswith('_precomp'):
    train_loader = get_precomp_loader(dpath, 'train_data', vocab, opt,
            batch_size, True, workers)
    val_loader   = get_precomp_loader(dpath, 'val_data', vocab, opt,
          batch_size, False, workers)
  return train_loader, val_loader

def get_test_loader(split_name, data_name, vocab, batch_size,
      workers, opt):
  dpath = os.path.join(opt.data_path, data_name)
  if opt.data_name.endswith('_precomp'):
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
           batch_size, False, workers)
  return test_loader
