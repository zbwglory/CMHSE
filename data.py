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

class PrecompDataset(data.Dataset):
  """
  Load precomputed captions and image features
  Possible options: f8k, f30k, coco, 10crop
  """

  def __init__(self, data_path, data_split, vocab):
    self.vocab = vocab
    loc = '/data1/hexianghu/activitynet/captions/'

    # Captions
    self.jsondict = jsonmod.load(open(loc+'{}.json'.format(data_split), 'r'))
    self.ann_id = {}
    for i, keys in enumerate(self.jsondict.keys()):
      self.ann_id[i] = keys

    # Image features
    self.video_emb = h5py.File('sub_activitynet_v1-3.c3d.hdf5')

    self.length = len(self.ann_id)


  def __getitem__(self, index):
    # handle the image redundancy
    cur_vid = self.ann_id[index]
    image_data = self.video_emb[cur_vid]['c3d_features'].value
    max_frames = 600.0
    if image_data.shape[0] > max_frames:
      ind = np.arange(0, image_data.shape[0], image_data.shape[0]/max_frames).astype(int).tolist()
      image_data = image_data[ind,:]

    image = torch.Tensor(image_data)
    captions = self.jsondict[cur_vid]['sentences']
    # caption  = captions[segment_rand_pick_ind]
    caption = ' '.join(captions)
    vocab = self.vocab

    # Convert caption (string) to word ids.
    tokens = nltk.tokenize.word_tokenize(
      caption.lower())
    caption = []
    caption.append(vocab('BOS'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('EOS'))
    target = torch.Tensor(caption)

    return image, target

  def __len__(self):
    return self.length

def collate_fn(data_batch):
  _images, _captions = zip(*data_batch)

  # Merge images
  lengths_img = [len(img) for img in _images]
  images = torch.zeros(len(_images), max(lengths_img), 500)
  for i, img in enumerate(_images):
    end = lengths_img[i]
    images[i, :end, :] = img[:end, :]

  # Merget captions
  lengths_cap = [len(cap) for cap in _captions]
  captions = torch.zeros(len(_captions), max(lengths_cap)).long()
  for i, cap in enumerate(_captions):
    end = lengths_cap[i]
    captions[i, :end] = cap[:end]

  return images, captions, lengths_img, lengths_cap

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
