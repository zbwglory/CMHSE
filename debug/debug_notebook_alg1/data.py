import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
import h5py

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = '/data2/bwzhang/anet_img/captions/'

        # Captions
        self.jsondict = jsonmod.load(open(loc+'{}.json'.format(data_split), 'r'))
        self.ann_id = {}
        for i, keys in enumerate(self.jsondict.keys()):
            self.ann_id[i] = keys

        # Image features
        self.video_emb = h5py.File('/home/bwzhang/sub_activitynet_v1-3.c3d.hdf5')

        self.length = len(self.ann_id)

    def img_feat_combine(self, c3d_feat, cur_vid):
        frame_duration = self.jsondict[cur_vid]['duration']
        segment_number = len(self.jsondict[cur_vid]['timestamps'])
        segment_rand_pick_ind = torch.randperm(segment_number)[0]
        picked_segment = self.jsondict[cur_vid]['timestamps'][segment_rand_pick_ind]
        c3d_feat_len = c3d_feat.shape[0]
        start_frame = int(np.floor(picked_segment[0] / frame_duration * c3d_feat_len))
        end_frame = int(np.ceil(picked_segment[1] / frame_duration * c3d_feat_len))

        if start_frame > end_frame:
            #one video is misaligned
            end_frame = start_frame

        picked_c3d_feat = c3d_feat[start_frame:end_frame+1, :]

        picked_c3d_feat_mean = picked_c3d_feat.mean(0)

        video_c3d_feat_mean = c3d_feat.mean(0)

        video_repre = torch.cat((video_c3d_feat_mean, picked_c3d_feat_mean), 0)

        return video_repre


    def __getitem__(self, index):
        # handle the image redundancy
        cur_vid = self.ann_id[index]

        c3d_feat = torch.Tensor(self.video_emb[cur_vid]['c3d_features'].value)
        image = self.img_feat_combine(c3d_feat, cur_vid)
        captions = self.jsondict[cur_vid]['sentences']
        caption = ''
        for cap in captions:
            caption = caption+cap
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            caption.lower())
        caption = []
        caption.append(vocab('BOS'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('EOS'))
        target = torch.Tensor(caption)


        return image, target, index, cur_vid

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


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
        val_loader = get_precomp_loader(dpath, 'val_1', vocab, opt,
                                        batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith('_precomp'):
        test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                         batch_size, False, workers)
    return test_loader
