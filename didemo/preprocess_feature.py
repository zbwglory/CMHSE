import subprocess

import datetime
import time
from IPython import embed
import numpy as np
import torch

import os

import h5py
import numpy as np
import os
from IPython import embed
import json as jsonmod


class VideoInformation(object):
  def __init__(self, vid_path, img_path, feat_path, video_name):
    self.vid_filename = os.path.join(vid_path, video_name)
    self.img_filename = os.path.join(img_path, video_name.split('.')[0])
    self.feat_filename = os.path.join(feat_path, video_name.split('.')[0]+'.npz')

    self.vid_length = self.__getLength_mix(self.vid_filename)
    self.img_length = len(os.listdir(self.img_filename))

  def __getLength(self, filename):
    result = subprocess.Popen(["ffprobe", filename],
        stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    return [x for x in result.stdout.readlines() if "Duration" in x]

  def __string2time(self, instr):
    x = time.strptime(instr.split('.')[0], '%H:%M:%S')
    out = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
    out = out + float('0.'+instr.split('.')[1])
    return out

  def __getLength_mix(self, filename):
    mixstr = self.__getLength(filename)
    mixstr = self.__getLength(filename)[0]
    mixstr = mixstr.split('Duration:')[1]
    mixstr = mixstr.split(', start:')[0]
    mixstr = mixstr[1:]
    output = self.__string2time(mixstr)
    return output

def main():
  a = DatasetInformation('train_data')

if __name__ == "__main__":
  main()


