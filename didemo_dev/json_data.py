import os
import numpy as np
import json as jsonmod
import h5py
from IPython import embed
import os.path as osp
import torch

this_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))

def __major_vote(times, total_seg):
  time_slot = torch.zeros(total_seg)
  for time in times:
    time_slot[time[0]:time[1]+1] = time_slot[time[0]:time[1]+1] + 1
  tmax = time_slot.max()
  time_max = []
  for i, t in enumerate(time_slot):
    if t == tmax:
      time_max.append(i)
  output = [min(time_max), max(time_max)]

  return output

def __pre_compile_dataset(jsondict):
  compile_dict = {}
  for i in range(len(jsondict)):
    cur_vid = jsondict[i]
    cur_vid_name = cur_vid['video']
    cur_vid_num_seg = cur_vid['num_segments']
    if cur_vid_name not in compile_dict:
      compile_dict[cur_vid_name] = {}
      compile_dict[cur_vid_name]['num_segments'] = cur_vid_num_seg
      compile_dict[cur_vid_name]['times'] = []
      compile_dict[cur_vid_name]['times'].append(__major_vote(cur_vid['times'], cur_vid_num_seg))
      compile_dict[cur_vid_name]['description'] = []
      compile_dict[cur_vid_name]['description'].append(cur_vid['description'])
    else:
      compile_dict[cur_vid_name]['times'].append(__major_vote(cur_vid['times'], cur_vid_num_seg))
      compile_dict[cur_vid_name]['description'].append(cur_vid['description'])

  for key in compile_dict.keys():
    start_time = [x[0] for x in compile_dict[key]['times']]
    sort_id = np.argsort(start_time)
    compile_dict[key]['times'] = np.array(compile_dict[key]['times'])[sort_id].tolist()
    compile_dict[key]['description'] = np.array(compile_dict[key]['description'])[sort_id].tolist()

  return compile_dict

def json_data(data_split):
  jsondict = jsonmod.load(open('{}/{}.json'.format(osp.join(this_dir, 'data', 'didemo'), data_split), 'r'))

  json_compiled = __pre_compile_dataset(jsondict)

  return json_compiled

