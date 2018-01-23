import os
import numpy as np
import json as jsonmod
import h5py
from IPython import embed
import os.path as osp
import torch
import subprocess
import time
import datetime

this_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))

def __getLength( filename):
    result = subprocess.Popen(["ffprobe", filename],
        stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    return [x for x in result.stdout.readlines() if "Duration" in x]

def __string2time( instr):
    x = time.strptime(instr.split('.')[0], '%H:%M:%S')
    out = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
    out = out + float('0.'+instr.split('.')[1])
    return out

def __getLength_mix(filename):
    mixstr = __getLength(filename)[0]
    mixstr = mixstr.split('Duration:')[1]
    mixstr = mixstr.split(', start:')[0]
    mixstr = mixstr[1:]
    output = __string2time(mixstr)
    return output


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

  # add information to the dict
  compile_dict = {}
  for i in range(len(jsondict)):
    cur_vid = jsondict[i]
    cur_vid_name = cur_vid['video'].split('.')[0]
    cur_vid_num_seg = cur_vid['num_segments']
    if cur_vid_name not in compile_dict:
      compile_dict[cur_vid_name] = {}
      compile_dict[cur_vid_name]['duration'] = __getLength_mix('/data1/bwzhang/research/data/LocalizingMoments/videos/'+cur_vid['video'])
      feat_leng = np.load('/data2/bwzhang/research/data/LocalizingMoments/ICEP_V3_global_pool_no_skip_direct_resize/'+cur_vid_name+'.npz')['frame_scores'].shape[0]
      compile_dict[cur_vid_name]['feature_leng'] = feat_leng
      compile_dict[cur_vid_name]['num_segments'] = cur_vid_num_seg
      compile_dict[cur_vid_name]['times'] = []
      frame_per_sec = compile_dict[cur_vid_name]['feature_leng'] / compile_dict[cur_vid_name]['duration']
      time_count = __major_vote(cur_vid['times'], cur_vid_num_seg)
      time_count = [int(time_count[0] * 5), min(int((time_count[1]+1)*5), compile_dict[cur_vid_name]['duration'])]
      time_count = [int(time_count[0] * frame_per_sec), int(time_count[1] * frame_per_sec)]
      compile_dict[cur_vid_name]['times'].append(time_count)
      compile_dict[cur_vid_name]['description'] = []
      compile_dict[cur_vid_name]['description'].append(cur_vid['description'])
    else:
      time_count = __major_vote(cur_vid['times'], cur_vid_num_seg)
      frame_per_sec = compile_dict[cur_vid_name]['feature_leng'] / compile_dict[cur_vid_name]['duration']
      time_count = [int(time_count[0] * 5), min(int((time_count[1]+1)*5), compile_dict[cur_vid_name]['duration'])]
      time_count = [int(time_count[0] * frame_per_sec), int(time_count[1] * frame_per_sec)]
      compile_dict[cur_vid_name]['times'].append(time_count)
      compile_dict[cur_vid_name]['description'].append(cur_vid['description'])

  #sorting as the start time
  for key in compile_dict.keys():
    end_time = [x[1] for x in compile_dict[key]['times']]
    sort_id = np.argsort(end_time)
    compile_dict[key]['times'] = np.array(compile_dict[key]['times'])[sort_id].tolist()
    compile_dict[key]['description'] = np.array(compile_dict[key]['description'])[sort_id].tolist()


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

def main():
    data = json_data('train_data')
    with open('train_data_bwzhang.json', 'w') as outfile:
        jsonmod.dump(data, outfile)
    data = json_data('val_data')
    with open('val_data_bwzhang.json', 'w') as outfile:
        jsonmod.dump(data, outfile)
    data = json_data('test_data')
    with open('testdata_bwzhang.json', 'w') as outfile:
        jsonmod.dump(data, outfile)


main()
