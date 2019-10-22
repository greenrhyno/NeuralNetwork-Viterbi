#!/usr/bin/python

#set default GPU to gpu 0
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import glob
import re
import numpy as np

def recog_file(filename, ground_truth_path):

    # read ground truth
    gt_file = ground_truth_path + re.sub('.*/','/',filename) + '.txt'
    with open(gt_file, 'r') as f:
        ground_truth = f.read().split('\n')[0:-1]
        f.close()
    # read recognized sequence
    with open(filename, 'r') as f:
        recognized = f.read().split('\n')[5].split() # framelevel recognition is in 6-th line of file
        f.close()

    n_frame_errors = 0
    n_subactions = 0
    for i in range(len(recognized)):
        if not recognized[i] == ground_truth[i]:
            n_frame_errors += 1
        if i == 0 or not ground_truth[i] == ground_truth[i-1]:
            n_subactions += 1

    return n_frame_errors, len(recognized), n_subactions


### MAIN #######################################################################

### arguments ###
### --recog_dir: the directory where the recognition files from inferency.py are placed
### --ground_truth_dir: the directory where the framelevel ground truth can be found
parser = argparse.ArgumentParser()
parser.add_argument('--recog_dir', default='results')
parser.add_argument('--ground_truth_dir', default='data/groundTruth')
args = parser.parse_args()

filelist = glob.glob(args.recog_dir + '/P*')

print('Evaluate %d video files...' % len(filelist))

n_frames = 0
n_errors = 0
stats = []
# loop over all recognition files and evaluate the frame error
for filename in filelist:
    errors, frames, n_subactions = recog_file(filename, args.ground_truth_dir)
    n_errors += errors
    n_frames += frames
    stats.append([errors, frames, n_subactions])


# print frame accuracy (1.0 - frame error rate)
print('frame accuracy: %f' % (1.0 - float(n_errors) / n_frames))

np.save(stats, 'errors_frames_subactions_stats.npy')
