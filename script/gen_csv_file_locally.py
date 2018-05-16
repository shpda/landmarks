
# move_data.py
# move data from Google Cloud Bucket to local disk

import os
import os.path as osp
import csv
import re
import glob
from tqdm import tqdm

def getBaseFileName(fileName):
    match = re.search("validate/(\S+)/(\S+)\.jpg", fileName)
    if match:
        return match.group(1), match.group(2)
    return 'Error'

def main():
    f = open('./tiny_tiny_landmarks_validate.csv', 'w')
    for fn in glob.glob('/home/gangwu/projects/landmarks/data/tiny-landmarks/validate/*/*.jpg'):
        labelMap = {}
        label, idx = getBaseFileName(fn)
        f.write('%s, %s, %s\n' % (idx, fn, label))
    f.closed

if __name__ == "__main__":
    main()

