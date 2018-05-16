
# fetch_data.py
# analysis and fetch data from Google Cloud Bucket to local disk

import os.path as osp
from tqdm import tqdm

import argparse
import csv

parser = argparse.ArgumentParser(description='fetch data')
parser.add_argument('--mode', metavar='M', default='analysis', 
                    help='select mode')

path = '/home/gangwu/landmarks-data/landmarks-data'

def analysisData():
    with open('./train.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        labelMap = {}
        missingFiles = 0
        totalFiles = 0
        for row in tqdm(readCSV):
            baseName = row[0]
            label = row[2]
            fName = path + '/train/' + baseName + '.jpg'
            if label in labelMap.keys():
                labelMap[label].append(fName)
            else:
                labelMap[label] = list(fName)
            '''
            if osp.isfile(fName):
                if label in labelMap.keys():
                    labelMap[label].append(fName)
                else:
                    labelMap[label] = list(fName)
            else:
                missingFiles = missingFiles + 1
            '''
            totalFiles = totalFiles + 1
        print('Found %d missing train files' % missingFiles)
        print('Got %d train files' % totalFiles)
        print('Got %d unique labels' % len(labelMap))

def main():
    args = parser.parse_args()

    if args.mode == 'analysis':
        analysisData()
    elif args.mode == 'fetch':
        print('Fetching data...')
    else:
        raise Exception('Unknown mode %s. Exiting...' % args.mode)


if __name__ == "__main__":
    main()

