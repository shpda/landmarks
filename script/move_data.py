
# move_data.py
# move data from Google Cloud Bucket to local disk

import os
import os.path as osp
import csv
import re
from tqdm import tqdm

path = '/home/gangwu/landmarks-data/landmarks-data'

def getLabelMap():
    with open('../data/train.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        labelMap = {}
        uniqueLabelList = []
        labelList = []
        missingFiles = 0
        totalFiles = 0
        first = True
        for row in tqdm(readCSV):
            if first:
                first = False
                continue # skip table head
            baseName = row[0]
            label = row[2]
            fName = path + '/train/' + baseName + '.jpg'
            if label in labelMap.keys():
                labelMap[label].append(fName)
            else:
                labelMap[label] = list([fName])
                uniqueLabelList.append(int(label))
            labelList.append(int(label))
            totalFiles += 1
        print('Found %d missing train files' % missingFiles)
        print('Got %d train files' % totalFiles)
        print('Got %d unique labels' % len(labelMap))
        return labelMap, labelList, uniqueLabelList

def getBaseFileName(fileName):
    match = re.search("train/(\S+)\.jpg", fileName)
    if match:
        return match.group(1)
    return 'Error'

def main():
    labelMap, labelList, uniqueLabelList = getLabelMap()

    maxDataLabel = 0
    minDataLabel = 0
    tpList = []
    
    for k in labelMap.keys():
        dataLen = len(labelMap[k])
        tpList.append((int(k), dataLen))
    
    tpList.sort(key=lambda tup: tup[1])
    
    print('max data in class: %d with class label: %d' % (tpList[-1][1], tpList[-1][0]))
    print('min data in class: %d with class label: %d' % (tpList[0][1], tpList[0][0]))

    s = 0
    fTrain = open('./tiny_landmarks_train.csv', 'w')
    fValidate = open('./tiny_landmarks_validate.csv', 'w')
    fTest = open('./tiny_landmarks_test.csv', 'w')
    for e in reversed(tpList):
        print('classIdx: %d, size: %d, label: %d' % (s, e[1], e[0]))
        label = e[0]
        fileList = labelMap[str(label)]
        cnt = 0
        os.system('mkdir -p /home/gangwu/projects/landmarks/data/tiny-landmarks/train/'+str(label))
        os.system('mkdir -p /home/gangwu/projects/landmarks/data/tiny-landmarks/validate/'+str(label))
        os.system('mkdir -p /home/gangwu/projects/landmarks/data/tiny-landmarks/test/'+str(label))
        for fn in fileList:
            if osp.isfile(fn):
                if cnt < 500:
                    os.system('cp ' + fn + ' /home/gangwu/projects/landmarks/data/tiny-landmarks/train/'+str(label)+'/')
                    fTrain.write('%s, %s, %d\n' % (getBaseFileName(fn), fn, label))
                elif cnt < 550:
                    os.system('cp ' + fn + ' /home/gangwu/projects/landmarks/data/tiny-landmarks/validate/'+str(label)+'/')
                    fValidate.write('%s, %s, %d\n' % (getBaseFileName(fn), fn, label))
                else:
                    os.system('cp ' + fn + ' /home/gangwu/projects/landmarks/data/tiny-landmarks/test/'+str(label)+'/')
                    fTest.write('%s, %s, %d\n' % (getBaseFileName(fn), fn, label))
                cnt+=1
                if cnt % 50 == 0:
                    print(str(cnt) + ' ',end='') 
            if cnt > 599:
                print('')
                break
        s += 1
        if s > 199:
            break
    fTrain.closed
    fValidate.closed
    fTest.closed

if __name__ == "__main__":
    main()

