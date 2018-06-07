
# batcher.py
# read in data and separate into batches

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import numpy as np
from PIL import Image

from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import time

from utils import saveLabel2Idx, loadLabel2Idx

# Resolve 'Set changed size during iteration'
#tqdm.monitor_interval = 0

def readCSVhelper(csvFileName, 
                  checkMissingFile=False,
                  readLabel=True):
    with open(csvFileName, 'r') as csvFile:
        CSVreader = csv.reader(csvFile, skipinitialspace=True, delimiter=',')
        fileIds = []
        fileNames = []
        labels = []
        missingFiles = 0
        print('Reading file %s' % csvFileName)
        for row in tqdm(CSVreader):
            baseName = row[0]
            #fName = path + '/train/' + baseName + '.jpg'
            fId   = row[0]
            fName = row[1]
            if readLabel:
                label = row[2]
            if checkMissingFile:
                if not osp.isfile(fName):
                    missingFiles = missingFiles + 1
                    continue
            fileIds.append(fId)
            fileNames.append(fName)
            if readLabel:
                labels.append(label)
        print('Found %d missing files' % missingFiles)
        print('Got %d picture ids' % (len(fileNames)))
        print('Got %d picture filenames' % (len(fileNames)))

        label2idx = {}
        if readLabel:
            idx = 0
            for l in labels:
                if int(l) not in label2idx.keys():
                    label2idx[int(l)] = idx
                    idx += 1
            saveLabel2Idx('/home/gangwu/projects/landmarks/csvFiles/label2idx.csv', label2idx)
            print('Got %d unique landmark labels' % (len(label2idx.keys())))
        return fileIds, fileNames, labels, label2idx

def readCSV(csvFile, checkMissingFile, readLabel=True):
    tic = time.time()
    # read filenames
    #csvFile = '%s/pruned_tiny_landmarks_%s.csv' % (self.root,targetSet)
    fileids, filenames, labels, label2idx = readCSVhelper(csvFile, checkMissingFile, readLabel)
    toc = time.time()
    print("Read filenames took %.2f s" % (toc-tic))
    return (fileids, filenames, labels, label2idx)

def getImageList(mode, checkMissingFile):
    #rec_train_csv = '/home/gangwu/projects/landmarks/csvFiles/new_rec_train.csv'
    #rec_test_csv = '/home/gangwu/projects/landmarks/csvFiles/new_rec_test.csv'
    #ret_index_csv = '/home/gangwu/projects/landmarks/csvFiles/new_ret_index.csv'
    #ret_test_csv = '/home/gangwu/projects/landmarks/csvFiles/new_ret_test.csv'
    #ret_index_csv = '/home/gangwu/projects/landmarks/csvFiles/small_new_ret_index.csv'
    #ret_test_csv = '/home/gangwu/projects/landmarks/csvFiles/small_new_ret_test.csv'

    pruned_train_csv = '/home/gangwu/projects/landmarks/csvFiles/pruned_tiny_landmarks_train.csv'
    filter_train_csv = '/home/gangwu/projects/landmarks/csvFiles/new_filter_train-256.csv'

    rec_train_csv = '/home/gangwu/projects/landmarks/csvFiles/new_rec_train-256.csv'
    rec_test_csv = '/home/gangwu/projects/landmarks/csvFiles/new_rec_test-256.csv'
    ret_index_csv = '/home/gangwu/projects/landmarks/csvFiles/new_ret_index-256.csv'
    ret_test_csv = '/home/gangwu/projects/landmarks/csvFiles/new_ret_test-256.csv'

    if mode == 'train':
        return readCSV(rec_train_csv, checkMissingFile=True)
    elif mode == 'train-pruned':
        return readCSV(pruned_train_csv, checkMissingFile=True)
    elif mode == 'train-filter':
        return readCSV(filter_train_csv, checkMissingFile=True)
    elif mode == 'submit0':
        return readCSV(rec_test_csv, checkMissingFile=True, readLabel=False)
    elif mode == 'extract':
        indexImages = readCSV(ret_index_csv, checkMissingFile=True, readLabel=False)
        queryImages = readCSV(ret_test_csv, checkMissingFile=True, readLabel=False)
        return (indexImages, queryImages)
    return None

class LandmarksData(Dataset):
    """
    Data loader for landmarks data.
    """
    def __init__(self,
                 imageList,
                 percent=1.0,
                 transform=None,
                 num_train=0,
                 tgtSet='train',
                 preload=False):

        self.images = None

        if tgtSet=='train':
            self.filenames = imageList[1][:num_train]
            self.labels = imageList[2][:num_train]
        elif tgtSet=='dev':
            self.filenames = imageList[1][num_train:]
            self.labels = imageList[2][num_train:]
        else:
            print('ERROR: unknow tgtSet')

        self.transform = transform
        self.label2idx = imageList[3]

        fullLen = len(self.filenames)
        shorterLen = int(fullLen * percent)
        print('Percentage to load = {}/{} ({:.0f}%)'.format(shorterLen, fullLen, 100. * percent))
        self.filenames = self.filenames[:shorterLen]
        self.labels = self.labels[:shorterLen]

        # if preload dataset into memory
        if preload:
            self._preload()
            
        self.len = len(self.filenames)
                              
    def _preload(self):
        """
        Preload dataset to memory
        """
        #self.labels = []
        self.images = []
        print('Preloading...')
        tic = time.time()
        #for image_fn in self.filenames:
        for image_fn in tqdm(self.filenames):            
            # load images
            image = Image.open(image_fn)
            # avoid too many opened files bug
            self.images.append(image.copy())
            image.close()
        toc = time.time()
        print("Preload took %.2f s" % (toc-tic))

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        labelIdx = 0
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
            if self.labels:
                label = int(self.labels[index])
                labelIdx = self.label2idx[label]
        else:
            # If on-demand data loading
            image_fn = self.filenames[index]
            image = Image.open(image_fn)
            if self.labels:
                label = int(self.labels[index])
                labelIdx = self.label2idx[label]
            
        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)
        # return image and label
        return image, labelIdx

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

class LandmarksDataSubmit(Dataset):
    """
    Data loader for landmarks submission file.
    """
    def __init__(self, imageList, percent=1.0, transform=None):

        self.fileids = imageList[0]
        self.filenames = imageList[1]
        self.transform = transform

        fullLen = len(self.filenames)
        shorterLen = int(fullLen * percent)
        print('Percentage to load = {}/{} ({:.0f}%)'.format(shorterLen, fullLen, 100. * percent))
        self.filenames = self.filenames[:shorterLen]
        self.fileids = self.fileids[:shorterLen]

        self.len = len(self.filenames)
                              
    def __getitem__(self, index):

        image_fn = self.filenames[index]
        image_id = self.fileids[index]
        image = Image.open(image_fn)
        if self.transform is not None:
            image = self.transform(image)

        return image, image_id

    def __len__(self):
        return self.len

class Batcher(object):
    """
    Get preprocessed data batches
    """
    def __init__(self,
                 imageList=None,
                 percent=1.0, # load a subset of data
                 preload=False,
                 batchSize=64,
                 num_train=0,
                 tgtSet='train',
                 isSubmit=False):

        # preprocessing stuff
        #myTrans = transforms.Compose([transforms.Resize((32, 32)),
        #                              transforms.ToTensor()])
        #myTrans = transforms.Compose([transforms.CenterCrop(256),
        #                              transforms.ToTensor()])
        myTrans = transforms.Compose([
            #transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                  std = [ 0.229, 0.224, 0.225 ])])

        if not isSubmit:
            dataset = LandmarksData(imageList, percent=percent, preload=preload, 
                                    transform=myTrans, num_train=num_train, tgtSet=tgtSet)
            self.loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=10)
            self.dataiter = iter(self.loader)
        else:
            dataset = LandmarksDataSubmit(imageList, percent=percent, transform=myTrans)
            self.loader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=10)
            self.dataiter = iter(self.loader)
        #print(len(trainset))
        #print(len(testset))

# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.figure(figsize = (50, 50))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def showDataInClass(classId):
    path = '/home/gangwu/landmarks-data/landmarks-data'
    first = True
    myTrans = transforms.Compose([transforms.Resize((128, 128)),
                                  transforms.ToTensor()])
    #myTrans = transforms.Compose([transforms.ToTensor()])
    with open('/home/gangwu/projects/landmarks/data/train.csv') as csvfile:
        CSVreader = csv.reader(csvfile, delimiter=',')
        fileNames = []
        images = []
        #for row in CSVreader:
        for row in tqdm(CSVreader):
            if first:
                first = False
                continue
            baseName = row[0]
            label = row[2]
            if int(label) != classId:
                continue
            fName = path + '/train/' + baseName + '.jpg'
            if osp.isfile(fName):
                fileNames.append(fName)
                image = Image.open(fName)
                images.append(myTrans(image.copy()))
                image.close()
            if len(fileNames) > 63:
                break
        print('Got %d train picture files with class %d' % (len(fileNames), classId))
        imshow(torchvision.utils.make_grid(images))

def run_test():
    #path = '/home/gangwu/small-landmarks-data'
    path = '/home/gangwu/projects/landmarks/data/tiny-landmarks'
    batcher = Batcher(path, preload=True)
    imagesIter = batcher.dataiter
    images, labels = imagesIter.next()

    # visualize the dataset
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % labels[j] for j in range(16)))

if __name__ == "__main__":
    run_test()

