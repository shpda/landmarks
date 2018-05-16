
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

# Resolve 'Set changed size during iteration'
#tqdm.monitor_interval = 0

def readCSV(path, targetSet):
    with open('%s/tiny_tiny_landmarks_%s.csv' % (path,targetSet)) as csvfile:
        CSVreader = csv.reader(csvfile, skipinitialspace=True, delimiter=',')
        fileNames = []
        labels = []
        missingFiles = 0
        for row in CSVreader:
            baseName = row[0]
            #fName = path + '/train/' + baseName + '.jpg'
            fName = row[1]
            label = row[2]
            #if osp.isfile(fName):
            fileNames.append(fName)
            labels.append(label)
            #else:
            #missingFiles = missingFiles + 1
        print('Found %d missing %s files' % (missingFiles, targetSet))
        print('Got %d %s picture filenames' % (len(fileNames), targetSet))
        print('Got %d %s picture labels' % (len(labels), targetSet))
        return fileNames, labels

class LandmarksData(Dataset):
    """
    Data loader for landmarks data.
    """
    def __init__(self,
                 root,
                 percent=1.0,
                 transform=None,
                 preload=False,
                 targetSet='train'):

        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        self.label2idx = {}

        tic = time.time()
        # read filenames
        self.filenames, self.labels = readCSV(self.root, targetSet)
        toc = time.time()
        print("Read filenames took %.2f s" % (toc-tic))

        fullLen = len(self.filenames)
        shorterLen = int(fullLen * percent)
        print('Percentage to load = {}/{} ({:.0f}%)'.format(shorterLen, fullLen, 100. * percent))
        self.filenames = self.filenames[:shorterLen]
        self.labels = self.labels[:shorterLen]

        idx = 0
        for l in self.labels:
            if int(l) not in self.label2idx.keys():
                self.label2idx[int(l)] = idx
                idx += 1
        
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

class Batcher(object):
    """
    Get preprocessed data batches
    """
    def __init__(self,
                 root='/home/gangwu/small-landmarks-data',
                 percent=1.0, # load a subset of data
                 preload=False,
                 batchSize=64,
                 targetSet='train'):

        # preprocessing stuff
        #myTrans = transforms.Compose([transforms.Resize((32, 32)),
        #                              transforms.ToTensor()])
        myTrans = transforms.Compose([transforms.CenterCrop(256),
                                      transforms.ToTensor()])

        dataset = LandmarksData(root=root, percent=percent, preload=preload, transform=myTrans, targetSet=targetSet)
        self.loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=1)
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

