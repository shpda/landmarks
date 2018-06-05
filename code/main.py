
# main.py
# main entry point

from utils import *
from lm_model import getModel
from batcher import Batcher, getImageList
from trainer import Trainer
from nnsearch import nnsearch

import os
import torch.optim as optim
import numpy as np
import time

parser = getArgParser()

def main():
    print('Landmark Recogintion & Retrieval Project')

    args = parser.parse_args()
    printArgs(args)

    root = '/home/gangwu/projects/landmarks'
    exp_path = root + '/experiment/' + args.experiment_name
    os.system('mkdir -p ' + exp_path)

    #input_size = 128
    input_size = 224 # after crop
    testCSVfile = '/home/gangwu/projects/landmarks/csvFiles/new_ret_test-256.csv'
    #testCSVfile = '/home/gangwu/projects/landmarks/csvFiles/pruned_tiny_landmarks_test.csv'

    imageList = getImageList(args.mode, checkMissingFile=True)

    #num_classes = len(imageList[3].keys())
    num_classes = 14951
    print('%d classes' % num_classes)
    num_train, num_dev = splitTrainDevSet(imageList, 0.98)

    # percentage of data to load
    pct = 1.0 

    device = getDevice()
    model = getModel(args.mode, device, num_classes, input_size)

    if args.mode == 'train' or args.mode == 'train-pruned':
        # resnet50 batch size: train = 100, dev = 256
        # densenet161 batch size: train = 40, dev = 128
        # seresnet101 batch size: train = 48, dev = 128
        trainBatcher = Batcher(imageList, percent=pct, preload=False, batchSize=48, num_train=num_train, tgtSet='train')
        loader = trainBatcher.loader
    
        devBatcher = Batcher(imageList, percent=pct, preload=False, batchSize=128, num_train=num_train, tgtSet='dev')
        dev_loader = devBatcher.loader

        #optimizer = optim.SGD(model.getParameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(model.getParameters(), lr=0.0001, betas=(0.9, 0.999))

        trainer = Trainer(args.mode, model, loader, dev_loader, optimizer, device, exp_path)
        print('Start training...')
        trainer.train(epoch=60)

        '''
        elif args.mode == 'test':
            testBatcher = Batcher(percent=pct, preload=False, batchSize=512, targetSet='test')
            test_loader = testBatcher.loader
    
            trainer = Trainer(model, None, None, None, device, exp_path)
            print('Start evaluation on test set...')
            trainer.eval(test_loader, 'test')
        '''

    elif args.mode == 'submit0':
        submitBatcher = Batcher(imageList, percent=pct, batchSize=512, isSubmit=True)
        submit_loader = submitBatcher.loader

        trainer = Trainer(args.mode, model, None, None, None, device, exp_path)
        print('Start generating landmarks recognization submition file...')
        _, idx2label = loadLabel2Idx('/home/gangwu/projects/landmarks/csvFiles/label2idx.csv')
        label2res = trainer.calc(submit_loader, idx2label)
        resultCSVfile = exp_path + '/rec_results.csv'
        genResultFile(args.mode, testCSVfile, resultCSVfile, label2res)

    elif args.mode == 'extract':
        idxImageBatcher = Batcher(imageList[0], percent=pct, batchSize=512, isSubmit=True)
        queryImageBatcher = Batcher(imageList[1], percent=pct, batchSize=512, isSubmit=True)

        trainer = Trainer(args.mode, model, None, None, None, device, exp_path)
        print('Start extracting index image features...')
        idxLabel, idxFeature = trainer.extract(idxImageBatcher.loader)
        idxLabelPath = exp_path + '/idxLabel.npy'
        idxFeaturePath = exp_path + '/idxFeature.npy'
        np.save(idxLabelPath, idxLabel)
        np.save(idxFeaturePath, idxFeature)
        print('Extracted features saved at %s' % idxFeaturePath)

        print('Start extracting query image features...')
        queryLabel, queryFeature = trainer.extract(queryImageBatcher.loader)
        queryLabelPath = exp_path + '/queryLabel.npy'
        queryFeaturePath = exp_path + '/queryFeature.npy'
        np.save(queryLabelPath, queryLabel)
        np.save(queryFeaturePath, queryFeature)
        print('Extracted features saved at %s' % queryFeaturePath)

    elif args.mode == 'submit1':
        print('Loading features...')
        idxLabelPath = exp_path + '/idxLabel.npy'
        idxFeaturePath = exp_path + '/idxFeature.npy'
        idxLabel = np.load(idxLabelPath)
        idxFeature = np.load(idxFeaturePath)
        print('idxLabel shape: %s' % str(idxLabel.shape))
        print('idxFeature shape: %s' % str(idxFeature.shape))

        queryLabelPath = exp_path + '/queryLabel.npy'
        queryFeaturePath = exp_path + '/queryFeature.npy'
        queryLabel = np.load(queryLabelPath)
        queryFeature = np.load(queryFeaturePath)
        print('queryLabel shape: %s' % str(queryLabel.shape))
        print('queryFeature shape: %s' % str(queryFeature.shape))

        print('Searching neighbors...')
        tic = time.time()
        label2res = nnsearch(idxFeature, queryFeature, idxLabel, queryLabel)
        toc = time.time()
        print("Search neighbors took %.2f s" % (toc-tic))

        print('Start generating landmarks retrieval submition file...')
        resultCSVfile = exp_path + '/ret_results.csv'
        genResultFile(args.mode, testCSVfile, resultCSVfile, label2res)

    else:
        raise Exception('Unknown mode %s. Exiting...' % args.mode)

    print('Done!')

if __name__ == "__main__":
    main()

