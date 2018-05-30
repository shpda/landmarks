
# main.py
# main entry point

from utils import *
from lm_model import LandmarksModel
from batcher import Batcher, getImageList
from trainer import Trainer

import os
import torch.optim as optim
import numpy as np

parser = getArgParser()

def main():
    print('Landmark Recogintion Project')

    args = parser.parse_args()
    printArgs(args)

    root = '/home/gangwu/projects/landmarks'
    path = '/home/gangwu/tiny-landmarks'
    exp_path = root + '/experiment/' + args.experiment_name
    os.system('mkdir -p ' + exp_path)

    imageList = getImageList(args.mode, checkMissingFile=True)

    #num_classes = 120
    num_classes = len(imageList[3].keys())
    num_train   = int(len(imageList[0]) * 0.95)
    num_dev     = len(imageList[0]) - num_train
    print('%d classes' % num_classes)
    print('%d train pictures' % num_train)
    print('%d dev pictures' % num_dev)

    # percentage of data to load
    pct = 0.01 

    device = getDevice()
    if device != None:
        model = LandmarksModel(num_classes).cuda(device)
    else:
        model = LandmarksModel(num_classes)

    if args.mode == 'train':
        trainBatcher = Batcher(path, imageList, percent=pct, preload=False, batchSize=128, num_train=num_train, tgtSet='train')
        loader = trainBatcher.loader
    
        devBatcher = Batcher(path, imageList, percent=pct, preload=False, batchSize=512, num_train=num_train, tgtSet='dev')
        dev_loader = devBatcher.loader

        #optimizer = optim.SGD(model.getParameters(), lr=0.001, momentum=0.9)
        #optimizer = optim.Adam(model.getParameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer = optim.Adam(model.getParameters(), lr=0.0001, betas=(0.9, 0.999))

        trainer = Trainer(model, loader, dev_loader, optimizer, device, exp_path)
        print('Start training...')
        trainer.train(epoch=60)

    elif args.mode == 'test':
        testBatcher = Batcher(path, percent=pct, preload=False, batchSize=512, targetSet='test')
        test_loader = testBatcher.loader

        trainer = Trainer(model, None, None, None, device, exp_path)
        print('Start evaluation on test set...')
        trainer.eval(test_loader, 'test')

    elif args.mode == 'submit':
        path = '/projects/landmarks/csvFiles'
        submitBatcher = Batcher(path, percent=pct, batchSize=512, isSubmit=True)
        submit_loader = submitBatcher.loader

        trainer = Trainer(model, None, None, None, device, exp_path)
        print('Start generating submition file...')
        _, idx2label = loadLabel2Idx('/home/gangwu/projects/landmarks/csvFiles/label2idx.csv')
        label2res = trainer.calc(submit_loader, idx2label)
        testCSVfile = '/home/gangwu/projects/landmarks/csvFiles'
        resultCSVfile = exp_path + '/results.csv'
        genResultFile(testCSVfile, resultCSVfile, label2res)

    elif args.mode == 'extract':
        path = '/projects/landmarks/csvFiles'
        idxImageBatcher = Batcher(path, percent=pct, batchSize=512, isSubmit=True)
        queryImageBatcher = Batcher(path, percent=pct, batchSize=512, isSubmit=True)

        trainer = Trainer(model, None, None, None, device, exp_path)
        print('Start extracting index image features...')
        label2idxFeature   = trainer.extract(idxImageBatcher.loader)
        np.save(exp_path + '/label2idxFeature.npy', label2idxFeature)

        print('Start extracting query image features...')
        label2queryFeature = trainer.extract(queryImageBatcher.loader)
        np.save(exp_path + '/label2queryFeature.npy', label2queryFeature)

    else:
        raise Exception('Unknown mode %s. Exiting...' % args.mode)

    print('Done!')

if __name__ == "__main__":
    main()

