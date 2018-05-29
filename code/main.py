
# main.py
# main entry point

from utils import *
from lm_model import LandmarksModel
from batcher import Batcher
from trainer import Trainer

import os
import torch.optim as optim

parser = getArgParser()

def main():
    print('Landmark Recogintion Project')

    args = parser.parse_args()
    printArgs(args)

    root = '/home/gangwu/projects/landmarks'
    path = '/home/gangwu/tiny-landmarks'
    exp_path = root + '/experiment/' + args.experiment_name
    os.system('mkdir -p ' + exp_path)

    num_classes = 120

    # CPU: 10% data
    # GPU: 100% data
    pct = 1.0

    device = getDevice()
    if device != None:
        model = LandmarksModel(num_classes).cuda(device)
    else:
        model = LandmarksModel(num_classes)

    if args.mode == 'train':
        trainBatcher = Batcher(path, percent=pct, preload=False, batchSize=128, targetSet='train')
        loader = trainBatcher.loader
    
        devBatcher = Batcher(path, percent=pct, preload=False, batchSize=512, targetSet='validate')
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
        '''
        submitBatcher = Batcher(path, percent=pct, preload=False, batchSize=512, targetSet='test')
        test_loader = testBatcher.loader

        trainer = Trainer(model, None, test_loader, None, device, exp_path)
        print('Start evaluation on test set...')
        trainer.eval(test_loader, 'test')
        '''
    else:
        raise Exception('Unknown mode %s. Exiting...' % args.mode)

    print('Done!')

if __name__ == "__main__":
    main()

