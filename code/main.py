
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
    path = root + '/data/tiny-landmarks'
    exp_path = root + '/experiment/' + args.experiment_name
    os.system('mkdir -p ' + exp_path)

    num_classes = 12

    trainBatcher = Batcher(path, percent=1.0, preload=False, batchSize=64, targetSet='train')
    loader = trainBatcher.loader

    devBatcher = Batcher(path, percent=1.0, preload=False, batchSize=128, targetSet='validate')
    dev_loader = devBatcher.loader

    if args.mode == 'train':
        device = getDevice()
        model = LandmarksModel(num_classes).to(device)
        optimizer = optim.SGD(model.getParameters(), lr=0.001, momentum=0.9)

        trainer = Trainer(model, loader, dev_loader, optimizer, device, exp_path)
        print('Start training...')
        trainer.train(epoch=1)

    elif args.mode == 'eval':
        print('Start evaluation on test set...')
    else:
        raise Exception('Unknown mode %s. Exiting...' % args.mode)

    print('Done!')

if __name__ == "__main__":
    main()

