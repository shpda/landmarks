
# main.py
# main entry point

from utils import *
from lm_model import LandmarksModel
from batcher import Batcher
from trainer import Trainer

import torch.optim as optim

parser = getArgParser()

def main():
    print('Landmark Recogintion Project')

    args = parser.parse_args()
    printArgs(args)

    batcher = Batcher('/home/gangwu/small-landmarks-data', percent=0.1, preload=True)
    loader = batcher.loader

    if args.mode == 'train':
        device = getDevice()
        model = LandmarksModel().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        #model = LandmarksModel().to(getDevice())
        trainer = Trainer(model, loader, optimizer, device)
        print('Start training...')
        trainer.train(epoch=1)

    elif args.mode == 'eval':
        print('Start evaluation...')
    else:
        raise Exception('Unknown mode %s. Exiting...' % args.mode)

    print('Done!')

if __name__ == "__main__":
    main()

