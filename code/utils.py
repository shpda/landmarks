
import torch
import argparse
import csv

def getArgParser():
    parser = argparse.ArgumentParser(description='Landmark Recognition Project')
    parser.add_argument('--experiment_name', metavar='EXP_NAME', default='unknown', 
                        help='name for the experiment')
    parser.add_argument('--mode', metavar='M', default='train', 
                        help='select mode')
    return parser

def printArgs(args):
    print('experiment_name = %s' % args.experiment_name)
    print('mode = %s' % args.mode)

def getDevice():
    use_cuda = torch.cuda.is_available()
    device = torch.cuda.device(1 if use_cuda else "cpu")
    print('device = %s' % device)
    return 0

def saveModel(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def loadModel(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

class Logger():
    def __init__(self, exp_path, name):
        fileName = exp_path + '/' + name + '.csv'
        self.logFile = open(fileName, 'w', 1) # line buffering
        self.writer = csv.writer(self.logFile)
    def writeLoss(self, itr, loss):
        self.writer.writerow((itr, loss))
    def __del__(self):
        self.logFile.close()
