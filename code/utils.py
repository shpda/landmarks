
import torch
import argparse
import csv
import os.path

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
    if not use_cuda:
        print('device = cpu')
        return None
    device = torch.cuda.device(0) # GPU 0
    print('device = %s' % device)
    return 0

def saveModel(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def loadModel(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    if model != None:
        model.load_state_dict(state['state_dict'])
    else:
        print('model does not exist')
    if optimizer != None:
        optimizer.load_state_dict(state['optimizer'])
    else:
        print('optimizer does not exist')
    print('model loaded from %s' % checkpoint_path)

def tryRestore(fname, model, optimizer):
    if os.path.isfile(fname):
        print('Restoring best checkpoint')
        loadModel(fname, model, optimizer)
        return True
    return False

class Logger():
    def __init__(self, exp_path, name, writeType):
        fileName = exp_path + '/' + name + '.csv'
        self.logFile = open(fileName, writeType, 1) # line buffering
        self.writer = csv.writer(self.logFile)
    def writeLoss(self, itr, loss, accuracy = 0.0):
        self.writer.writerow((itr, loss, accuracy))
    def __del__(self):
        self.logFile.close()

def saveLabel2Idx(fileName, label2idx):
    with open(fileName, 'w') as csvFile:
        writter = csv.writer(csvFile)
        for label, idx in label2idx.items():
            writter.writerow((idx, label))

def loadLabel2Idx(fileName):
    label2idx = {}
    idx2label = {}
    with open(fileName, 'r') as csvFile:
        CSVreader = csv.reader(csvFile, skipinitialspace=True, delimiter=',')
        for row in CSVreader:
            idx = int(row[0])
            label = int(row[1])
            label2idx[label] = idx
            idx2label[idx] = label

    return label2idx, idx2label

def genResultFile(testCSVfile, resultCSVfile, label2result):
    outputFile = open(resultCSVfile, 'w')
    CSVwriter = csv.writer(outputFile)
    CSVwriter.writerow(('id', 'landmarks'))
    with open(testCSVfile, 'r') as csvFile:
        CSVreader = csv.reader(csvFile, skipinitialspace=True, delimiter=',')
        first = True
        for row in CSVreader:
            if first:
                first = False
                continue
            label = row[0]
            if label in label2result.keys():
                CSVwriter.writerow((label, label2result[label]))
            else:
                CSVwriter.writerow((label, '0 0.0'))
    outputFile.close()

