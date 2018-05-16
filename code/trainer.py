
# trainer.py
# helper function to train on epochs

import torch
import torch.nn.functional as func
from utils import saveModel, loadModel
import time
import sys

class Trainer():
    def __init__(self, model, loader, dev_loader, optimizer, device, exp_path, 
                log_interval=5, eval_interval=20, save_interval=20):
        self.model = model
        self.loader = loader
        self.dev_loader = dev_loader
        self.optimizer = optimizer
        self.device = device
        self.exp_path = exp_path
        self.log_interval = log_interval 
        self.eval_interval = eval_interval 
        self.save_interval = save_interval 

    def train(self, epoch=5):
        self.model.train()  # set training mode
        iteration = 0 
        best_dev_loss = sys.float_info.max
        for ep in range(epoch):
            epoch_tic = time.time()
            for batch_idx, (data, target) in enumerate(self.loader):
                data, target = data.cuda(self.device), target.cuda(self.device)
                self.optimizer.zero_grad()

                # forward pass
                output = self.model(data)
                loss = func.nll_loss(output, target)

                # backward pass
                loss.backward()

                # weight update
                self.optimizer.step()

                if iteration % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(ep, 
                        batch_idx * len(data), len(self.loader.dataset), 
                        100. * batch_idx / len(self.loader), loss.item()))

                '''
                if iteration % self.eval_interval == 0:
                    dev_loss = self.devEval()
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        saveModel('%s/lm-best.pth' % self.exp_path, 
                                  self.model, self.optimizer)
                '''

                '''
                if iteration % self.save_interval == 0:
                    self.saveModel()
                '''

                iteration += 1
            epoch_toc = time.time()
            print('End of epoch %i. Seconds took: %.2f s.' % (ep, epoch_toc - epoch_tic))
            dev_loss = self.devEval()
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                saveModel('%s/lm-best.pth' % self.exp_path, self.model, self.optimizer)

    def devEval(self):
        self.model.eval()  # set evaluation mode
        dev_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.dev_loader:
                data, target = data.to(self.device), target.to(self.device)

                # calculate accumulated loss
                output = self.model(data)
                dev_loss += func.nll_loss(output, target, size_average=False).item() # sum up batch loss

                # calculate accumulated accuracy
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        dev_loss /= len(self.dev_loader.dataset)
        print('Dev set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(dev_loss, 
            correct, len(self.dev_loader.dataset), 100. * correct / len(self.dev_loader.dataset)))

        return dev_loss

