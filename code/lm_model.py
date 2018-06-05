
# lm_model.py
# landmarks model to be trained

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as func

import pretrainedmodels

def getModel(mode, device, num_classes, input_size):
    model = None

    if mode == 'train' or mode == 'train-pruned' or mode == 'submit0':
        model = LandmarksModel(num_classes, input_size)
    elif mode == 'extract':
        model = FeatureExtractModel(num_classes, input_size)
    else:
        return None

    if device != None:
        model = model.cuda(device)

    return model

class LandmarksModel(nn.Module):
    def __init__(self, num_classes, input_size):
        super(LandmarksModel, self).__init__()
        #self.modelName = 'resnet'
        #self.nnet = torchvision.models.resnet101(pretrained=True)
        #self.nnet = torchvision.models.resnet50(pretrained=True)
        #self.modelName = 'densenet'
        #self.nnet = torchvision.models.densenet161(pretrained=True)
        self.modelName = 'se_resnet'
        self.nnet = pretrainedmodels.se_resnet101(pretrained='imagenet')
        self.nnet.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        #self.nnet.fc = nn.Linear(self.nnet.fc.in_features, num_classes)   # for resnet
        #self.nnet.classifier = nn.Linear(self.nnet.classifier.in_features, num_classes)   # for densenet
        self.nnet.last_linear = nn.Linear(self.nnet.last_linear.in_features, num_classes)   # for se_resnet

        '''
        self.features = nn.Sequential(*list(nnet.children())[:-1])
        self.classifier = nn.Sequential(
                            #nn.Linear(nnet.fc.in_features, num_classes)          # for resnet
                            #nn.Linear(nnet.classifier.in_features, num_classes)   # for densenet
                            nn.Linear(nnet.last_linear.in_features, num_classes)   # for se_resnet
                          )
        '''

        #for p in self.features.parameters():
        #    p.requires_grad = False


    def forward(self, x):
        x = self.nnet(x)
        #x = self.features(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x

    def getParameters(self):
        #return filter(lambda p: p.requires_grad, self.parameters())
        return self.nnet.parameters()

class FeatureExtractModel(nn.Module):
    def __init__(self, num_classes, input_size):
        super(FeatureExtractModel, self).__init__()
        self.modelName = 'featureExtract'
        #nnet = torchvision.models.resnet101(pretrained=True)
        #nnet = torchvision.models.resnet50(pretrained=True)
        nnet = torchvision.models.densenet161(pretrained=True)
        #nnet = pretrainedmodels.se_resnet101(pretrained='imagenet')

        #self.features = nn.Sequential(*list(nnet.children())[:-2]) # conv5_x
        self.features = nn.Sequential(*list(nnet.children())[:-4]) # conv3_x

        self.extractor = nn.Sequential(
                            #nn.MaxPool2d(input_size // 32, stride=1) # conv5_x
                            nn.MaxPool2d(input_size // 8, stride=1) # conv3_x
                         )

        #for p in self.features.parameters():
        #    p.requires_grad = False


    def forward(self, x):
        x = self.features(x)
        x = self.extractor(x)
        return x

'''
class LandmarksModel(nn.Module):
    def __init__(self):
        super(LandmarksModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 12)

    def forward(self, x):
        x = func.relu(func.max_pool2d(self.conv1(x), 2))
        x = func.relu(func.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = func.relu(self.fc1(x))
        x = func.dropout(x, training=self.training)
        x = self.fc2(x)
        return func.log_softmax(x, dim=1)
'''

def run_test():
    print('Test landmarks module')

if __name__ == "__main__":
    run_test()

