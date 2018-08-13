"""
deconv.py

pytorch version: 0.4.1

Implementation for:
- Convolution Network: VGG16 pretrained model
- Deconvolution Network

Some code reference :
- https://github.com/csgwon/pytorch-deconvnet/blob/master/models/vgg16_deconv.py

"""


import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

vgg16 = models.vgg16(pretrained=True)

class Vgg_Convnet(nn.Module):
    def __init__(self, n_classes = 2):
        super(Vgg_Convnet, self).__init__()

        #modules = list(vgg16.children())

        # Need to change MaxPool2d option (return_indices => True)
        self.features = nn.Sequential(
            # conv1
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            # conv2
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv3
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            # conv4
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv5
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv6
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv7
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True))

        self.classifier = nn.Sequential(

            torch.nn.Linear(512 * 7 * 7, 4096),  # 224x244 image pooled down to 7x7 from features
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, n_classes))

        # store feature_outputs, pool_indices
        self.feature_outputs = [0]*len(self.features)
        self.pool_indices = dict()

        self.layerIdx = {1:1, 2:4, 3:6, 4:9, 5:16, 6:23, 7:30}

    def forward(self, images, layer_num):
        """
        :param images: Image
        :param layer_num: Layer Number which we want to see
        :return: pool_indices, feature maps of the certain layer
        """
        output = images
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                output, indices = layer(output)
                self.feature_outputs[i] = output
                self.pool_indices[i] = indices


            else:
                output = layer(output)
                self.feature_outputs[i] = output

            if i == self.layerIdx[layer_num]:
                return self.pool_indices, self.feature_outputs[i]

        return self.pool_indices, self.feature_outputs[i]


class Vgg_Deconvnet(nn.Module):
    def __init__(self, cnn_model):
        super(Vgg_Deconvnet, self).__init__()
        self.conv2DeconvIdx = {0: 17, 2: 16, 5: 14, 7: 13, 10: 11, 12: 10, 14: 9, 17: 7, 19: 6, 21: 5, 24: 3, 26: 2,
                               28: 1}
        self.conv2DeconvBiasIdx = {0: 16, 2: 14, 5: 13, 7: 11, 10: 10, 12: 9, 14: 7, 17: 6, 19: 5, 21: 3, 24: 2, 26: 1,
                                   28: 0}

        self.layerNumIdx = {1:17, 2: 15, 3: 14, 4: 12, 5: 8, 6:4, 7:0}
      

        self.unpool2PoolIdx = {0:30, 4:23, 8:16 , 12:9, 15:4}

        self.cnn_model = cnn_model

        self.features = nn.Sequential(
            nn.MaxUnpool2d(2, stride=2),
            #nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            #nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            #nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.MaxUnpool2d(2, stride=2),
            #nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            #nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            #nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 3, padding=1),
            nn.MaxUnpool2d(2, stride=2),
            #nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            #nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            #nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.MaxUnpool2d(2, stride=2),
            #nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 3, padding=1),
            #nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.MaxUnpool2d(2, stride=2),
            #nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            #nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, padding=1))


        self._initialize_weights()

    def _initialize_weights(self):
        # initializing weights using ImageNet-trained model from PyTorch
        for i, layer in enumerate(self.cnn_model.features):
            if isinstance(layer, nn.Conv2d):
                #weight_shape = layer.weight.data.shape
                self.features[self.conv2DeconvIdx[i]].weight.data = torch.transpose(layer.weight.data, 2, 3) #Transposed version of the same filters
                biasIdx = self.conv2DeconvBiasIdx[i]
                if biasIdx > 0 :
                    self.features[biasIdx].bias.data = layer.bias.data #need multiply -1 ?

        #print("Initialize Completed")

    def forward(self, layer_number, pool_indices, feature_map):
        start_idx = self.layerNumIdx[layer_number]
        output = feature_map
        for i in range(start_idx, len(self.features)):
            if isinstance(self.features[i], nn.MaxUnpool2d):
                output = self.features[i](output, pool_indices[self.unpool2PoolIdx[i]])
            else:
                output = self.features[i](output)


        return output


def get_activation(feature_map):
    tensor_size = feature_map.shape
    output = torch.zeros((tensor_size))

    activation = torch.sum(torch.abs(feature_map), (2,3))
    _, indices = torch.sort(activation, 1, descending=True)

    for i in range(10):
        listIdx = indices[0][i]
        output[0][listIdx] = feature_map[0][listIdx]
    return listIdx, output
