import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Downsampling convolution
        self.down_conv1 = nn.Conv2d(1, 64, 4, 1, 2)
        self.down_conv2 = nn.Conv2d(64, 128, 4, 1, 2)
        self.down_conv3 = nn.Conv2d(128, 256, 4, 1, 2)
        self.down_conv4 = nn.Conv2d(256, 512, 4, 1, 2)
        self.down_conv5 = nn.Conv2d(512, 1024, 4, 1, 2)

        # Upsampling convolution
        self.up_conv5 = nn.ConvTranspose2d(1024, 512, 4, 1, 2)
        self.up_conv4 = nn.ConvTranspose2d(512, 256, 4, 1, 2)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 4, 1, 2)
        self.up_conv2 = nn.ConvTranspose2d(128, 64, 4, 1, 2)
        self.up_conv1 = nn.ConvTranspose2d(64, 1, 4, 1, 2)

        # Affine Operations
        
