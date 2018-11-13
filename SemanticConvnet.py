import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Up-pull and Down-pull operations
        self.pull_1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pull_2 = nn.MaxPool2d(kernel_size=3, stride=1,padding=1, return_indices=True)
        self.unpull = nn.MaxUnpool2d(kernel_size=4, stride=2,padding=1)

        # Downsampling convolution
        self.down_conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.down_conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.down_conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.down_conv4 = nn.Conv2d(256, 512, 3, 1, 1)
        self.down_conv5 = nn.Conv2d(512, 1024, 3, 1, 1)

        # Upsampling convolution
        self.up_conv5 = nn.ConvTranspose2d(1024, 512, 3, 1, 1)
        self.up_conv4 = nn.ConvTranspose2d(512, 256, 3, 1, 1)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 3, 1, 1)
        self.up_conv2 = nn.ConvTranspose2d(128, 64, 3, 1, 1)
        self.up_conv1 = nn.ConvTranspose2d(64, 1, 3, 1, 1)



    def forward(self, x):
        # DownSampling
        x = self.pull_1(F.relu(self.down_conv1(x)))
        x,indconv1 = self.pull_2(x)

        x = self.pull_1(F.relu(self.down_conv2(x)))
        x,indconv2 = self.pull_2(x)

        x = self.pull_1(F.relu(self.down_conv3(x)))
        x,indconv3 = self.pull_2(x)

        x = self.pull_1(F.relu(self.down_conv4(x)))
        x,indconv4 = self.pull_2(x)

        x = self.pull_1(F.relu(self.down_conv5(x)))
        x,indconv5 = self.pull_2(x)


        # UpSampling started
        x = F.relu(self.up_conv5(self.unpull(x,indconv5)))

        x = F.relu(self.up_conv4(self.unpull(x,indconv4)))

        x = F.relu(self.up_conv3(self.unpull(x,indconv3)))

        x = F.relu(self.up_conv2(self.unpull(x,indconv2)))

        x = F.relu(self.up_conv1(self.unpull(x,indconv1)))

        x = x.view(-1, self.num_flat_features(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    @classmethod
    def create(net,*args,**kwargs):
        return  net().double()
