import SemanticConvnet as sc
import DataLoader as dl
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np
import torch

absFilePath = os.path.dirname(os.path.abspath(__file__))

class Train:
    def __init__(
                    self,
                    root_dir = './VolSegData/',
                    checkpoint_dir = './checkpoints/',
                    batch_size = 10,
                    shuffle = True,
                    num_workers = 0,
                    num_epochs = 2,
                    load_from = None
                ):

        self.model = sc.Net.create()

        if load_from:
            self.model.load_state_dict(torch.load(load_from),strict=False)

        self.dataloader = dl.LitsDataSet.create(root_dir = root_dir, batch_size = batch_size,
                                            shuffle = True, num_workers = 0)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SparseAdam(self.model.parameters(), lr=0.001,
                                            betas=(0.9, 0.999), eps=1e-08)
        self.checkpoint = checkpoint_dir
        self.num_epochs = num_epochs

    def trainData(
                    self,
                    save_per_epoch = False
                ):
        n = len(self.dataloader)
        allLoss = []
        self.train_loss = 0.0
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            self.train_loss = 0.0
            for i,sample_batched in enumerate(self.dataloader):
                inputs , label = sample_batched['scan'], sample_batched['segmentation']
                output = self.model.forward(inputs)
                loss = self.criterion(output, label)
                loss = Variable(loss, requires_grad = True)
                loss.backward()
                self.optimizer.step()

                #stat
                running_loss += loss.item()
                self.train_loss += running_loss
                if(i%1 == 0):
                    print('[%d, %10d] running_loss: %.3f' % (epoch + 1, i + 1, running_loss/1))
                    running_loss = 0
                    if i == 1:
                        break
            self.train_loss /= 1
            allLoss.append(self.train_loss)
            print('[%d] train_loss: %.10f' % (epoch + 1,self.train_loss))
            if save_per_epoch:
                torch.save(self.model.state_dict(),self.checkpoint+'semanet_'+str(epoch+1)+'.torch')
                print("Model saved at location : " + self.checkpoint+'semanet_'+str(epoch+1)+'.torch')
        np.save('allLoss',np.array(allLoss))

    def testModel(
                    self,
                    test_dir = './VolSegData_Test/'
                ):

        self.testdataloader = dl.LitsDataSet.create(root_dir = test_dir, batch_size = 1,
                                                    shuffle = False, num_workers = 0)
        n = len(self.testdataloader)
        with torch.no_grad():
            self.test_loss = 0
            for i,sample_batched in enumerate(self.testdataloader):
                inputs , label = sample_batched['scan'], sample_batched['segmentation']
                output = self.model.forward(inputs)
                self.test_loss += self.criterion(output, label).item()
                if i == 1:
                    break

            print("Train Loss %.10f and Test Loss %.10f" % (self.train_loss,self.test_loss/1))

    def saveModel(self):
        torch.save(self.model.state_dict(), self.checkpoint+'semanet.torch')
        print("Model saved at location : " + self.checkpoint)


if __name__ == '__main__':
    checkpoint_dir = os.path.join(absFilePath,'checkpoints/')
    train_data = os.path.join(absFilePath,'VolSegData/')
    test_data_dir = os.path.join(absFilePath,'VolSegData/')

    tr = Train(
                root_dir = train_data,
                checkpoint_dir = checkpoint_dir,
                batch_size = 10,
                shuffle = True,
                num_workers = 0,
                num_epochs = 2,
                load_from = None
                )

    tr.trainData(save_per_epoch = False)
    tr.saveModel()
    tr.testModel(test_dir = test_data_dir)
