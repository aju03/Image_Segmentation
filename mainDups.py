import SemanticConvnet as sc
import DataLoader as dl
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np
import torch
import LitsDataSets as LITS
from logLib import init_logging as initLog

printWithLogging = initLog()

absFilePath = os.path.dirname(os.path.abspath(__file__))

class Train:
    def __init__(
                    self,
                    root_dir = ["./trainBatch1/batch1/","./trainBatch2/batch2/"],
                    checkpoint_dir = './checkpoints/',
                    batch_size = 10,
                    shuffle = True,
                    num_workers = 0,
                    num_epochs = 2,
                    load_from = None
                ):

        self.model = sc.Net.create()
        self.batch_size = batch_size

        if load_from:
            self.model.load_state_dict(torch.load(load_from),strict=False)

        self.dataloader = LITS.LitsDataLoader.create(root_dir = root_dir, batch_size = batch_size,
                                            shuffle = False, num_workers = 0)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SparseAdam(self.model.parameters(), lr=0.001,
                                            betas=(0.9, 0.999), eps=1e-08)
        self.checkpoint = checkpoint_dir
        self.num_epochs = num_epochs
        self.N = 0
        self.epoch_loss = []
        self.batch_loss = []
        self.scan_loss = []
        self.test_loss = 0.0
        self.train_loss = 0.0
        self.N_test = 0

    def trainData(
                    self,
                    save_per_epoch = False,
                    save_per_batch = False,
                    save_per_scan = False
                ):
        per_epoch_loss = 0.0
        self.epoch_loss = []
        for epoch in range(self.num_epochs):

            per_scan_loss = 0.0
            self.scan_loss = []
            for i_scan, tensor_loader in enumerate(self.dataloader):

                per_batch_loss = 0.0
                self.batch_loss = []
                for m_batch, train_batch in enumerate(tensor_loader):
                    inputs, label = train_batch['scan'], train_batch['segmentation']
                    output = self.model.forward(inputs)
                    loss = self.criterion(output, label)
                    loss = Variable(loss, requires_grad = True)
                    loss.backward()
                    self.optimizer.step()

                    # print statistics
                    per_batch_loss += loss.item()
                    if m_batch % 5 == 4:
                        per_batch_loss = per_scan_loss / 5
                        printWithLogging.debug("[epoch : %10d | batch : %10d | per_batch_loss : %.10f]" % (epoch + 1, m_batch + 1, per_batch_loss ))
                        per_scan_loss += per_batch_loss
                        self.batch_loss.append(per_batch_loss)
                        per_batch_loss = 0.0

                    if save_per_batch:
                        torch.save(self.model.state_dict(),self.checkpoint+'semanet_per_batch_'+str(m_batch+1)+'.torch')

                    # if(m_batch == 1):
                        # break

                if epoch == 0:
                    self.N += len(tensor_loader)
                per_scan_loss /= len(tensor_loader)
                per_epoch_loss += per_scan_loss
                printWithLogging.debug("[epoch : %10d | scan : %10d | per_scan_loss : %.10f]" % (epoch + 1, i_scan + 1, per_scan_loss))
                self.scan_loss.append(per_scan_loss)
                per_scan_loss = 0.0

                if save_per_scan:
                    torch.save(self.model.state_dict(),self.checkpoint+'semanet_per_scan_'+str(i_scan+1)+'.torch')

                # if(i_scan == 0):
                    # break

            per_epoch_loss /= self.N

            printWithLogging.debug("[epoch : %10d | scan : %10d | per_epoc_loss : %.10f]" % (epoch + 1, i_scan + 1, per_epoch_loss))
            self.epoch_loss.append(per_epoch_loss)
            self.train_loss = per_epoch_loss
            per_epoch_loss = 0.0

            if save_per_epoch:
                torch.save(self.model.state_dict(),self.checkpoint+'semanet_per_epoch_'+str(epoch+1)+'.torch')

            # if(epoch == 0):
                # break

        np.save('train_loss',np.array([self.train_loss/self.N]))
        np.save('per_batch_loss',np.array(per_batch_loss))
        np.save("per_scan_loss",np.array(per_scan_loss))
        np.save("per_epoc_loss",np.array(per_epoch_loss))
        printWithLogging.debug("-----Done---------")

    def testModel(
                    self,
                    test_dir = ["./testBatch/testbatch/batch1/","./testBatch/testbatch/batch2/"]
                ):

        # self.testdataloader = dl.LitsDataLoader.create(root_dir = test_dir, batch_size = 1,
        #                                             shuffle = False, num_workers = 0)
        self.testdataloader = LITS.LitsDataLoader.create(root_dir = test_dir, batch_size = 1,
                                            shuffle = False, num_workers = 0)

        with torch.no_grad():
            for i , tensor_loader in enumerate(self.testdataloader):
                for j, sample_batched in enumerate(tensor_loader):
                    inputs, label = sample_batched['scan'], sample_batched['segmentation']
                    output = self.model.forward(inputs)
                    self.test_loss += self.criterion(output, label).item()

                    # if(j == 0):
                        # break

                self.N_test += len(tensor_loader)

                # if(i == 0):
                    # break


        np.save('test_loss',np.array([self.test_loss/self.N_test]))
        printWithLogging.debug("Train Loss %.10f and Test Loss %.10f" % (self.train_loss,self.test_loss/self.N_test))


    def saveModel(self):
        torch.save(self.model.state_dict(), self.checkpoint+'semanet.torch')
        printWithLogging.debug("Model saved at location : " + self.checkpoint)


if __name__ == '__main__':
    checkpoint_dir = os.path.join(absFilePath,'checkpoints/')
    # train_data = os.path.join(absFilePath,'VolSegData/')
    # test_data_dir = os.path.join(absFilePath,'VolSegData/')

    tr = Train(
                # root_dir = train_data,
                checkpoint_dir = checkpoint_dir,
                batch_size = 10,
                shuffle = True,
                num_workers = 0,
                num_epochs = 10,
                load_from = None
                )

    tr.trainData(save_per_epoch = True)
    tr.saveModel()
    tr.testModel()
