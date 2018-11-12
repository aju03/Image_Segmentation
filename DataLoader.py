from __future__ import print_function, division
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2

class Rescale(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,sample):
        scan , segmentation = sample['scan'], sample['segmentation']
        h , w = scan.shape
        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h , new_w = int(new_h), int(new_w)

        return {'scan' : cv2.resize(scan, (new_h,new_w)),
                'segmentation' : cv2.resize(segmentation, (new_h,new_w))}



class ToTensor(object):
    def __call__(self,sample):
        scan , segmentation = sample['scan'], sample['segmentation']
        scan = scan.reshape(1,scan.shape[0],scan.shape[1])
        segmentation = segmentation.reshape(1,segmentation.shape[0],
                                                segmentation.shape[1])

        return {'scan' : torch.from_numpy(scan),
                'segmentation' : torch.from_numpy(segmentation)}

class standardize(object):
    def __call__(self,sample):
        stdrdizde = lambda x : (x-x.min())/(x.max()-x.mean()) if not (x.min() == x.max() == 0) else x
        return  {'scan' : stdrdizde(sample['scan']),
                'segmentation' : stdrdizde(sample['segmentation'])}

class LitsDataSet(Dataset):
    """ LITS Data Set """

    def __init__(self, root_dir, transform = None):
        """
        Args
            root_dir (str): path to root_dir
            transform (callable, optional): Optional transform to be
                                            applied for dataset

        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = os.listdir(root_dir)
        self.data_list.sort(key = lambda k:int(k.split('.')[0]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,idx):
        """
        Args:
            idx (int) : index to access data
        """
        data = np.load(os.path.join(self.root_dir,self.data_list[idx]))
        scan = data[:,:512]
        segmentation = data[:,512:]
        sample = {'scan' : scan,'segmentation' : segmentation}
        if self.transform:
            sample = self.transform(sample)
        return sample

    @classmethod
    def create(_class, root_dir, batch_size = 100,
                    shuffle = True, num_workers = 0):
                    transform = transforms.Compose([
                                                    # Rescale(256),
                                                    ToTensor(),
                                                    standardize()
                                                    ])
                    # transform.Normalize(mean = [0], std = [1])
                    transformed_dataset = _class(root_dir = root_dir,
                                                transform = transform)

                    dataloader = DataLoader(transformed_dataset, batch_size = batch_size,
                                            shuffle = shuffle, num_workers = num_workers)

                    return dataloader


if __name__ == '__main__':
    transformed_dataset = LitsDataSet(root_dir='VolSegData/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               ToTensor()
                                           ]))
    transformed_dataset.Normalize(mean = [0], std = [1])

    dataloader = DataLoader(transformed_dataset, batch_size=1000,
                        shuffle=True, num_workers=0)


    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['scan'].size(),
          sample_batched['segmentation'].size())

        # observe 4th batch and stop.
        # if i_batch == 3:
        #     break
