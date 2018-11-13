from __future__ import print_function, division
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import nibabel as nib
import glob
from DataLoader import (ToTensor,LabelDiscritize,Standardize)


class CTDataLoader(object):
    def __init__(self, root_dir):
        nii = []
        if isinstance(root_dir,(list,tuple)):
            for path in root_dir:
                nii.extend(glob.glob(path+"volume*"))
        else:
            nii.extend(glob.glob(root_dir+"volume*"))

        self.niiTuple = [{'scan' : i, 'segmentation' : i.replace("volume","segmentation")} for i in nii]

    def __len__(self):
        return  len(self.niiTuple)

    def enumData(self):
        return self.niiTuple



class LitsDataLoader(Dataset):
    def __init__(self, nii_dict, transform = None):
        self.scan = nib.load(nii_dict['scan']).get_fdata(dtype = np.double)
        # self.scan = nib.load(nii_dict['scan']).get_data()
        self.segmentation = nib.load(nii_dict['segmentation']).get_fdata(dtype = np.double)
        # self.segmentation = nib.load(nii_dict['segmentation']).get_data()
        self.transform = transform

    def __len__(self):
        return self.scan.shape[-1]

    def __getitem__(self,idx):
        sample = {'scan' : self.scan[:,:,idx],'segmentation' : self.segmentation[:,:,idx]}

        if self.transform:
            sample = self.transform(sample)
        return sample

    @classmethod
    def create(_class, root_dir = ["./trainBatch1/batch1/","./trainBatch2/batch2/"], batch_size = 1, shuffle = True, num_workers = 0):
        transform = transforms.Compose([
                                        ToTensor(),
                                        Standardize(),
                                        LabelDiscritize(),
                                        ])
        ctdataloader = CTDataLoader(root_dir)
        for i,j in enumerate(ctdataloader.enumData()):
            yield DataLoader(_class(j,transform),
                            batch_size = batch_size,
                            shuffle = shuffle,
                            num_workers = num_workers)








# g = CTDataLoader(["./trainBatch1/batch1/","./trainBatch2/batch2/"])
if __name__ == "__main__":
    # ctdataloader = CTDataLoader(["./trainBatch1/batch1/","./trainBatch2/batch2/"])
    # for i,j in enumerate(ctdataloader.enumData()):
    #
    #     transform = transforms.Compose([
    #                                     ToTensor(),
    #                                     Standardize(),
    #                                     LabelDiscritize(),
    #                                     ])
    #     transformed_dataset = LitsDataLoader(j,transform)
    #     dataloader = DataLoader(transformed_dataset, batch_size=1,shuffle=True, num_workers=0)
    #
    #     for m,n in enumerate(dataloader):
    #         print(n['scan'].shape)
    #         print(type(n['scan']))
    #         if(m==1):
    #             break
    #     break
    d = LitsDataLoader.create(["./trainBatch1/batch1/","./trainBatch2/batch2/"])

    for i,j in enumerate(d):
        print("i "+str(i))
        # for m,n in enumerate(j):
        #     print(n['scan'].shape)
        #     print(type(n['scan']))
        #     if(m==1):
        #         break

        # if(i==4):
        #     break
