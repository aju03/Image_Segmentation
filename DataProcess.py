#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize
from functools import reduce
import cv2


dig = lambda x : int(''.join([*filter(str.isdigit,x.split('-')[-1])]))
filtDig = lambda x : dig(x)
flatList = lambda x:[*itertools.chain.from_iterable(x)]

interPolate = lambda x: np.interp(x,(x.min(),x.max()),(0,255))
resize = lambda x,size: cv2.resize(x,(size,size))


absFilePath = os.path.dirname(os.path.abspath(__file__))

destinationFolder = os.path.join(absFilePath,'VolSegData1')

allPath = []
# ['trainBatch1/batch1/*.nii','trainBatch2/batch2/*.nii']

class DataPathProcessing(object):
    def __init__(self,pathList):
        if not isinstance(pathList,(list,tuple)):
            raise Exception('pathList must be list to nii data path')
        self.__pathList = pathList
        self.allDataPath = self.__getAllDataPath()
        self.N = self.__countExample()
        self.counter = 0

    def __getitem__(self,key):
        return self.allDataPath[key]


    def __returnSize(self,dPath):
        return nib.load(dPath).shape[0]

    def __countExample(self):
        return reduce(lambda x,y:x+y,[*map(lambda x:self.__returnSize(x[0]),self.allDataPath)])

    def __mapToTuples(self,path):
        data_path = os.path.join(absFilePath,path)
        data_train_path = glob.glob(data_path)
        data_seg = [i for i in data_train_path if 'segmentation' in i]
        data_vol = [i for i in data_train_path if 'volume' in i]
        data_seg.sort(key = filtDig)
        data_vol.sort(key = filtDig)
        data_vol_path = [(i,j) for i,j in zip(data_vol,data_seg)]
        return data_vol_path

    def __getAllDataPath(self):
        return flatList([*map(lambda x:self.__mapToTuples(x),self.__pathList)])

    def __writeToFolder(self,data_path,size = 512):
        vol = nib.load(data_path[0]).get_data()
        seg = nib.load(data_path[1]).get_data()

        # vol = interPolate(vol)
        # seg = interPolate(seg)

        for i in range(vol.shape[2]):
            if not os.path.exists(destinationFolder):
                os.makedirs(destinationFolder)
            imname = os.path.join(destinationFolder,str(self.counter))
            if size < vol.shape[0]:
                vol_resized = resize(vol[:,:,i],size)
                seg_resized = resize(seg[:,:,i],size)
            vol_resized = vol[:,:,i]
            seg_resized = seg[:,:,i]

            vol_seg = np.concatenate((vol_resized,seg_resized),axis=1)
            np.save(imname,vol_seg)
            self.counter += 1
            if (self.counter%100 == 0):
                print(str(self.counter)+'/'+str(self.N))

        print("done")

    def writeImageData(self):
        for data_path in self.allDataPath:
            self.__writeToFolder(data_path)



if __name__ == '__main__':
    # dtUtil = DataPathProcessing(['trainBatch1/batch1/*.nii','trainBatch2/batch2/*.nii'])
    dtUtil = DataPathProcessing(['trainBatch1/batch1/*.nii'])
    dtUtil.writeImageData()
