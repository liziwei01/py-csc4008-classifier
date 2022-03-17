'''
Author: liziwei01
Date: 2022-02-18 18:35:53
LastEditors: liziwei01
LastEditTime: 2022-03-17 15:47:49
Description: 加载数据
'''
from array import array
from sklearn.cluster import KMeans
from collections import defaultdict
import scipy.io as sio
import numpy as np
import os

class MatLoader:
    __filenames = ['testX', 'trainX', 'trainY', 'testY']
    __dataTypes = ['train', 'test']
    __suffix = '.mat'
    __preData = defaultdict(np.ndarray)
    __data = defaultdict(dict)
    __cenrtoidData = defaultdict(dict)

    def __init__(self, p_foldernames):
        self.__foldernames = p_foldernames

    def Do(self):
        return self.__load().__squeeze().__transpose().__link().__kmeans()

    def __load(self):
        # save data to dict
        # eg: {'testX': [], 'trainX': [], 'trainY': [], 'testY': []}
        for foldername in self.__foldernames:
            for filename in self.__filenames:
                filepath = self.__getFileAbsPath(foldername, filename)
                matfile = sio.loadmat(filepath)
                matrix = matfile[filename]
                self.__preData[foldername+filename] = matrix
        return self

    def __getFileAbsPath(self, p_foldername, p_filename):
        cwd = os.getcwd()
        return os.path.join(cwd, p_foldername, p_filename + self.__suffix)

    def __squeeze(self):
        for k, v in self.__preData.items():
            self.__preData[k] = v.squeeze()
        return self

    def __transpose(self):
        # reverse the data
        # to make the data more convenient to use
        for k, v in self.__preData.items():
            self.__preData[k] = v.T
        return self

    def __link(self):
        # eg: {'ATNT face/': {
        #                    'train': {
        #                             '1': [vec1, vec2, ...],
        #                             ...,
        #                             }, 
        #                    'test': {
        #                            '1': [vec1, vec2, ...],
        #                             ...,
        #                            },
        #                    }
        #     ...
        #     }
        for foldername in self.__foldernames:
            self.__data[foldername] = {'train': {}, 'test': {}}
            for dataType in self.__dataTypes:
                mid_dict = defaultdict(list)
                size = self.__labelFile(foldername, dataType).shape[0]
                for idx in range(size):
                    classification = self.__labelFile(foldername, dataType)[idx]
                    vec = self.__vectorFile(foldername, dataType)[idx]
                    mid_dict[classification].append(vec)
                self.__data[foldername][dataType] = mid_dict
        return self

    def __kmeans(self):
        # eg: {'ATNT face/': {
        #                    'train': {
        #                             '1': vec1,
        #                             ...,
        #                             }, 
        #                    'test': {
        #                            '1': vec1,
        #                             ...,
        #                            },
        #                    }
        #     ...
        #     }
        for foldername in self.__foldernames:
            self.__cenrtoidData[foldername] = {'train': {}, 'test': {}}
            for dataType in self.__dataTypes:
                mid_dict = defaultdict(array)
                for idx in range(1, len(self.GetData(foldername, dataType))):
                    arr = self.__data[foldername][dataType][idx]
                    KMeans_model = KMeans(n_clusters=1).fit(arr)
                    mid_dict[idx] = KMeans_model.cluster_centers_[0]
                self.__cenrtoidData[foldername][dataType] = mid_dict
        return self

    def __vectorFile(self, p_foldername, dataType):
        return self.__preData[p_foldername+dataType+'X']

    def __labelFile(self, p_foldername, dataType):
        return self.__preData[p_foldername+dataType+'Y']

    def GetData(self, p_foldername, p_dataType):
        return self.__data[p_foldername][p_dataType]

    def GetCentroidData(self, p_foldername, p_dataType):
        return self.__cenrtoidData[p_foldername][p_dataType]

    def GetTestData(self, p_foldername):
        return self.__data[p_foldername]['test']

    def GetCentroidTestData(self, p_foldername):
        return self.__cenrtoidData[p_foldername]['test']

    def GetTestDataLen(self, p_foldername):
        length = 0
        for k, v in self.__data[p_foldername]['test'].items():
            length += len(v)
        return length

    def GetTrainData(self, p_foldername):
        return self.__data[p_foldername]['train']

    def GetCentroidTrainData(self, p_foldername):
        return self.__cenrtoidData[p_foldername]['train']

    def GetTrainDataLen(self, p_foldername):
        length = 0
        for k, v in self.__data[p_foldername]['train'].items():
            length += len(v)
        return length
    
    def GetVectorSet(self, p_foldername, p_dataType):
        return self.__vectorFile(p_foldername, p_dataType)

    def GetTrainVectorSet(self, p_foldername):
        return self.__vectorFile(p_foldername, 'train')

    def GetTestVectorSet(self, p_foldername):
        return self.__vectorFile(p_foldername, 'test')

    def GetFoldernames(self):
        return self.__foldernames