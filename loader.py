'''
Author: liziwei01
Date: 2022-02-18 18:35:53
LastEditors: liziwei01
LastEditTime: 2022-02-18 21:43:48
Description: file content
'''
from collections import defaultdict
import scipy.io as sio
import numpy as np
import os

class MatLoader:
    filenames = ['testX', 'trainX', 'trainY', 'testY']
    data_types = ['train', 'test']
    suffix = '.mat'
    pre_data = defaultdict(np.ndarray)
    data = defaultdict(dict)

    def __init__(self, p_foldernames):
        self.foldernames = p_foldernames

    def do(self):
        return self.load().squeeze().transpose().link()

    def load(self):
        # save data to dict
        # eg: {'testX': [], 'trainX': [], 'trainY': [], 'testY': []}
        for foldername in self.foldernames:
            for filename in self.filenames:
                filepath = self.__get_file_abs_path(foldername, filename)
                matfile = sio.loadmat(filepath)
                matrix = matfile[filename]
                self.pre_data[foldername+filename] = matrix
        return self

    def __get_file_abs_path(self, p_foldername, p_filename):
        cwd = os.getcwd()
        return os.path.join(cwd, p_foldername, p_filename + self.suffix)

    def squeeze(self):
        for k, v in self.pre_data.items():
            self.pre_data[k] = v.squeeze()
        return self

    def transpose(self):
        # reverse the data
        # to make the data more convenient to use
        for k, v in self.pre_data.items():
            self.pre_data[k] = v.T
        return self

    def link(self):
        # eg: {'ATNT face': {
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
        for foldername in self.foldernames:
            self.data[foldername] = {'train': {}, 'test': {}}
            for data_type in self.data_types:
                mid_dict = defaultdict(list)
                size = self.pre_data[foldername+data_type+'Y'].shape[0]
                for idx in range(size):
                    classification = self.pre_data[foldername+data_type+'Y'][idx]
                    vec = self.pre_data[foldername+data_type+'X'][idx]
                    mid_dict[classification].append(vec)
                self.data[foldername][data_type] = mid_dict
        return self


if __name__ == "__main__":
    loader = MatLoader(['ATNT face/']).do()
    # print(loader.data)


