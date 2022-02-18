'''
Author: liziwei01
Date: 2022-02-18 18:28:30
LastEditors: liziwei01
LastEditTime: 2022-02-18 20:05:04
Description: file content
'''
import scipy.io as sio
import numpy as np
import os

if __name__ == "__main__":
    data = sio.loadmat('ATNT face/trainY.mat')
    print(data)