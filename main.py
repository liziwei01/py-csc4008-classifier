'''
Author: liziwei01
Date: 2022-02-18 18:23:45
LastEditors: liziwei01
LastEditTime: 2022-02-20 16:54:59
Description: file content
'''
from loader import MatLoader
from predictor import Predictor

foldernames = ['ATNT face/', 'Binalpha handwritten/']

if __name__ == "__main__":
    loader = MatLoader(p_foldernames=foldernames)
    predictor = Predictor(p_loader=loader)
    predictor.do()