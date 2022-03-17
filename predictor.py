'''
Author: liziwei01
Date: 2022-02-18 21:09:15
LastEditors: liziwei01
LastEditTime: 2022-03-17 17:03:40
Description: 预测
'''
from collections import defaultdict, Counter
from loader import MatLoader
import csv

class Predictor:
    knn_neighbor = 5
    accuracy = 4
    maxAccuracyScore = 0
    bestSize = 0
    
    def __init__(self, p_loader=MatLoader, p_generateFile=False):
        self.loader = p_loader
        self.generateFile = p_generateFile

    def Do(self):
        self.loader.Do()

    def IterateKNN(self):
        print('knn')
        for size in range(3, 16):
            print('size: {}'.format(size))
            for foldername in self.loader.__foldernames:
                print('foldername: {}'.format(foldername))
                self.__test(self.__knnByClosest, foldername, size)

    def IterateDensityBased(self):
        print('densityBased')
        for foldername in self.loader.GetFoldernames():
            print('foldername: {}'.format(foldername))
            maxDis = self.__maxDistance(foldername, 'test')
            # maxDis = 9
            print('maxDis: {}'.format(maxDis))
            radius = 1
            while radius <= maxDis:
                print('radius: {}'.format(radius))
                self.__test(self.__densityBased, foldername, radius)
                radius = radius + 1

    def IterateCentroid(self):
        print('centroid')
        for foldername in self.loader.GetFoldernames():
            print('foldername: {}'.format(foldername))
            self.__test(self.__centroid, foldername)


    def __maxDistance(self, p_foldername, dataType='train'):
        set = self.loader.GetVectorSet(p_foldername, dataType)
        max_distance = 0
        for v in set:
            for w in set:
                dist = self.__distance(v, w)
                if dist > max_distance:
                    max_distance = dist
        return max_distance

    def __test(self, f, p_foldername='ATNT face/', size=knn_neighbor):
        test_data = self.loader.GetTestData(p_foldername)
        g_test_bad = defaultdict(int)
        g_test_good = defaultdict(int)
        predict_digits = defaultdict(int)
        for d, vectors in test_data.items():
            for v in vectors:
                predict_digit = f(p_foldername, v, size)
                predict_digits[d] = predict_digit
                if predict_digit == d:
                    g_test_good[d] += 1
                else:
                    g_test_bad[d] += 1
        self.__analyze(f, size, p_foldername, predict_digits, g_test_good, g_test_bad)

    def __analyze(self, f, size, p_foldername, predict_digits, g_test_good, g_test_bad):
        print('{}'.format(p_foldername))
        print('predict_digits: {}'.format(predict_digits))
        print('good: {}'.format(g_test_good))
        print('bad: {}'.format(g_test_bad))
        good_sum = sum(g_test_good.values())
        bad_sum = sum(g_test_bad.values())
        accuracyScore = round(float(good_sum) / float(good_sum+bad_sum), self.accuracy)
        if accuracyScore > self.maxAccuracyScore:
            self.maxAccuracyScore = accuracyScore
            self.bestSize = size
            print('bestSizeChanged: {}'.format(size))
            print('-'*50)
        print('accuracy: {}'.format(accuracyScore))
        print('maxAccuracy: {}'.format(self.maxAccuracyScore))
        print('bestSize: {}'.format(self.bestSize))
        print('\n')
        if self.generateFile:
            with open(f.__name__+p_foldername[:len(p_foldername)-1]+'.csv', 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([size, accuracyScore])

    def __knnByClosest(self, foldername, p_v, size=knn_neighbor):
        while True:
            nearest_neighbors_with_distance = self.__nearestNeighborsWithDistanceBySize(foldername, p_v, size)
            nearest_neighbors = []
            for neighbor in nearest_neighbors_with_distance:
                nearest_neighbors.append(neighbor[1])
            predict_digits = Counter(nearest_neighbors)
            predict_digit = predict_digits.most_common()
            if len(predict_digit) == 0:
                raise Exception('predict_digit is empty')
            elif len(predict_digit) == 1:
                return predict_digit[0][0]
            elif predict_digit[0][1] == predict_digit[1][1]:
                size = size + 1
            else:
                return predict_digit[0][0]

    def __densityBased(self, foldername, p_v, radius):
        neighbors_within_radius = self.__nearestNeighborsWithDistanceByRadius(foldername, p_v, radius)
        nearest_neighbors = []
        for neighbor in neighbors_within_radius:
            nearest_neighbors.append(neighbor[1])
        predict_digits = Counter(nearest_neighbors)
        predict_digit = predict_digits.most_common()
        if len(predict_digit) == 0:
            # raise Exception('predict_digit is empty')
            return -1
        elif len(predict_digit) == 1:
            return predict_digit[0][0]
        elif predict_digit[0][1] == predict_digit[1][1]:
            return predict_digit[0][0]
        else:
            return predict_digit[0][0]

    def __centroid(self, foldername, p_v, size):
        predict_digit = self.__nearestCentroid(foldername, p_v)
        return predict_digit

    def __nearestNeighborsWithDistanceBySize(self, foldername, p_v, size=knn_neighbor):
        # [(1.1, 4), (4.3, 5), (3.3, 4)...]
        # [(1.1, 4), (3.3, 4), (4.3, 5)...]
        neighbors_with_distance = []
        data = self.loader.GetTrainData(foldername)
        for d, vectors in data.items():
            for v in vectors:
                dist = round(self.__distance(p_v, v), self.accuracy)
                neighbors_with_distance.append((dist, d))
        nearest_neighbors_with_distance = sorted(neighbors_with_distance)
        return nearest_neighbors_with_distance[:size]

    def __nearestNeighborsWithDistanceByRadius(self, foldername, p_v, radius):
        neighbors_with_distance = []
        data = self.loader.GetTrainData(foldername)
        for d, vectors in data.items():
            for v in vectors:
                dist = round(self.__distance(p_v, v), self.accuracy)
                if dist == 0:
                    continue
                elif dist <= radius:
                    neighbors_with_distance.append((dist, d))
        if len(neighbors_with_distance) == 0:
            print('when radius is {}, there is no neighbor'.format(radius))
        return neighbors_with_distance

    def __nearestCentroid(self, foldername, p_v):
        centroids_with_distance = []
        data = self.loader.GetCentroidTrainData(foldername)
        for d, centroid in data.items():
            dist = round(self.__distance(p_v, centroid), self.accuracy)
            centroids_with_distance.append((dist, d))
        nearest_centroid = sorted(centroids_with_distance)
        return nearest_centroid[0][1]

    def __distance(self, v, w):
        d = []
        for v_i, w_i in zip(v, w):
            d.append((float(v_i) - float(w_i)) ** 2)
        return sum(d)**0.5