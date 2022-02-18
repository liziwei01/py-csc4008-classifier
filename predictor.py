'''
Author: liziwei01
Date: 2022-02-18 21:09:15
LastEditors: liziwei01
LastEditTime: 2022-02-18 21:53:47
Description: file content
'''
from collections import defaultdict, Counter
from loader import MatLoader

class Predictor:
    knn_neighbor = 5
    g_test_bad = defaultdict(int)
    g_test_good = defaultdict(int)
    
    def __init__(self, p_loader=MatLoader):
        self.loader = p_loader

    def do(self):
        self.loader.do()

    def test(self):
        # for foldername in self.loader.foldernames:
        foldername = 'ATNT face/'
        test_data = self.loader.data[foldername]['test']
        for d, vectors in test_data.items():
            for v in vectors:
                predict_digit = self.predict(v)
                if predict_digit == d:
                    self.g_test_good[d] += 1
                else:
                    self.g_test_bad[d] += 1

    def predict(self, p_v):
        return self.knn_by_closest(p_v)

    def knn_by_closest(self, p_v):
        sizen = self.knn_neighbor
        while True:
            nearest_neighbors_with_distance = self.knn(p_v, sizen)
            nearest_neighbors = [str(n[1]) for n in nearest_neighbors_with_distance]
            predict_digits = Counter(nearest_neighbors)
            predict_digit = predict_digits.most_common()
            try:
                if predict_digit[0][1] == predict_digit[1][1]:
                    sizen = sizen + 1
                else:
                    return int(predict_digit[0][0])
            except:
                return int(predict_digit[0][0])

    def knn(self, p_v, size=knn_neighbor):
        neighbors_with_distance = []
        foldername = 'ATNT face/'
        data = self.loader.data[foldername]['train']
        for d, vectors in data.items():
            for v in vectors:
                dist = round(self.distance(p_v, v), 2)
                neighbors_with_distance.append((dist, d))
        nearest_neighbors_with_distance = sorted(neighbors_with_distance)
        return nearest_neighbors_with_distance[:size]

    def distance(self, v, w):
        d = [(v_i - w_i)**2 for v_i, w_i in zip(v, w)]
        return sum(d)**0.5

if __name__ == '__main__':
    p = Predictor(p_loader=MatLoader(['ATNT face/', 'Binalpha handwritten/']))
    p.do()
    p.test()
    print(p.g_test_good)
    print(p.g_test_bad)