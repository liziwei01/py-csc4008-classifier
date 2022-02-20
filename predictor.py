'''
Author: liziwei01
Date: 2022-02-18 21:09:15
LastEditors: liziwei01
LastEditTime: 2022-02-20 19:45:16
Description: file content
'''
from collections import defaultdict, Counter
from loader import MatLoader

class Predictor:
    knn_neighbor = 5
    
    def __init__(self, p_loader=MatLoader):
        self.loader = p_loader

    def do(self):
        self.loader.do()
        self.iterate()

    def iterate(self):
        # size = self.knn_neighbor
        for size in range(3, 15):
            print('size: {}'.format(size))
            for foldername in self.loader.foldernames:
                self.test(foldername, size)

    def test(self, p_foldername='ATNT face/', size=knn_neighbor):
        test_data = self.loader.data[p_foldername]['test']
        g_test_bad = defaultdict(int)
        g_test_good = defaultdict(int)
        for d, vectors in test_data.items():
            for v in vectors:
                predict_digit = self.predict(p_foldername, v, size)
                if predict_digit == d:
                    g_test_good[d] += 1
                else:
                    g_test_bad[d] += 1
        self.analyze(p_foldername, g_test_good, g_test_bad)
            
    def analyze(self, p_foldername, g_test_good, g_test_bad):
        print('{}'.format(p_foldername))
        print('good: {}'.format(g_test_good))
        print('bad: {}'.format(g_test_bad))
        good_sum = sum(g_test_good.values())
        bad_sum = sum(g_test_bad.values())
        print('accuracy: {}'.format(round(good_sum / (good_sum+bad_sum), 4)))
        print('\n')

    def predict(self, foldername, p_v, size=knn_neighbor):
        return self.knn_by_closest(foldername, p_v, size)

    def knn_by_closest(self, foldername, p_v, size=knn_neighbor):
        while True:
            nearest_neighbors_with_distance = self.knn(foldername, p_v, size)
            nearest_neighbors = [str(n[1]) for n in nearest_neighbors_with_distance] # [4, 4, 4, 5, 6]
            predict_digits = Counter(nearest_neighbors)
            predict_digit = predict_digits.most_common()
            try:
                if predict_digit[0][1] == predict_digit[1][1]:
                    size = size + 1
                else:
                    return int(predict_digit[0][0])
            except:
                return int(predict_digit[0][0])

    def knn(self, foldername, p_v, size=knn_neighbor):
        # [(1.1, 4), (4.3, 5), (3.3, 4)...]
        # [(1.1, 4), (3.3, 4), (4.3, 5)...]
        neighbors_with_distance = []
        data = self.loader.data[foldername]['train']
        for d, vectors in data.items():
            for v in vectors:
                dist = round(self.distance(p_v, v), 2)
                neighbors_with_distance.append((dist, d))
        nearest_neighbors_with_distance = sorted(neighbors_with_distance)
        return nearest_neighbors_with_distance[:size]

    def distance(self, v, w):
        d = []
        for v_i, w_i in zip(v, w):
            d.append((float(v_i) - float(w_i)) ** 2)
        return sum(d)**0.5

if __name__ == '__main__':
    p = Predictor(p_loader=MatLoader(['ATNT face/', 'Binalpha handwritten/']))
    p.do()
    print(p.g_test_good)
    print(p.g_test_bad)