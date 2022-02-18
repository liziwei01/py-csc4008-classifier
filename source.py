'''
Author: liziwei01
Date: 2022-02-18 18:25:31
LastEditors: liziwei01
LastEditTime: 2022-02-18 21:55:05
Description: file content
'''
import random
from collections import defaultdict, Counter
from datetime import datetime
from functools import reduce

g_dataset = {}
g_test_good = {}
g_test_bad = {}
NUM_ROWS = 32
NUM_COLS = 32
DATA_TRAINING = 'digit-training.txt'
DATA_TESTING = 'digit-testing.txt'
DATA_PREDICT = 'digit-predict.txt'
KNN_NEIGHBOR = 7
globalNone = ''

def read_digit(p_fp):
    bits = p_fp.read(NUM_ROWS * (NUM_COLS + 1))
    if bits == '':
        return -1, bits
    vec = [int(b) for b in bits if b != '\n']
    val = int(p_fp.readline())
    return val, vec

def load_data(p_filename=DATA_TRAINING):
    global g_dataset
    g_dataset = defaultdict(list)
    with open(p_filename) as f:
        while True:
            val,vec = read_digit(f)
            if val == -1:
                break
            g_dataset[val].append(vec)
    
def knn(p_v, size=KNN_NEIGHBOR):
    nn = []
    for d,vectors in g_dataset.items():
        for v in vectors:
            dist = round(distance(p_v,v),2)
            nn.append((dist,d))
    nn1 = sorted(nn)
    return nn1[:size]

def knn_by_closest(p_v):
    sizen = 7
    while True:
        nn = knn(p_v, sizen)
        nn1 = [str(n[1]) for n in nn]
        predict_digits = Counter(nn1)
        predict_digit = predict_digits.most_common()
        try:
            if predict_digit[0][1] == predict_digit[1][1]:
                sizen = sizen + 1
            else:
                return int(predict_digit[0][0])
        except:
            return int(predict_digit[0][0])
        
def predict(p_filename=DATA_PREDICT):
    start=datetime.now()
    print(' Beginning of Prediction @ ', start, '\n', '-'*50, '\n', ' '*16, 'Prediction Info', '\n', '-'*50)
    with open(p_filename) as f:
        while True:
            val, vec = read_digit(f)
            if val == -1:
                break
            predict_digit = knn_by_closest(vec)
            print(globalNone, predict_digit)
    stop=datetime.now()
    print(' End of Prediction @ ', stop, '\n')

def validate(p_filename=DATA_TESTING):
    global g_test_bad, g_test_good
    g_test_bad = defaultdict(int)
    g_test_good = defaultdict(int)
    start=datetime.now()
    with open(p_filename) as f:
        while True:
            val, vec = read_digit(f)
            if val == -1:
                break
            predict_digit = knn_by_closest(vec)
            # print(predict_digit)###############
            if predict_digit == val:
                g_test_good[val] += 1
            else:
                g_test_bad[val] += 1
    stop=datetime.now()
    show_test(start, stop)

def data_by_random(size=25):
    global g_dataset
    for digit in g_dataset.keys():
        g_dataset[digit] = random.sample(g_dataset[digit],size)

def distance(v, w):
    d = [(v_i - w_i)**2 for v_i, w_i in zip(v, w)]
    return sum(d)**0.5

def show_info():
    totalSamples = 0
    start=datetime.now()
    print(' Beginning of Training @ ', start, '\n', '-'*50, '\n', ' '*16, 'Training Info', '\n', '-'*50)
    for d in range(10):
        print(' '*20, d, '=', len(g_dataset[d]))
        totalSamples = totalSamples + len(g_dataset[d])
    print(globalNone, '-'*50, '\n', '  Total Samples = ', totalSamples, '\n', '-'*50, '\n')

def show_test(start = "????", stop="????"):
    print(globalNone, '-'*50, '\n', ' '*16, 'Testing Info', '\n', '-'*50)    
    Correct = 0
    Total = 0
    for d in range(10):
        good = g_test_good[d]
        bad = g_test_bad[d]
        if good + bad == 0:
            cp = 100
        else:
            cp = round(good/(good + bad)*100)
        print(' '*18, d, '  = ', '%4d,%4d,%4d'%(good, bad, cp), '%', sep='')
        Correct += good
        Total += good + bad
    accuracy = round(Correct/Total*100, 2)
    print(' ', '-'*50, '\n', ' '*5, 'Accuracy = ', accuracy, '%', '\n', '  Correct/Total = ', Correct, '/', Total, '\n', ' ', '-'*50, sep='')
    print(' End of Training @ ', stop, '\n')  

if __name__ == '__main__':
    load_data()
    show_info()
    # data_by_random()
    validate()
    predict()