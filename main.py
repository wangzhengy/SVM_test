import gzip
import pickle
import time
import torch
import numpy as np
from sklearn import svm

T1 = time.time()

f = gzip.open('./data/mnist.pkl.gz','rb')
load_training_data ,load_validation_data,load_test_data = pickle.load(f,encoding='bytes')
f.close()

training_data,training_label,test_data,test_label = \
    load_training_data[0],load_training_data[1],load_test_data[0],load_test_data[1]

# model = svm.SVC(C=0.001,kernel='',gamma='auto')
model = svm.SVC(C=0.05,kernel='linear',gamma='auto')

model.fit(training_data,training_label)
Z = model.predict(test_data)

val_acc = np.sum(Z == test_label)/len(Z)

T2 = time.time()

print('程序运行时间:%s秒' % ((T2 - T1)))

print('acc:',val_acc)