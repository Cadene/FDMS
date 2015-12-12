# coding=utf-8

import sys
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier

t_start = time.time()

l_dfTrain = []
for i in xrange(1,10):
        print "Loading dataset " + str(i) + 's16...'
        l_dfTrain.append(pd.read_csv('./data/f_df_train_basic_'+str(i)+'sur16.csv', index_col=0))

dfTrain = pd.concat(l_dfTrain)
dfTrain = dfTrain.apply(lambda x: x.fillna(x.mean()), axis=0)
y_train = np.where(dfTrain.Expected > 100, 1, -1)

dfTrain.drop('Expected',1,inplace=True)
#dfTrain = dfTrain[y_train == 1]


#print 'Training One Class SVM...'
#clf = OneClassSVM(kernel="rbf")
#clf.fit(dfTrain)

print "Training SVM..."
clf = SVC(C=1, kernel="rbf", class_weight='balanced')
clf.fit(dfTrain, y_train)
#print "Training Random Forest..."
#clf = RandomForestClassifier(class_weight="balanced")
#clf.fit(dfTrain, y_train)


l_dfTest = []
for i in xrange(10,16):
        print "Loading dataset " + str(i) + 's16...'        
        l_dfTest.append(pd.read_csv('./data/f_df_train_basic_'+str(i)+'sur16.csv', index_col=0))

dfTest = pd.concat(l_dfTest)
dfTest = dfTest.apply(lambda x: x.fillna(x.mean()), axis=0)
y_test = np.where(dfTest.Expected > 100, 1, -1)
dfTest.drop('Expected',1,inplace=True)

t_end = time.time()
print "Time to process: ", t_end - t_start

print 'Predicting Test values...'
predTest = clf.predict(dfTest)

conf = np.zeros((2,2))
for i in xrange(len(y_test)):
        if (y_test[i] ==  1) and (predTest[i] ==  1):
                conf[0,0] += 1
        if (y_test[i] == -1) and (predTest[i] == -1):
                conf[1,1] += 1
        if (y_test[i] ==  1) and (predTest[i] == -1):
                conf[0,1] += 1
        if (y_test[i] == -1) and (predTest[i] ==  1):
                conf[1,0] += 1
print "Score test :", (predTest == y_test).mean()
print conf

#Loading dataset 1s16...
#Training SVM...
#Loading dataset 5s16...
#Time to process:  511.004703999
#Score test : 0.994597904774
#[[     0.    247.]
# [     0.  45476.]]


#Loading dataset 1sur16...
#Loading dataset 2sur16...
#Loading dataset 3sur16...
#Loading dataset 4sur16...
#Loading dataset 5sur16...
#Loading dataset 6sur16...
#Loading dataset 7sur16...
#Training One Class SVM...
#Loading dataset 8sur16...
#Loading dataset 9sur16...
#Time to process:  119.112707853
#Predicting Test values...
#Score test : 0.999978129408
#[[  0.00000000e+00   2.00000000e+00]
# [  0.00000000e+00   9.14450000e+04]]
