import sys
import time
import numpy as np
import pandas as pd

from sklearn.svm import SVR

featuresSet="basic"
#featuresSet="interpolate"

t_start = time.time()

model = SVR(C=float(sys.argv[1]))

l_dfTrain = []
l_dfTest = []
testSets = sys.argv[2:]

print "Model: SVR(C=", sys.argv[1] +")"

for i in xrange(1,10+1):
    if not str(i) in testSets:
        print "Loading dataset basic_" + str(i) + 'sur16 into trainSets...'        
        l_dfTrain.append(pd.read_csv('./data/f_df_train_' + featuresSet + '_'+str(i)+'sur16.csv', index_col=0))
dfTrain = pd.concat(l_dfTrain)
dfTrain.dropna(inplace=True)
yTrain = dfTrain.Expected
dfTrain.drop('Expected',axis=1,inplace=True)
dfTrain.drop('Id',axis=1,inplace=True)

print "Training SVR with rbf Kernel..."
model.fit(dfTrain, yTrain)

print "Test Sets:", testSets
for i in testSets:
    print "Loading dataset basic_" + str(i) + 'sur16 into testSets...'        
    l_dfTest.append(pd.read_csv('./data/f_df_train_' + featuresSet + '_'+str(i)+'sur16.csv', index_col=0))
dfTest = pd.concat(l_dfTest)
dfTest.dropna(inplace=True)
yTest = dfTest.Expected
dfTest.drop('Expected',axis=1,inplace=True)
dfTest.drop('Id',axis=1,inplace=True)

print "Predicting values..."
pred = model.predict(dfTest)

print "Calculating MAE scores:"
scores = (np.abs(pred - yTest)).mean()
print scores
print ""

t_end = time.time()
print "Time to process: ", t_end - t_start
print "-----------------------------------------"
