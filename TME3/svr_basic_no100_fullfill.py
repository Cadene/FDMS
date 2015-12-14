import sys
import time
import numpy as np
import pandas as pd
import pickle

from sklearn.svm import SVR

featuresSet="basic"
#featuresSet="interpolate"

t_start = time.time()

model = SVR(C=1)

l_dfTrain = []
l_dfTest = []

print "Model: SVR(C=1)"

for i in xrange(1,17):
    print "Loading dataset basic_" + str(i) + 'sur16 into trainSets...'        
    l_dfTrain.append(pd.read_csv('./data/f_df_train_' + featuresSet + '_'+str(i)+'sur16.csv', index_col=0))
dfTrain = pd.concat(l_dfTrain)

dfTrain = dfTrain[dfTrain.Expected <= 69]
dfTrain.fillna(dfTrain.mean(), inplace=True)
#dfTrain.dropna(inplace=True)
yTrain = dfTrain.Expected
dfTrain.drop('Expected',axis=1,inplace=True)
dfTrain.drop('Id',axis=1,inplace=True)


print "Training SVR with rbf Kernel..."
model.fit(dfTrain, yTrain)

print "Saving model..."
pickle.dump(model,open("svr_basic_fullfill.pkl",'wb'))

#pickle.load(open("svr_basic_full.pkl",'rb'))

t_end = time.time()
print "Time to process: ", t_end - t_start
print "-----------------------------------------"
