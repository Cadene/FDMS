import sys
import time
import numpy as np
import pandas as pd
import pickle

from sklearn.svm import SVR

i = sys.argv[1]

featuresSet="basic"

t_start = time.time()

model = SVR(C=.1)

l_dfTest = []

print "Loading Model svr_basic_no100_full.pkl ..."
model = pickle.load(open("./data/svr_basic_no100_full.pkl", "rb"))

#for i in xrange(1,17):
print "Loading dataset basic_" + str(i) + 'sur16 into testSets...'        
l_dfTest.append(pd.read_csv('./data/f_df_test_' + featuresSet + '_'+str(i)+'sur16.csv', index_col=0))


dfTest = pd.concat(l_dfTest)

dfTest.fillna(dfTest.mean(), inplace=True)

ids_test =  dfTest['Id']
dfTest.drop('Id',axis=1,inplace=True)

print "Predicting results..."
pred = model.predict(dfTest)

print "Saving results into pred_svr_basic_no100_full" + str(i) + "sur16.csv"
dfPred = pd.DataFrame(ids_test, columns=['Id'])
dfPred['Expected'] = pred
dfPred.to_csv("./data/test_pred_svr_basic_no100_full_"+ str(i) + "sur16.csv", index=False)

t_end = time.time()
print "Time to process: ", t_end - t_start
print "-----------------------------------------"
