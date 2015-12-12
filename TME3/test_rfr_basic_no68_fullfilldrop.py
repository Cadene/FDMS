import sys
import time
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor

featuresSet="basic"

t_start = time.time()

l_dfTest = []

print "Loading Model rfr_basic_no69_fullfilldrop.pkl ..."
model = pickle.load(open("./data/rfr_basic_no69_fullfilldrop.pkl", "rb"))

for i in xrange(1,2):
    print "Loading dataset basic_" + str(i) + 'sur16 into testSets...'        
    l_dfTest.append(pd.read_csv('./data/f_df_test_' + featuresSet + '_'+str(i)+'sur16.csv', index_col=0))
    
dfTest = pd.concat(l_dfTest)

dfTest = dfTest.groupby('Id').mean()

dfTest = dfTest[dfTest.isnull().any(axis=1)]
dfTest.fillna(-9999, inplace=True)

print "Predicting results..."
pred = model.predict(dfTest)

print "Saving results into pred_rfr_basic_no69_fullfilldrop.csv"
print dfTest.index
dfPred = pd.DataFrame(dfTest.index, columns=['Id'])
dfPred['Expected'] = pred
dfPred.to_csv("./data/test_pred_rfr_basic_no69_fullfilldrop.csv", index=False)

t_end = time.time()
print "Time to process: ", t_end - t_start
print "-----------------------------------------"
