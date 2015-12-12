import sys
import time
import numpy as np
import pandas as pd
import pickle

from sklearn.svm import SVR

featuresSet="basic_no100_full"

t_start = time.time()

l_dfPred = []

for i in xrange(1,17):
    print "Loading test_pred" + featuresSet + "_" + str(i) + 'sur16 ...'
    l_dfPred.append(pd.read_csv('./data/test_pred_svr_' + featuresSet + '_' + str(i) + 'sur16.csv'))
dfCat = pd.concat(l_dfPred)

print 'Saving results into pred_svr_' + featuresSet + ".csv"



dfCat = dfCat.groupby('Id').mean()


dfPred = pd.DataFrame(range(717626), columns=['Id'])
dfPred['Expected'] = dfCat.Expected

dfPred.drop(0, inplace=True)
dfPred.fillna(0., inplace=True)

dfPred.to_csv('./data/pred_svr_' + featuresSet + ".csv", index=False)



t_end = time.time()
print "Time to process: ", t_end - t_start
print "-----------------------------------------"
