import sys
import time
import numpy as np
import pandas as pd
import pickle


from sklearn.ensemble import RandomForestRegressor

featuresSet="basic"

t_start = time.time()

model = RandomForestRegressor(n_estimators=1000, max_depth=7, min_samples_split=20, n_jobs=4)

l_dfTrain = []

print "Model: Random Forest"

for i in xrange(1,17):
    print "Loading dataset basic_" + str(i) + 'sur16 into trainSets...'        
    l_dfTrain.append(pd.read_csv('./data/f_df_train_' + featuresSet + '_'+str(i)+'sur16.csv', index_col=0))
dfTrain = pd.concat(l_dfTrain)
dfTrain = dfTrain.groupby('Id').mean()

dfTrain = dfTrain[dfTrain.Expected <= 69]
dfTrain.fillna(-9999, inplace=True)
#dfTrain.dropna(inplace=True)
yTrain = dfTrain.Expected
dfTrain.drop('Expected',axis=1,inplace=True)
print dfTrain.columns


print "Training SVR with rbf Kernel..."
model.fit(dfTrain, yTrain)

print "Saving model..."
pickle.dump(model,open("./data/rfr_basic_no69_fullfill.pkl",'wb'))

#pickle.load(open("svr_basic_full.pkl",'rb'))

t_end = time.time()
print "Time to process: ", t_end - t_start
print "-----------------------------------------"
