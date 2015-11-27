# coding=utf-8

import sys
import time
import numpy as np
import pandas as pd

t_start = time.time()

l_df = []
for i in xrange(1,17):
	l_df.append(pd.read_csv('./data/f_df_train_'+str(i)+'on16.csv', index_col=0))

y_df = pd.read_csv('./data/y_df_train.csv', index_col=0)
l_df.append(y_df)

df = pd.concat(l_df)

## K-Means

df = df.apply(lambda x: x.fillna(x.mean()), axis=0)

ids = df.index.values
size = len(ids)
pc = .001
idTo = ids[int(size * pc)]
df_try = df.loc[:idTo]

#y_train = y[:idTo]
#X_test  = f_df[idTo:]
#y_test  = y[idTo:]

#print len(X_train), '+', len(X_test), '=', len(f_df)

from sklearn.cluster import KMeans

clf = KMeans(n_clusters=4) 

columns = df_try.columns.values
columns.remove('Id')
columns.remove('Expected')

cluster = clf.fit_predict(df_try[columns])

df_try['cluster'] = cluster

g_df = df_try.groupby('cluster', sort=True)
mean_cluster = g_df['Expected'].mean()

df_try['cluster'].to_csv('./data/km_df_train.csv')

t_end = time.time()

print "Time to process: ", t_end - t_start



