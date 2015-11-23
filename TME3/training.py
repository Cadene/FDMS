# coding=utf-8

import sys
import time
import numpy as np
import pandas as pd

l_df = []
for i in xrange(1,17):
	l_df.append(pd.read_csv('./data/f_df_train_'+str(i)+'on16.csv'))

df = pd.concat(l_df)
df = df.set_index('Id')
df.drop('Id')

f_cols = ['radardist_km', 'Ref', 'Ref_5x5_10th', 'Ref_5x5_50th',
       'Ref_5x5_90th', 'RefComposite', 'RefComposite_5x5_10th',
       'RefComposite_5x5_50th', 'RefComposite_5x5_90th', 'RhoHV',
       'RhoHV_5x5_10th', 'RhoHV_5x5_50th', 'RhoHV_5x5_90th', 'Zdr',
       'Zdr_5x5_10th', 'Zdr_5x5_50th', 'Zdr_5x5_90th', 'Kdp',
       'Kdp_5x5_10th', 'Kdp_5x5_50th', 'Kdp_5x5_90th', 'Marshall_Palmer']

y = f_df.Expected
f_df = f_df[f_cols].apply(lambda x: x.fillna(x.mean()), axis=0)

size = len(f_df)
pc = .80
X_train = f_df[:int(size* pc)]
y_train = y[:int(size* pc)]
X_test  = f_df[int(size* pc):]
y_test  = y[int(size* pc):]
print len(X_train), '+', len(X_test), '=', len(f_df)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

def train_test(model, X_train, y_train, X_test, y_test):
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	return mean_absolute_error(y_pred, y_test)

l_model = []
l_model.append(LinearRegression)
l_model.append(Ridge)
l_model.append(SVR)
l_model.append(RandomForestRegressor)
l_model.append(ExtraTreesRegressor)
l_model.append(GradientBoostingRegressor)

for model in l_model:
	t_start = time.time()	
	print train_test(model, X_train, y_train, X_test, y_test)
	t_end = time.time()
	print t_end - t_start


