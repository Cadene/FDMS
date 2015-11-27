# coding=utf-8

import sys
import time
import numpy as np
import pandas as pd

t_start = time.time()

f_df = pd.read_csv('./data/train_nnull.csv')

f_df = df.set_index('Id')

## K-Means

y = f_df.Expected
f_df = f_df[f_cols].apply(lambda x: x.fillna(x.mean()), axis=0)

size = len(f_df)
pc = .80
X_train = f_df[:int(size* pc)]
y_train = y[:int(size* pc)]
X_test  = f_df[int(size* pc):]
y_test  = y[int(size* pc):]
print len(X_train), '+', len(X_test), '=', len(f_df)

#f_df.to_csv('./data/f_df_train_' + str(part) + 'on' + str(total_part) + '.csv')

t_end = time.time()

print "Time to process: ", t_end - t_start



