# coding=utf-8

import sys
import time
import numpy as np
import pandas as pd

t_start = time.time()

df = pd.read_csv('./data/train_nnull.csv')

df['Id_index'] = df['Id']
df = df.set_index('Id_index')
    
df_g = df.groupby(df.index)

y_df = pd.DataFrame(index=df.index.unique())

y_df['Expected'] = df_g['Expected'].mean()

y_df.to_csv('./data/y_df_train.csv')

t_end = time.time()

print "Time to process: ", t_end - t_start



