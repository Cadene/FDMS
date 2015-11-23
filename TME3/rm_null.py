import numpy as np
import pandas as pd

df = pd.read_csv('./data/train.csv')
df['Id_index'] = df['Id']
df = df.set_index('Id_index')

def rm_null(seq):
    nb = len(seq['Ref'])
    if (seq['Ref'].isnull()).sum() == nb:
        return seq['Id'].values[0]
    return -1
    
df_g = df.groupby(df.index)
df_id2rm = df_g.apply(rm_null)
df = df.drop(df_id2rm.values,axis=0)

df.to_csv('./data/train_nnull.csv')

