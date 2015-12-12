import numpy as np
import pandas as pd

#df = pd.read_csv('./data/train.csv')
df = pd.read_csv('./data/test.csv')
df['Id_index'] = df['Id']
df = df.set_index('Id_index')

f_ref = ['Ref', 'Ref_5x5_10th', 'Ref_5x5_50th',
       'Ref_5x5_90th', 'RefComposite', 'RefComposite_5x5_10th',
       'RefComposite_5x5_50th', 'RefComposite_5x5_90th', 'RhoHV',
       'RhoHV_5x5_10th', 'RhoHV_5x5_50th', 'RhoHV_5x5_90th']

for f_name in f_ref:
    df.loc[df[f_name] < 0, f_name] = np.nan

def rm_null(seq):
    nb = len(seq['Ref'])
    if (seq['Ref'].isnull()).sum() == nb:
        return seq['Id'].values[0]
    return -1
    
df_g = df.groupby(df.index)
df_id2rm = df_g.apply(rm_null)
df = df.drop(df_id2rm.values,axis=0)

#df.to_csv('./data/train_nnull.csv')
df.to_csv('./data/test_nnull.csv')

