# coding=utf-8

import sys
import time
import numpy as np
import pandas as pd

t_start = time.time()

df = pd.read_csv('./data/train_nnull.csv')

df['Id_index'] = df['Id']
df = df.set_index('Id_index')

part = int(sys.argv[1])
total_part = 16
pc = 1. / total_part
ids = df.index.unique()
idFrom = int((part-1) * pc * len(ids))
idTo   = int((part) * pc * len(ids))

print "Total size of dataset:", len(ids)
print "IdFrom:", ids[idFrom], "IdTo:", ids[idTo]

df = df.loc[ids[idFrom]:ids[idTo]]

f_cols = ['radardist_km', 'Ref', 'Ref_5x5_10th', 'Ref_5x5_50th',
       'Ref_5x5_90th', 'RefComposite', 'RefComposite_5x5_10th',
       'RefComposite_5x5_50th', 'RefComposite_5x5_90th', 'RhoHV',
       'RhoHV_5x5_10th', 'RhoHV_5x5_50th', 'RhoHV_5x5_90th', 'Zdr',
       'Zdr_5x5_10th', 'Zdr_5x5_50th', 'Zdr_5x5_90th', 'Kdp',
       'Kdp_5x5_10th', 'Kdp_5x5_50th', 'Kdp_5x5_90th', 'Marshall_Palmer']

f_interpolable = ['Ref', 'Ref_5x5_10th', 'Ref_5x5_50th',
       'Ref_5x5_90th', 'RefComposite', 'RefComposite_5x5_10th',
       'RefComposite_5x5_50th', 'RefComposite_5x5_90th', 'RhoHV',
       'RhoHV_5x5_10th', 'RhoHV_5x5_50th', 'RhoHV_5x5_90th', 'Zdr',
       'Zdr_5x5_10th', 'Zdr_5x5_50th', 'Zdr_5x5_90th', 'Kdp',
       'Kdp_5x5_10th', 'Kdp_5x5_50th', 'Kdp_5x5_90th']

def marshall_palmer(ref, minutes_past):
    #print "Estimating rainfall from {0} observations".format(len(minutes_past))
    # how long is each observation valid?
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in xrange(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for dbz, hours in zip(ref, valid_time):
        # See: http s://en.wikipedia.org/wiki/DBZ_(meteorology)
        if np.isfinite(dbz):
            mmperhr = pow(pow(10, dbz/10)/200, 0.625)
            sum = sum + mmperhr * hours
    return sum

def process_mp(seq):
    #seq = seq.sort_values('minutes_past', ascending=True)
    seq = seq.sort('minutes_past', ascending=True)
    mp = marshall_palmer(seq['Ref'], seq['minutes_past'])
    return mp

def process_mp_intrpl(seq):
    #seq = seq.sort_values('minutes_past', ascending=True)
    seq = seq.sort('minutes_past', ascending=True)
    mp = marshall_palmer(seq['Ref'].interpolate(), seq['minutes_past'])
    return mp

def interpolate(seq):
    seq = seq.interpolate()
    return seq
    
# création de colonnes booléennes : True si NaN
for f_name in f_interpolable:
    df[f_name + '_isnull'] = df[f_name].isnull()
    
df_g = df.groupby(df.index)

f_df = pd.DataFrame(index=df.index.unique())

f_df['length'] = df_g.size()
f_df['radardist_km'] = df_g['radardist_km'].mean() # tous les éléments d'une séquence sont égaux
f_df['Id'] = df_g['Id'].mean()

for f_name in f_interpolable:
    f_df[f_name + '_nbNaN'] = df_g[f_name + '_isnull'].sum()
    f_df[f_name + '_isNaN'] = f_df[f_name + '_nbNaN'] == f_df['length']
    f_df[f_name + '_pcNaN'] = f_df[f_name + '_nbNaN'] * 1. / f_df['length']
    f_df.drop(f_name + '_nbNaN', axis=1)
    
f_df['Marshall_Palmer'] = df_g[['Ref','minutes_past']].apply(process_mp)
for f_name in f_interpolable:
    f_df[f_name + '_mean'] = df_g[f_name].mean()
    f_df[f_name + '_std'] = df_g[f_name].std()
    f_df[f_name + '_min'] = df_g[f_name].min()
    f_df[f_name + '_max'] = df_g[f_name].max()
    for pc in [.1,.3,.5,.7,.9]:
        f_df[f_name + '_qtil_' + str(pc)] = df_g[f_name].quantile(pc)

f_df['Marshall_Palmer_intrpl'] = df_g[['Ref','minutes_past']].apply(process_mp_intrpl)
df_g_intrpl = df_g[f_interpolable].apply(interpolate).groupby(df.index) 
for f_name in f_interpolable: 
    f_df[f_name + '_intrpl_mean'] = df_g_intrpl[f_name].mean()
    f_df[f_name + '_intrpl_std'] = df_g_intrpl[f_name].std()
    f_df[f_name + '_intrpl_min'] = df_g_intrpl[f_name].min()
    f_df[f_name + '_intrpl_max'] = df_g_intrpl[f_name].max()
    for pc in [.1,.3,.5,.7,.9]:
        f_df[f_name + '_intrpl_qtil_' + str(pc)] = df_g_intrpl[f_name].quantile(pc)

f_df.to_csv('./data/f_df_train_' + str(part) + 'on' + str(total_part) + '.csv')

t_end = time.time()

print "Time to process: ", t_end - t_start



