import numpy as np
import pandas as pd

def convertTime(df):
    time = df['minutes_past']
    dtime = np.zeros_like(time)
    seqId = df['Id'].iloc[0]
    dtime[0] = time.iloc[0]
    for n in xrange(1,len(time)):
        # Si on change de séquence:
        if (seqId != df['Id'].iloc[n]):
            #Completer la sequence précédente:
            dtime[n-1] = dtime[n-1] + 60 - time.iloc[n-1]
            #Initialiser la nouvelle séquence:
            seqId = df['Id'].iloc[n]
            dtime[n] = time.iloc[n]
        else:
            dtime[n] = time.iloc[n] - time.iloc[n-1]
    dtime = dtime / 60.
    df.loc[:,'dtime'] = dtime
    df.drop('minutes_past', inplace=True)

# Train
dfTrain = pd.read_csv('./data/train_nnull.csv')
dfTrain['Id_index'] = dfTrain['Id']
dfTrain = dfTrain.set_index('Id_index')
dfTrain = dfTrain[['Id', 'minutes_past', 'Ref', 'RhoHV', 'Zdr', 'Kdp', 'Expected']]
dfTrain['Zdr'] = np.power(10, dfTrain['Zdr'] * 0.1)
dfTrain['Ref'] = np.power(10, dfTrain['Ref'] * 0.1)
dfTrain['Kdp'] = np.power(10, dfTrain['Kdp'] * 0.1)

# Test
dfTest = pd.read_csv('./data/test_nnull.csv')
dfTest['Id_index'] = dfTest['Id']
dfTest = dfTest.set_index('Id_index')
dfTest = dfTest[['Id', 'minutes_past', 'Ref', 'RhoHV', 'Zdr', 'Kdp']]
dfTest['Zdr'] = np.power(10, dfTest['Zdr'] * 0.1)
dfTest['Ref'] = np.power(10, dfTest['Ref'] * 0.1)
dfTest['Kdp'] = np.power(10, dfTest['Kdp'] * 0.1)




# Ref
dfTrain1 = dfTrain[['Id', 'minutes_past', 'Ref', 'Expected']].dropna()
Expected1 = pd.DataFrame(dfTrain.groupby('Id').mean()['Expected'])
convertTime(dfTrain1)
dfTrain1.drop('Expected', inplace=True)
dfTrain1.to_csv("./data/xTrain_ref.csv",index=False)
Expected1.to_csv("./data/yTrain_ref.csv",index='Id')

dfTest1 = dfTest[['Id', 'minutes_past', 'Ref']]
dfTest1.fillna(dfTest1.mean(), inplace=True)
convertTime(dfTest1)
dfTest1.to_csv("./data/xTest_ref.csv",index=False)


# Kdp
dfTrain1 = dfTrain[['Id', 'minutes_past', 'Kdp', 'Expected']].dropna()
Expected1 = pd.DataFrame(dfTrain.groupby('Id').mean()['Expected'])
convertTime(dfTrain1)
dfTrain1.drop('Expected', inplace=True)
dfTrain1.to_csv("./data/xTrain_kdp.csv",index=False)
Expected1.to_csv("./data/yTrain_kdp.csv",index='Id')

dfTest1 = dfTest[['Id', 'minutes_past', 'Kdp']]
dfTest1.fillna(dfTest1.mean(), inplace=True)
convertTime(dfTest1)
dfTest1.to_csv("./data/xTest_kdp.csv",index=False)

# Ref_Zdr
dfTrain1 = dfTrain[['Id', 'minutes_past', 'Ref', 'Zdr','Expected']].dropna()
Expected1 = pd.DataFrame(dfTrain.groupby('Id').mean()['Expected'])
convertTime(dfTrain1)
dfTrain1.drop('Expected', inplace=True)
dfTrain1.to_csv("./data/xTrain_ref_zdr.csv",index=False)
Expected1.to_csv("./data/yTrain_ref_zdr.csv",index='Id')

dfTest1 = dfTest[['Id', 'minutes_past', 'Ref', 'Zdr']]
dfTest1.fillna(dfTest1.mean(), inplace=True)
convertTime(dfTest1)
dfTest1.to_csv("./data/xTest_ref_zdr.csv",index=False)


# Kdp_Zdr
dfTrain1 = dfTrain[['Id', 'minutes_past', 'Kdp', 'Zdr','Expected']].dropna()
Expected1 = pd.DataFrame(dfTrain.groupby('Id').mean()['Expected'])
convertTime(dfTrain1)
dfTrain1.drop('Expected', inplace=True)
dfTrain1.to_csv("./data/xTrain_kdp_zdr.csv",index=False)
Expected1.to_csv("./data/yTrain_dkp_zdr.csv",index='Id')

dfTest1 = dfTest[['Id', 'minutes_past', 'Kdp', 'Zdr']]
dfTest1.fillna(dfTest1.mean(), inplace=True)
convertTime(dfTest1)
dfTest1.to_csv("./data/xTest_kdp_zdr.csv",index=False)
