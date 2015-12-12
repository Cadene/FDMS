import pandas as pd

dfPredSVR = pd.read_csv('./data/pred_svr_basic_no100_full.csv', index_col='Id')
dfPredRFR = pd.read_csv('./data/test_pred_rfr_basic_no69_fullfilldrop.csv', index_col='Id')

dfPredSVR["Expected"][dfPredRFR.index] = dfPredRFR["Expected"]

dfPredSVR.to_csv('./data/pred_svr_no100_full_rfr_no69_drop.csv', index='Id')
