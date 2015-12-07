import numpy as np
import pandas as pd

dfTrain = pd.read_csv("./data/xTrain_kdp.csv")
yTrain = pd.read_csv("./data/yTrain_kdp.csv")

dfValid = dfTrain[dfTrain.Id > 591773]
yValid =  yTrain[yTrain.Id > 591773]
dfTrain = dfTrain[dfTrain.Id <= 591773]
yTrain =  yTrain[yTrain.Id <= 591773]

marshallPalmer = rainFallRegressor(eps=1e-5, nepoch=10)
marshallPalmer.resetRef()
marshallPalmer.fit(dfTrain, yTrain)
print marshallPalmer.score(dfValid,yValid['Expected'])
