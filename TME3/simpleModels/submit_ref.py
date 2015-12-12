import numpy as np
import pandas as pd

dataset = "ref"

from model import rainFallRegressor


dfTest = pd.read_csv("./data/xTest_" + dataset + ".csv")

p =  [0.0687952041995, 0.496971234005, 0., 0.19045]

marshallPalmer = rainFallRegressor(eps=1e-5, nepoch=15)
marshallPalmer.resetRef()
marshallPalmer.c = p[0]
marshallPalmer.a1= p[1]
#marshallPalmer.a2= p[2]
marshallPalmer.d = p[3]

pred = marshallPalmer.predict(dfTest)

dfPred = pd.DataFrame(range(717626), columns=[['Expected']])
dfPred.drop(0, inplace=True)
dfPred.Expected = pred
dfPred.fillna(dfPred.mean(), inplace=True)
dfPred['Id'] = dfPred.index
dfPred = dfPred[['Id','Expected']]

dfPred.to_csv("./data/submit_" + dataset + ".csv", index=False)

# BEST:
# [-0.0056959094655 ,0.264798269811  ,-2.46924886623 , 1.82866      ]
# train 23.2606354012
# valid 21.4594944525
