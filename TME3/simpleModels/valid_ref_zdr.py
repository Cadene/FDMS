import numpy as np
import pandas as pd

from model import rainFallRegressor

import time

dataset = "ref_zdr"

print "Model: ", dataset

dfTrain = pd.read_csv("./data/xTrain_" + dataset + ".csv")
yTrain = pd.read_csv("./data/yTrain_" + dataset + ".csv")

tstart = time.time()

subSet = yTrain.Id[int(len(yTrain) * .5)]

dfValid = dfTrain[dfTrain.Id > subSet]
yValid =  yTrain[yTrain.Id > subSet]
dfTrain = dfTrain[dfTrain.Id <= subSet]
yTrain =  yTrain[yTrain.Id <= subSet]

marshallPalmer = rainFallRegressor(eps=1e-5, nepoch=15)
marshallPalmer.resetRefZdr()
marshallPalmer.fit(dfTrain, yTrain)
print marshallPalmer.score(dfValid,yValid['Expected'])

tstop = time.time()

print 'Time: ', tstop - tstart, ' sec'
