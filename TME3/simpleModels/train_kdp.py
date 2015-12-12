import numpy as np
import pandas as pd

from model import rainFallRegressor

import time

dataset = "kdp"

print "Model: ", dataset

dfTrain = pd.read_csv("./data/xTrain_" + dataset + ".csv")
yTrain = pd.read_csv("./data/yTrain_" + dataset + ".csv")

tstart = time.time()

marshallPalmer = rainFallRegressor(eps=1e-5, nepoch=8)
marshallPalmer.resetKdp()
marshallPalmer.fit(dfTrain, yTrain)

tstop = time.time()

print 'Time: ', tstop - tstart, ' sec'

