import numpy as np
import pandas as pd

class rainFallRegressor():
    def __init__(self, eps=1e-1, nepoch=10):
        self.eps = eps
        self.nepoch = nepoch
    def resetRef(self):
        self.x1 = 'Ref'
        self.x2 = None
        self.c = 5e-3
        self.a1 = 0.625
        self.a2 = None
        self.d = 0.
    def resetRefZdr(self):
        self.x1 = 'Ref'
        self.x2 = 'Zdr'
        self.c = 5e-3
        self.a1= 1.
        self.a2=-3.
        self.d =0.
    def resetKdrZdr(self):
        self.x1 = 'Kdr'
        self.x2 = 'Zdr'
        self.c =60.
        self.a1=.91
        self.a2=-1.
        self.d =0.
    def resetKdr(self):
        self.x1 = 'Kdr'
        self.x2 = None
        self.c =32.
        self.a1=.85
        self.a2 = None
        self.d =0.
    def fit(self, data, y):
        for t in xrange(self.nepoch):
            permutation = np.random.permutation(y)
            loss = 0.
            lossE = 0.
            i = 0
            for yseq in permutation:
                xseq = data[data['Id'] == yseq[0]]
                if (self.x2):
                    powx = np.power(xseq[self.x1], self.a1) * np.power(xseq[self.x2], self.a2)
                    d = np.zeros((len(xseq), 4))
                else:
                    powx = np.power(xseq[self.x1], self.a1)
                    d = np.zeros((len(xseq), 3))
                d[:,1] = powx * xseq['dtime'] # dc
                d[:,2] = self.c * powx * np.log(xseq[self.x1]) * xseq['dtime'] # da1
                if (self.x2):
                    d[:,3] = self.c * powx * np.log(xseq[self.x2]) * xseq['dtime'] #da2
                d[:,0] = (self.c * powx + self.d) * xseq['dtime'] #pred
                aggSeq = d.sum(0)
                sign = -1 if (yseq[1] >= aggSeq[0]) else 1
                self.c = self.c - self.eps * sign * aggSeq[1]
                self.a1= self.a1- self.eps * sign * aggSeq[2]
                if (self.x2):
                    self.a2= self.a2- self.eps * sign * aggSeq[3]
                self.d = self.d - self.eps * sign
                loss += np.abs(yseq[1] - aggSeq[0])
                lossE += np.abs(yseq[1] - aggSeq[0])
                i += 1
                #if (i % 500 == 0):
                    #print i, loss / 500, self.c, self.a1, self.a2, self.d
                    #loss = 0.
            print t, lossE / len(y), self.c, self.a1, self.a2, self.d
    def predict(self,data):
        powx1 = np.power(data[self.x1], self.a1)
        if (self.x2):
            powx2 = np.power(data[self.x2], self.a2)
            powx = powx1 * powx2
        else:
            powx = powx1
        data.loc[:,'pred'] = (self.c * powx + self.d) * data['dtime']
        return data.groupby('Id').sum()['pred']
    def score(self,x,y):
        return np.abs(self.predict(x) - y).mean()
