from __future__ import print_function

import sys

import numpy as np
import pickle
from collections import defaultdict
import random
import time

from pyspark import SparkContext

def readEpisodes(episode):
    ep = episode[:-3]
    ep = np.array([ept.split(":") for ept in ep.split(";")], float)
    return np.array(ep, int)

def expectation(episode, preds):
    times = np.unique(episode[:,1])
    users = episode[:,0]
    p = np.zeros((len(times),len(users)))
    p[0,episode[:,1] == times[0]] = 1
    for t in xrange(1,len(times)):
        for u,user in enumerate(users):
            hasPreds = False
            pdtu = 1.
            for v in episode[episode[:,1] < times[t]][:,0]:
                if (v in preds[user]):
                    hasPreds = True
                    pdtu = pdtu * (1 - preds[user][v])
            p[t,u] = 1-pdtu if hasPreds else 1
    return (episode, p)

def maximization(episode_p, preds):
    episode = episode_p[0]
    p = episode_p[1]
    dplus = np.zeros((len(preds), len(preds)))
    dminus = np.zeros((len(preds), len(preds)))
    theta  = np.zeros((len(preds), len(preds)))
    times = list(np.unique(episode[:,1]))
    users = episode[:,0]
    for u,uId in enumerate(users):
        dminus[uId,:] = 1
        for v,vId in enumerate(users):
            dminus[uId, vId] = 0
            if (episode[v,1] > episode[u,1]):
                dplus[uId, vId] = 1
                tv = times.index(episode[v,1])
                theta[uId, vId] = theta[uId, vId] + (preds[vId][uId] / p[tv, v])
    return (episode, p, np.array([theta, dplus, dminus]))

def vraisemblance(episode_p_theta_dp_dm, theta, preds):
    episode = episode_p_theta_dp_dm[0]
    p       = episode_p_theta_dp_dm[1]
    dplus   = episode_p_theta_dp_dm[2][1]
    dminus  = episode_p_theta_dp_dm[2][2]
    times = list(np.unique(episode[:,1]))
    users = list(episode[:,0])
    l = 0
    log = np.log(theta)
    colog = np.log(1 - theta)
    for u,uId in enumerate(users):
        for vId in range(len(theta[uId])):
            if uId in preds[vId]:
                if dplus[uId, vId] == 1:
                    v = users.index(vId)
                    tv = times.index(episode[v,1])
                    div = (preds[vId][uId] / p[tv, v])
                    l += (div * log[uId, vId] + ((1 - div) * colog[uId, vId]))
                if dminus[uId, vId] == 1:
                    l += colog[uId, vId]
    return l
    
# MAP ----------------------------------------------------------------------
def inference(s0, succs):
    infected = defaultdict(bool)
    s = []
    s.append(s0)
    t = 1
    stop = False
    while s[t-1] != []:
        s.append([])
        for i in s[t-1]:
            for j in succs[i].keys():
                if (not infected[j]) and (random.random() < succs[i][j]):
                    infected[j] = True
                    s[t].append(j)
        t = t + 1
    return s, infected

def predict(s0, succs, nIter=100):
    suminfected = defaultdict(float)
    for i in xrange(nIter):
        _, infected = inference(s0, succs)
        for j in infected.keys():
            suminfected[j] = suminfected[j] + infected[j]
    for j in xrange(len(suminfected)):
        suminfected[j] = suminfected[j] / nIter
    return suminfected

def scoreMAP(episode, succs):
    times = np.unique(episode[:,1])
    users = episode[:,0]
    sources = users[[episode[:,1] == times[0]]]
    pred = predict(sources, succs)
    rank = np.array(pred.keys())[(-np.array(pred.values())).argsort()]
    scoreEp = 0
    count = 0.0
    for i,u in enumerate(rank):
        if u in users:
            count += 1.0
            scoreEp += count / (i+1)
    score = scoreEp / len(users)            
    return score

# Main --------------------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: IC <file> <iterations> <delta>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="PythonIC")
    episodes = sc.textFile(sys.argv[1]).map(readEpisodes).cache()
    preds = pickle.load(open("/Vrac/3000693/FDMS/preds.pkl",'r'))
    succs = pickle.load(open("/Vrac/3000693/FDMS/succs.pkl",'r'))
    
    iterations = int(sys.argv[2])
    delta = float(sys.argv[3])

    stop = False
    it = 0
    clocks = []
    while not stop:
        #clockstart = time.clock()
        clockstart = time.time()
        episodes_p = episodes.map(lambda x:expectation(x, preds))
        episode_p_theta_dp_dm = episodes_p.map(lambda x:maximization(x, preds))
        theta_dp_dm = episode_p_theta_dp_dm.map(lambda x:x[2])
        sum_theta_dp_dm = theta_dp_dm.reduce(lambda x,y:x+y)
        theta = sum_theta_dp_dm[0] / (sum_theta_dp_dm[1] + sum_theta_dp_dm[2])
        l = episode_p_theta_dp_dm.map(lambda x:vraisemblance(x, theta, preds))
        sum_l = l.reduce(lambda x,y:x+y)
       
        for u in preds:
            for v in preds[u]:
                preds[u][v] = theta[v,u]
                succs[v][u] = theta[v,u]
        print(it, sum_l)
        if (it != 0) and ((it == iterations) or (sum_l - prev_sum_l < delta)):
            stop = True
        it = it+1
        prev_sum_l = sum_l
        #clocks.append(time.clock() - clockstart)
        clocks.append(time.time() - clockstart)
    #scores = episodes.map(lambda x:scoreMAP(x,succs))
    #print(scores.reduce(lambda x,y:x+y) / scores.count())

    print(np.mean(clocks))
    #    23.85138599999998
    #[1] 24.3687281609
    #[4] 15.1469723384



