
# coding: utf-8

# # TME3: How Much Did It Rain? II
# 
# https://www.kaggle.com/c/how-much-did-it-rain-ii
# 
# 
# Car cancellation guide:
# https://github.com/numb3r33/Kaggle-Competitions/blob/master/cars-cancellation/cars_cancellation.ipynb
# 
# 
# En pluviométrie, pour mesurer les hauteurs des précipitations on utilise des jauges qui receuillent la pluie. On est alors capable de déterminer la quantité d'eau tombée durant un intervalle de temps donné, ici une heure.
# Cependant, les jauges ne peuvent pas couvrir l'ensemble des lieux que l'on souhaite observer. On utilise alors des radars et on estime la hauteur des précipitations à partir de leurs relevés.
# Cependant, ces estimations correspondent mal aux mesures effectués sur les jauges.
# L'objectif de ce challenge est de fournir un meilleur estimateur basée sur les relevés des radars.

# # Imports

# In[1]:

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#import seaborn as sb
get_ipython().magic(u'matplotlib inline')

from sklearn import linear_model
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error


# # Chargement des données
# 
# Les données sont des séquences de relevés de capteurs sur une durée d'une heure à des temps variables.
# A chaque séquence est associée un identifiant et la distance du capteur à une jauge dont il faut prédire les mesures à la fin de l'heure.
# Un élément de la séquence contient les différentes mesures effectués par le radar à un instant donné.

# In[2]:

get_ipython().run_cell_magic(u'time', u'', u"dfTrain = pd.read_csv('./data/train.csv')")


# In[3]:

dfTest = pd.read_csv('./data/test.csv')
dfTest = dfTest.set_index('Id')


# In[4]:

dfTrain['Id_index'] = dfTrain['Id']


# In[5]:

dfTrain = dfTrain.set_index('Id_index')


# # Premier aperçu
# 
# On commence par regarder succintement les données. On cherche à répondre à des questions d'ordre général sur les données:
# Nombre de séquences et nombre de relevés, nombre de dimension, il y a-t-il des dimensions catégorielles à traiter et y a-t-il des données manquantes?
# 
# Un premier résumé permet de dire qu'il y a 13765201 entrées de 24 dimensions (23 si on exclu la quantité a prédire), toutes numériques.
# On voit aussi en extrayant la tête et la queue de la table qu'il y a un certain nombre de données manquantes et qu'il y a 1180945 séquences.

# In[118]:

dfTrain.info()


# In[119]:

dfTrain.head(10)


# In[120]:

dfTrain.tail(10)


# ## Distribution des données
# 
# On s'intéresse à la distribution des données, en particulier la quantité de données manquantes et la distribution de la quantité à prédire.
# 
# La ligne count nous permet d'évaluer que pour la plupart des colonnes, environ la moitié des données sont manquantes.
# La distribution de la colonne Expected est particulière dans le sens où les 3 premiers quartiles sont très bas (respectivement 0.25, 1.02 et 3.81) alors que la moyenne est à 108.63 et le maximum à 33017.73.
# C'est-à-dire que la grande majorité des valeurs sont très petites et les grandes valeurs sont extrêmement grandes.
# Ceci explique aussi le choix d'une pénalisation sur la norme L1 plutot que L2 qui risquerait de donner un poids démesuré à ces valeurs extrêmes.

# In[121]:

dfTrain.describe()


# Il semble important de souligner le fait que pour chaque séquence de $n$ relevés radars, la production de ces derniers est espacée d'un temps égal. Le premier est produit à 00:00, le dernier à 59:00. De cette manière nous n'avons pas à créer de features trop complexes utilisant cette dimension "minutes_past" et pouvons nous contenter de mesures statistiques comme la moyenne, la std, les quantiles, le min, le max, etc.

# ### Données manquantes
# 
# Il y aurait 38% de données manquantes
# https://www.kaggle.com/c/how-much-did-it-rain-ii/forums/t/16572/38-missing-data

# Combien y'a-t-il de données manquantes par dimension ?

# In[19]:

l = float(len(dfTrain["Id"]))
comp = []
for i in dfTrain.columns:
    comp.append([1 - dfTrain[i].isnull().sum() / l , i])
comp.sort(key=lambda x: x[0], reverse=True)
print(comp)
compA = np.array(comp)


# Combien y'a-t-il de séquences totalement nulles sauf minutes_past, radardist_km et expected ?
# > environ 40% du trainset ne possède que des features nulles

# In[85]:

dfLight = dfTrain.drop(['minutes_past','radardist_km', 'Expected'],axis=1)
nbIds = len(dfLight.index)
nbIdsAllNan = len(dfLight.dropna(how='all').index)
nbIdsAnyNan = len(dfLight.dropna(how='any').index)
print "nombre de séquences avec NaN:", nbIds
print "nombre de séquences sans all NaN:", nbIdsAllNan, (nbIdsAllNan * 1. / nbIds * 100)
print "nombre de séquences sans any NaN:", nbIdsAnyNan, (nbIdsAnyNan * 1. / nbIds * 100)


# L'administrateur du challenge précise que les séquences pour lesquelles toutes les Ref (colonne \#4) sont manquantes ne seront pas prises en compte, alors nous allons les enlever.
# https://www.kaggle.com/c/how-much-did-it-rain-ii/forums/t/16622/ignored-ids
# 
# Nous travaillons souvent sur un subset pour tester notre code.

# In[59]:

# df = dfTrain[dfTrain.index < 5]


# In[6]:

def myfunc(seq):
    nb = len(seq['Ref'])
    if (seq['Ref'].isnull()).sum() == nb:
        return seq['Id'].values[0]
    return -1
    
def identity(seq):
    print(seq)
    return seq
    
dfGrouped = dfTrain.groupby(dfTrain.index)
dfIds2rmv = dfGrouped.apply(myfunc)


# In[61]:

#dfIds2rmv


# In[8]:

df = dfTrain.drop(dfIds2rmv.values,axis=0)


# Combien a-t-on enlevé de lignes ?

# In[173]:

print len(dfTrain), '*', len(df)*1./len(dfTrain)*100, '% = ', len(df)


# Et maintenant ? Combien y'a-t-il de données manquantes par dimension ?

# In[66]:

l = float(len(df["Id"]))
comp = []
for i in df.columns:
    comp.append([1 - df[i].isnull().sum() / l , i])
comp.sort(key=lambda x: x[0], reverse=True)
comp


# Essayons de trouver des typologies de séquences seulement en les regardants.
# > Ci-dessous on peut voir que certaines colonnes sont totalement vides. Nous pensons alors qu'il va nous falloir développer des algorithmes plus ou moins complexes de remplissage des ces valeurs NaN.

# In[73]:

df.loc[4]


# ### Outliers
# https://www.kaggle.com/sudalairajkumar/how-much-did-it-rain-ii/rainfall-test/log
# Les valeurs de expected au dessus de 1000 mm/h peuvent être des erreurs 
# 
# ou alors
# 
# https://www.kaggle.com/c/how-much-did-it-rain/forums/t/11479/expected/62593
# Some of the extremely high Expected values may be due to the melting of ice precipitation that has collected in the rain gauge, which would release a flood of water in a relatively short period of time. If the rain gauges are heated, then snow filling the gauge and then melting later should not be as much of an issue, but hail or graupel filling the top of the gauge could cause underestimation initially by blocking rain from entering the gauge, then overestimation due to melting and draining into the gauge. The real trick will be to train an algorithm to recognize these instances.

# In[9]:

dfTrainGb = df.groupby(df.index)
exp_rainfall = np.sort(np.array(dfTrainGb['Expected'].aggregate('mean')))
plt.figure()
plt.scatter(np.arange(exp_rainfall.shape[0]), exp_rainfall)
plt.title("Scatterplot for Rainfall distribution in train sample")
plt.ylabel("Rainfall in mm")
plt.show()


# In[22]:

print "nombre d'outliers ? ", exp_rainfall[exp_rainfall >= 1000].shape[0]
print "pourcentage d'outliers ? ", exp_rainfall[exp_rainfall >= 1000].shape[0] * 1. / exp_rainfall.shape[0] * 100


# - Doit-on retirer ces valeurs lors de l'apprentissage de notre modèle ?
# > Intuitivement non, car elles sont fortement prises en compte par notre critère MAE
# 
# - Correspondent-elles à des erreurs de mesures détectables ?
# > Deux grandes approches possibles pour http://eprints.whiterose.ac.uk/767/1/hodgevj4.pdf. La première consiste à utiliser un algorithme d'apprentissage non supervisé comme K-Means et voir si un ou plusieurs clusters regroupent une grande partie des outliers. La seconde consiste à entraîner un classifeur binaire à détecter les outliers. Intuitivement, cette deuxième approche parraît plus efficace si le classifieur généralise assez, car nous n'avons que 4864 exemples d'outliers. Il faudra aussi choisir 4864 exemples représentatifs des non outliers. On notera que d'après http://ijcttjournal.org/Volume3/issue-2/IJCTT-V3I2P118.pdf, il est possible de détecter des outliers de façon non supervisés avec des algorithmes comme One-class SVMs ou One-class Kernel Fisher Discriminants.
# 

# ### Modèles simples
# 
# Nous allons créer des modèles simples (marshall palmer, moyenne Extracted, median Extracted) afin de pouvoir interpréter les résultats de nos futurs modèles.

# In[71]:

size = len(df)
df_train = df[:(size*70/100)]
df_test  = df[(size*70/100):]
print len(df_train), '+', len(df_test), '=', len(df)


# In[75]:

def myfunc(seq):
    rslt = seq.mean()
    return rslt

f_df_test = df_test['Expected']
f_df_test = f_df_test.groupby(f_df_test.index).apply(myfunc)
f_df_test = f_df_test.to_frame('Expected')


# #### Marshall Palmer
# 
# https://en.wikipedia.org/wiki/DBZ_(meteorology)

# In[34]:

from sklearn.metrics import mean_absolute_error

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


# In[76]:

get_ipython().run_cell_magic(u'time', u'', u"# each unique Id is an hour of data at some gauge\ndef myfunc(seq):\n    #rowid = hour['Id'].iloc[0]\n    # sort hour by minutes_past\n    seq = seq.sort('minutes_past', ascending=True)\n    mp = marshall_palmer(seq['Ref'], seq['minutes_past'])\n    #print(type(seq.mean()))\n    rslt = seq.mean()\n    rslt['Marshall_Palmer'] = mp\n    return rslt\n\nf_df_mp = df[['minutes_past', 'Ref', 'Expected']]\nf_df_mp = f_df_mp.groupby(f_df_mp.index).apply(myfunc)")


# In[77]:

mean_absolute_error(f_df_mp['Marshall_Palmer'], f_df_mp['Expected'])


# #### Mean

# In[80]:

labels = np.zeros(len(f_df_test['Expected']))
labels.fill(f_df_mp['Expected'].mean())
mean_absolute_error(labels, f_df_test['Expected'])


# #### Median

# In[81]:

labels = np.zeros(len(f_df_test['Expected']))
labels.fill(f_df_mp['Expected'].median())
mean_absolute_error(labels, f_df_test['Expected'])


# ### Modèles bourins

# #### Few features

# https://www.kaggle.com/c/how-much-did-it-rain/forums/t/14242/congratulations
# 
# Extractions de features
# - Marshall Palmer
# - interpolation avant aggrégation sur la séquence
# - complétion des valeurs NaN : mean(, median, 0)

# In[165]:

size = len(df)
pc = .10 # 0.5% -> 2h pour tout process
dfTry = df[:int(size * pc)]
len(dfTry)


# In[116]:

f_cols = ['radardist_km', 'Ref', 'Ref_5x5_10th', 'Ref_5x5_50th',
       'Ref_5x5_90th', 'RefComposite', 'RefComposite_5x5_10th',
       'RefComposite_5x5_50th', 'RefComposite_5x5_90th', 'RhoHV',
       'RhoHV_5x5_10th', 'RhoHV_5x5_50th', 'RhoHV_5x5_90th', 'Zdr',
       'Zdr_5x5_10th', 'Zdr_5x5_50th', 'Zdr_5x5_90th', 'Kdp',
       'Kdp_5x5_10th', 'Kdp_5x5_50th', 'Kdp_5x5_90th', 'Marshall_Palmer']


# In[117]:

f_interpolable = ['Ref', 'Ref_5x5_10th', 'Ref_5x5_50th',
       'Ref_5x5_90th', 'RefComposite', 'RefComposite_5x5_10th',
       'RefComposite_5x5_50th', 'RefComposite_5x5_90th', 'RhoHV',
       'RhoHV_5x5_10th', 'RhoHV_5x5_50th', 'RhoHV_5x5_90th', 'Zdr',
       'Zdr_5x5_10th', 'Zdr_5x5_50th', 'Zdr_5x5_90th', 'Kdp',
       'Kdp_5x5_10th', 'Kdp_5x5_50th', 'Kdp_5x5_90th']


# In[154]:

get_ipython().run_cell_magic(u'time', u'', u"def myfunc(seq):\n    seq = seq.sort('minutes_past', ascending=True)\n    mp = marshall_palmer(seq['Ref'], seq['minutes_past'])\n    for f_name in f_interpolable: \n        seq[f_name] = seq[f_name].interpolate()\n    rslt = seq.mean()\n    rslt['Marshall_Palmer'] = mp\n    return rslt\n\nf_df = dfTry\nf_df = f_df.groupby(f_df.index).apply(myfunc)")


# Combien y'a-t-il encore de valeurs NaN pour chaque dimension ?

# In[133]:

f_df.describe()


# Voyons si certaines features sont corrélées avec la dimension expected ?
# > étonnemment toutes les features sont décorrélées deux à deux avec expected, même Marshall_Palmer (-0.1122)

# In[94]:

f_df.corr()


# Avant de finaliser notre modèle, nous devons trouver un moyen de remplir les valeurs nulles, nous choisissons la moyenne sur tous les exemples.

# In[155]:

y = f_df.Expected
f_df = f_df[f_cols].apply(lambda x: x.fillna(x.mean()), axis=0)


# In[156]:

size = len(f_df)
pc = .80
X_train = f_df[:int(size* pc)]
y_train = y[:int(size* pc)]
X_test  = f_df[int(size* pc):]
y_test  = y[int(size* pc):]
print len(X_train), '+', len(X_test), '=', len(f_df)


# ##### Machines d'apprentissage linéaires

# In[157]:

get_ipython().run_cell_magic(u'time', u'', u'from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(mean_absolute_error(y_pred, y_test)) # linear regression 41.82')


# In[163]:

get_ipython().run_cell_magic(u'time', u'', u'from sklearn.linear_model import Ridge\nmodel = Ridge()\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(mean_absolute_error(y_pred, y_test))')


# In[167]:

get_ipython().run_cell_magic(u'time', u'', u"from sklearn import svm\nmodel = svm.SVR(kernel='linear')\nmodel.fit(X_train, y_train) # 40 min pour 100% du trainset\ny_pred = model.predict(X_test)\nprint(mean_absolute_error(y_pred, y_test))")


# ##### Machines d'apprentissage non-linéaires (ou noyau rbf)

# In[158]:

get_ipython().run_cell_magic(u'time', u'', u"from sklearn import svm\nmodel = svm.SVR(kernel='rbf')\nmodel.fit(X_train, y_train) # 2h pour 100% du trainset\ny_pred = model.predict(X_test)\nprint(mean_absolute_error(y_pred, y_test)) # SVR 21.55")


# ##### Ensembles de machines d'apprentissages

# In[162]:

get_ipython().run_cell_magic(u'time', u'', u'from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor(n_estimators=50)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(mean_absolute_error(y_pred, y_test))')


# In[169]:

get_ipython().run_cell_magic(u'time', u'', u'from sklearn.ensemble import ExtraTreesRegressor\nmodel = ExtraTreesRegressor(n_estimators=800)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(mean_absolute_error(y_pred, y_test))')


# In[171]:

get_ipython().run_cell_magic(u'time', u'', u'from sklearn.ensemble import GradientBoostingRegressor\nmodel = GradientBoostingRegressor()\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(mean_absolute_error(y_pred, y_test))')


# #### Full Features

# https://www.kaggle.com/c/how-much-did-it-rain/forums/t/14242/congratulations
# 
# Extractions de features
# - Marshall Palmer
# - interpolation
# - dans la séquence
#   - longueur 
#   - si toutes les valeurs sont NaN
#   - pourcentage de valeurs NaN
#   - moyenne
#   - std
#   - min
#   - max
#   - quantiles
# - complétion des valeurs NaN : median(, mean, 0)

# In[188]:

size = len(df)
pc = .10 # 0.5% -> 2h pour tout process
dfTry = df[:int(size * pc)]
len(dfTry)


# In[ ]:

f_cols = ['radardist_km', 'Ref', 'Ref_5x5_10th', 'Ref_5x5_50th',
       'Ref_5x5_90th', 'RefComposite', 'RefComposite_5x5_10th',
       'RefComposite_5x5_50th', 'RefComposite_5x5_90th', 'RhoHV',
       'RhoHV_5x5_10th', 'RhoHV_5x5_50th', 'RhoHV_5x5_90th', 'Zdr',
       'Zdr_5x5_10th', 'Zdr_5x5_50th', 'Zdr_5x5_90th', 'Kdp',
       'Kdp_5x5_10th', 'Kdp_5x5_50th', 'Kdp_5x5_90th', 'Marshall_Palmer']


# In[ ]:

f_interpolable = ['Ref', 'Ref_5x5_10th', 'Ref_5x5_50th',
       'Ref_5x5_90th', 'RefComposite', 'RefComposite_5x5_10th',
       'RefComposite_5x5_50th', 'RefComposite_5x5_90th', 'RhoHV',
       'RhoHV_5x5_10th', 'RhoHV_5x5_50th', 'RhoHV_5x5_90th', 'Zdr',
       'Zdr_5x5_10th', 'Zdr_5x5_50th', 'Zdr_5x5_90th', 'Kdp',
       'Kdp_5x5_10th', 'Kdp_5x5_50th', 'Kdp_5x5_90th']


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u"def myfunc(seq):\n    seq = seq.sort('minutes_past', ascending=True)\n    mp = marshall_palmer(seq['Ref'], seq['minutes_past'])\n    for f_name in f_interpolable: \n        seq[f_name] = seq[f_name].interpolate()\n    rslt = seq.mean()\n    rslt['Marshall_Palmer'] = mp\n    rslt['length'] = len(seq)\n    for f_name in f_interpolable:\n        rslt[f_name + '_isNaN']    = (lambda seqi,length: seqi.isnull().sum() == length)(seq[f_name],rslt['length'])\n        rslt[f_name + '_pcNaN']    = (lambda seqi,length: seqi.isnull().sum()*1./length)(seq[f_name],rslt['length'])\n        # doublons : rslt[f_name + '_mean']     = (lambda seqi: seqi.mean())(seq[f_name])\n        rslt[f_name + '_std']      = (lambda seqi: seqi.std())(seq[f_name])\n        rslt[f_name + '_min']      = (lambda seqi: seqi.min())(seq[f_name])\n        rslt[f_name + '_max']      = (lambda seqi: seqi.max())(seq[f_name])\n        for pc in [.1,.3,.5,.7,.9]:\n            rslt[f_name + '_qtil_' + str(pc)] = (lambda seqi: seqi.quantile(pc))(seq[f_name])\n    return rslt\n\nf_df = dfTry\nf_df = f_df.groupby(f_df.index).apply(myfunc)")


# Est-ce que c'est pas mieux d'utiliser des fonctions nommées afin de ne peut avoir à recharger les fonctions anonymes (lambda) en mémoire ?
# > Non, car le même temps dans les deux cas

# In[183]:

get_ipython().run_cell_magic(u'time', u'', u"\ndef fe_isNaN(seqi,length):\n    return seqi.isnull().sum() == length\ndef fe_pcNaN(seqi,length):\n    return seqi.isnull().sum()*1./length\ndef fe_std(seqi):\n    return seqi.std()\ndef fe_mean(seqi):\n    return seqi.mean()\ndef fe_max(seqi):\n    return seqi.max()\ndef fe_min(seqi):\n    return seqi.min()\ndef fe_qtil(seqi,pc):\n    return seqi.quantile(pc)\n\ndef myfunc(seq):\n    seq = seq.sort('minutes_past', ascending=True)\n    mp = marshall_palmer(seq['Ref'], seq['minutes_past'])\n    for f_name in f_interpolable: \n        seq[f_name] = seq[f_name].interpolate()\n    rslt = seq.mean()\n    rslt['Marshall_Palmer'] = mp\n    rslt['length'] = len(seq)\n    for f_name in f_interpolable:\n        rslt[f_name + '_isNaN']    = fe_isNaN(seq[f_name],rslt['length'])\n        rslt[f_name + '_pcNaN']    = fe_pcNaN(seq[f_name],rslt['length'])\n        # doublons : rslt[f_name + '_mean']     = (lambda seqi: seqi.mean())(seq[f_name])\n        rslt[f_name + '_std']      = fe_std(seq[f_name])\n        rslt[f_name + '_min']      = fe_max(seq[f_name])\n        rslt[f_name + '_max']      = fe_min(seq[f_name])\n        for pc in [.1,.3,.5,.7,.9]:\n            rslt[f_name + '_qtil_' + str(pc)] = fe_qtil(seq[f_name],pc)\n    return rslt\n\nf_df = dfTry\nf_df = f_df.groupby(f_df.index).apply(myfunc)")


# In[ ]:

y = f_df.Expected
f_df = f_df[f_cols].apply(lambda x: x.fillna(x.median()), axis=0)


# In[ ]:

size = len(f_df)
pc = .80
X_train = f_df[:int(size* pc)]
y_train = y[:int(size* pc)]
X_test  = f_df[int(size* pc):]
y_test  = y[int(size* pc):]
print len(X_train), '+', len(X_test), '=', len(f_df)


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'from sklearn.linear_model import Ridge\nmodel = Ridge()\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(mean_absolute_error(y_pred, y_test))')


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u"from sklearn import svm\nmodel = svm.SVR(kernel='rbf')\nmodel.fit(X_train, y_train) # 2h pour 100% du trainset\ny_pred = model.predict(X_test)\nprint(mean_absolute_error(y_pred, y_test)) # SVR 21.55")


# In[ ]:



