####### Features: phone_brand, device_model, app, label
####### Model: logistic (optimized parameter), XGboost (optimized parameter), Keras (parameter borrowed from forum)
####### Ensemble: stacking
#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils

from sklearn.cross_validation import train_test_split,StratifiedKFold
from sklearn.metrics import log_loss


import os
os.chdir('C:/Brown/2016Summer/Kaggle/TalkingData Mobile User Demographics')
#%%
# Load data
datadir = 'C:/Brown/2016Summer/Kaggle/TalkingData Mobile User Demographics/Data/'
grtrain = pd.read_csv(datadir+'gender_age_train.csv',index_col='device_id')
grtest = pd.read_csv(datadir+'gender_age_test.csv',index_col='device_id')
phone = pd.read_csv(datadir+'phone_brand_device_model.csv')
# remove duplicate device_id
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(datadir+'events.csv',parse_dates=['timestamp'], index_col='event_id')
appevnt = pd.read_csv(datadir+'app_events.csv',dtype={'is_installed':bool,'is_active':bool})
applabl = pd.read_csv(datadir+'app_labels.csv')

grtrain['trainrow'] = np.arange(grtrain.shape[0])
grtest['testrow'] = np.arange(grtest.shape[0])

# https://www.kaggle.com/dvasyukova/talkingdata-mobile-user-demographics/a-linear-model-on-apps-and-labels/discussion

#%%
# Phone brand
brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
grtrain['brand'] = phone['brand']
grtest['brand'] = phone['brand']
X_trbrand = csr_matrix((np.ones(grtrain.shape[0]),(grtrain.trainrow, grtrain.brand)))
X_tebrand = csr_matrix((np.ones(grtest.shape[0]), (grtest.testrow, grtest.brand)))

print X_trbrand.shape
print X_tebrand.shape

#%%
# Device model
brandmodel = phone.phone_brand.str.cat(phone.device_model)
modelencoder = LabelEncoder().fit(brandmodel)
phone['model'] = modelencoder.transform(brandmodel)
grtrain['model'] = phone['model']
grtest['model'] = phone['model']
X_trmodel = csr_matrix((np.ones(grtrain.shape[0]),(grtrain.trainrow, grtrain.model)))
X_temodel = csr_matrix((np.ones(grtest.shape[0]), (grtest.testrow, grtest.model)))

print X_trmodel.shape
print X_temodel.shape

#%%
# App
appencoder = LabelEncoder().fit(appevnt.app_id)
appevnt['app'] = appencoder.transform(appevnt.app_id)
napps = len(appencoder.classes_)
device_apps = appevnt.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)\
            .groupby(['device_id','app'])['app'].agg(['size'])\
            .merge(grtrain[['trainrow']], how='left', left_index=True, right_index=True)\
            .merge(grtest[['testrow']], how='left', left_index=True, right_index=True)\
            .reset_index()

device_apps.head()

df = device_apps.dropna(subset=['trainrow'])
X_trapps = csr_matrix((np.ones(df.shape[0]), (df.trainrow, df.app)),shape=(grtrain.shape[0],napps))

df = device_apps.dropna(subset=['testrow'])
X_teapps = csr_matrix((np.ones(df.shape[0]), (df.testrow, df.app)),shape=(grtest.shape[0],napps))

print X_trapps.shape
print X_teapps.shape

#%%
# Labels
applabl = applabl.loc[applabl.app_id.isin(appevnt.app_id.unique())]
applabl['app'] = appencoder.transform(applabl.app_id)
labelencoder = LabelEncoder().fit(applabl.label_id)
applabl['label'] = labelencoder.transform(applabl.label_id)
nlabels = len(labelencoder.classes_)
device_labels = device_apps[['device_id','app','size']]\
                .merge(applabl[['app','label']])\
                .groupby(['device_id','label'])['size'].agg(['sum'])\
                .merge(grtrain[['trainrow']], how='left', left_index=True, right_index=True)\
                .merge(grtest[['testrow']], how='left', left_index=True, right_index=True)\
                .reset_index()

device_labels.head()

df = device_labels.dropna(subset=['trainrow'])
X_trlabels = csr_matrix((np.ones(df.shape[0]), (df.trainrow, df.label)),shape=(grtrain.shape[0],nlabels))

df = device_labels.dropna(subset=['testrow'])
X_telabels = csr_matrix((np.ones(df.shape[0]), (df.testrow, df.label)),shape=(grtest.shape[0],nlabels))

print X_trlabels.shape
print X_telabels.shape

#%%
# combine features
Xtr = hstack((X_trbrand, X_trmodel, X_trapps, X_trlabels), format='csr')
Xte = hstack((X_tebrand, X_temodel, X_teapps, X_telabels), format='csr')

print Xtr.shape
print Xte.shape

#%%
# target
targetencoder = LabelEncoder().fit(grtrain.group)
y = targetencoder.transform(grtrain.group)

#%%
# prepare base_models
# logistic model
clf = LogisticRegression(multi_class='multinomial',solver='lbfgs',C=0.02)
# xgboost
param = {'objective': 'multi:softprob', 'num_class': 12, 'eval_metric': 'mlogloss', 'silent': 1,
         'eta' : 0.1, 'max_depth' : 3, 'subsample' : 0.8, 'colsample_bytree' : 0.8}
num_round = 800
# keras
nclasses = len(targetencoder.classes_)
dummy_y = np_utils.to_categorical(y)

#%%
def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

def baseline_model(Xtr):
    # create model
    model = Sequential()
    #model.add(Dense(10, input_dim=Xtrain.shape[1], init='normal', activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(50, input_dim=Xtr.shape[1], init='normal', activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(12, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model

#%%
# stacking
n_base_models = 3 # base models: logistic model, xgboost, neural network
n_fold = 3 # for each model, 2-fold

skf = list(StratifiedKFold(y, n_folds=n_fold))

S_train = np.zeros((Xtr.shape[0], nclasses*n_base_models)) # each model will have nclasses probability
S_test = np.zeros((Xte.shape[0], nclasses*n_base_models))

# train base models
for i in range(n_base_models):
    col_index = np.array(range(i*nclasses, (i+1)*nclasses))

    S_test_i = np.zeros((Xte.shape[0], nclasses))

    for j, (train_idx, test_idx) in enumerate(skf):
            X_train = Xtr[train_idx]
            y_train = y[train_idx]
            X_holdout = Xtr[test_idx]
            if i == 0:
                # logistic - train
                clf.fit(X_train, y_train)
                # predict on holdout fold
                S_train[np.ix_(test_idx, col_index)] = clf.predict_proba(X_holdout)[:]
                # predict on test data
                S_test_i[:, :] += clf.predict_proba(Xte)[:]
            elif i == 1:
                # xgboost - train
                xg_train = xgb.DMatrix( X_train, label=y_train)
                bst = xgb.train(param, xg_train, num_round)
                # predict on holdout fold
                xg_holdout = xgb.DMatrix( X_holdout)
                S_train[np.ix_(test_idx, col_index)] = bst.predict( xg_holdout ).reshape( X_holdout.shape[0], nclasses)
                # predict on test data
                xg_test = xgb.DMatrix( Xte)
                S_test_i[:, :] += bst.predict( xg_test ).reshape( Xte.shape[0], nclasses)
            elif i == 2:
                # keras - train
                model=baseline_model(X_train)
                dummy_y_train = dummy_y[train_idx]
                fit= model.fit_generator(generator=batch_generator(X_train, dummy_y_train, 32, True),
                             nb_epoch=15,
                             samples_per_epoch=49760,
                             verbose=2
                             )
                # predict on holdout fold
                S_train[np.ix_(test_idx, col_index)] = model.predict_generator(generator=batch_generatorp(X_holdout, 32, False), val_samples=X_holdout.shape[0])
                # predict on test data
                S_test_i[:, :] += model.predict_generator(generator=batch_generatorp(Xte, 32, False), val_samples=Xte.shape[0])

    S_test[:,col_index] = S_test_i/n_fold

#%%
print S_train.shape
np.savetxt('Stracking_train.csv', S_train,  delimiter=',')

print S_test.shape
np.savetxt('Stracking_test.csv', S_test,  delimiter=',')

#%%
# normalize prediction by keras
for i in range(len(S_train)):
    S_train[i,col_index] = S_train[i,col_index]/sum(S_train[i,col_index])


# train combiner model - with logistic regression
cs = np.logspace(-4, 0, 5)
c_grid = [{'C': cs}]
n_cv_folds = 3

combine_clf = GridSearchCV(LogisticRegression(multi_class='multinomial',solver='lbfgs'),\
                   c_grid,scoring = "log_loss", cv = n_cv_folds,verbose=2)

combine_clf.fit(S_train, y)

combine_clf.grid_scores_

# Out[56]:
# [mean: -2.42385, std: 0.00045, params: {'C': 0.0001},
#  mean: -2.39470, std: 0.00423, params: {'C': 0.001},
#  mean: -2.31017, std: 0.02294, params: {'C': 0.01},
#  mean: -2.27322, std: 0.03800, params: {'C': 0.10000000000000001},
#  mean: -2.27146, std: 0.04133, params: {'C': 1.0}]



# train combiner model - with XGboost
# fixed parameters by setup
param = {'objective': 'multi:softprob', 'num_class': 12, 'eval_metric': 'mlogloss', 'silent': 1}

# parameters that might need tuning
param['eta'] = 0.1; param['max_depth'] = 5;

msk = np.random.rand(S_train.shape[0]) < 0.8
xg_cv_train = xgb.DMatrix( S_train[msk], label=y[msk])
xg_cv_val = xgb.DMatrix( S_train[~msk], label=y[~msk])

num_round = 500

watchlist = [ (xg_cv_train,'train'), (xg_cv_val, 'test') ]
bst = xgb.train(param, xg_cv_train, num_round, watchlist)

# max_depth = 5, [67]	train-mlogloss:2.14001	test-mlogloss:2.26




