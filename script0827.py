####### Features: phone_brand, device_model, app, label
####### Model: logistic (optimized parameter), XGboost (optimized parameter), Keras (parameter borrowed from forum)
####### Ensemble: simple blending -- take arithmetic average of predictions from the above three models

#%%
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
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

# Phone brand
brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
grtrain['brand'] = phone['brand']
grtest['brand'] = phone['brand']
X_trbrand = csr_matrix((np.ones(grtrain.shape[0]),(grtrain.trainrow, grtrain.brand)))
X_tebrand = csr_matrix((np.ones(grtest.shape[0]), (grtest.testrow, grtest.brand)))

print X_trbrand.shape
print X_tebrand.shape


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


# combine features
Xtr = hstack((X_trbrand, X_trmodel, X_trapps, X_trlabels), format='csr')
Xte = hstack((X_tebrand, X_temodel, X_teapps, X_telabels), format='csr')

print Xtr.shape
print Xte.shape


# target
targetencoder = LabelEncoder().fit(grtrain.group)
y = targetencoder.transform(grtrain.group)

#%%
# Pred 1: from logistic, without time info, cv mlogloss ~ 2.28939
clf = LogisticRegression(multi_class='multinomial',solver='lbfgs',C=0.02)
clf.fit(Xtr, y)
yprob = clf.predict_proba(Xte)

predict_logistic = yprob
np.savetxt('predict_logistic.csv', predict_logistic,  delimiter=',')

#%%
# Pred 2: from XGboost, without Time info, cv ~ 2.27171
param = {'objective': 'multi:softprob', 'num_class': 12, 'eval_metric': 'mlogloss', 'silent': 1,
         'eta' : 0.1, 'max_depth' : 3, 'subsample' : 0.8, 'colsample_bytree' : 0.8}
num_round = 800

xg_train = xgb.DMatrix( Xtr, label=y)
bst = xgb.train(param, xg_train, num_round)

xg_test = xgb.DMatrix( Xte)
yprob = bst.predict( xg_test ).reshape( Xte.shape[0], 12)

predict_xgboost = yprob
np.savetxt('predict_xgboost.csv', predict_xgboost,  delimiter=',')

#%%
# Model 3: Neural network
# https://www.kaggle.com/chechir/talkingdata-mobile-user-demographics/keras-on-labels-and-brands/code
nclasses = len(targetencoder.classes_)
dummy_y = np_utils.to_categorical(y) # Convert class vector (integers from 0 to nb_classes to binary class matrix, for use with categorical_crossentropy.


# np.random.shuffle: Modify a sequence in-place by shuffling its contents.
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


# define baseline model
def baseline_model():
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

model=baseline_model()

X_train, X_val, y_train, y_val = train_test_split(Xtr, dummy_y, test_size=0.002, random_state=42)

fit= model.fit_generator(generator=batch_generator(X_train, y_train, 32, True),
                         nb_epoch=15,
                         samples_per_epoch=70496,
                         validation_data=(X_val.todense(), y_val), verbose=2
                         )

# evaluate the model
scores_val = model.predict_generator(generator=batch_generatorp(X_val, 32, False), val_samples=X_val.shape[0])
scores = model.predict_generator(generator=batch_generatorp(Xte, 32, False), val_samples=Xte.shape[0])

print('logloss val {}'.format(log_loss(y_val, scores_val)))
# Logloss val 2.26221564631

scores2 = np.empty(scores.shape)
# Scaling to 1-0 probs
for i in xrange(Xte.shape[0]):
   scores2[i,]=scores[i,]/sum(scores[i,])


predict_keras = scores2
np.savetxt('predict_keras.csv', predict_keras,  delimiter=',')


#%%
# Model 4: random forest
param_grid = {'n_estimators':[1000], 'n_jobs':[-1], 
              'max_depth':range(30, 71, 20)}
n_cv_folds = 5

clf = GridSearchCV(RandomForestClassifier(),param_grid,scoring = "log_loss", cv = n_cv_folds,verbose=2)

clf.fit(Xtr, y)

print clf.grid_scores_

# n_estimators 2000, max_depth: 70

#%% 
# Model 4: random forest

from numpy import genfromtxt, savetxt


rf = RandomForestClassifier(n_estimators=4000, max_depth=70, n_jobs=-1)
rf.fit(Xtr, y)
savetxt('predict_random_forest.csv', rf.predict(Xte), delimiter=',', fmt='%f')


#%%
predict_rf = rf.predict(Xte)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, target)


#%%
# Model 5: 





#%%
# Simple blending
predict_blending = (predict_xgboost + predict_logistic + predict_keras)/3


submit = pd.DataFrame(predict_blending)
submit.columns = ['F23-','F24-26','F27-28','F29-32','F33-42','F43+','M22-','M23-26','M27-28','M29-31','M32-38','M39+']
submit['device_id'] = grtest.index

# reorder columns
cols = submit.columns.tolist()
cols = cols[-1:] + cols[:-1]
submit = submit[cols]

submit.to_csv('submission.csv',index=False)