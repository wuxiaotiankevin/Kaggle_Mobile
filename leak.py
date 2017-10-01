import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from sklearn.cross_validation import train_test_split

import os
os.chdir('C:/Brown/2016Summer/Kaggle/TalkingData Mobile User Demographics')

#%%
print "#### Feature Extraction Begin ####"
print "### Load Data ###"

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


print "### Extract Phone Brand ###"
# Phone brand
brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
grtrain['brand'] = phone['brand']
grtest['brand'] = phone['brand']
X_trbrand = csr_matrix((np.ones(grtrain.shape[0]),(grtrain.trainrow, grtrain.brand)))
X_tebrand = csr_matrix((np.ones(grtest.shape[0]), (grtest.testrow, grtest.brand)))

print X_trbrand.shape
print X_tebrand.shape


print "### Extract Device Model ###"
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


print "### Extract Apps ###"
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


print "### Extract Labels ###"
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


print "### Possible leak ###"
X_trid = csr_matrix(grtrain.trainrow/grtrain.shape[0]).transpose()
X_teid = csr_matrix(grtest.testrow/grtest.shape[0]).transpose()


print "### Combine Features ###"
# combine features
Xtr = hstack((X_trbrand, X_trmodel, X_trapps, X_trlabels, X_trid), format='csr')
Xte = hstack((X_tebrand, X_temodel, X_teapps, X_telabels, X_teid), format='csr')

print Xtr.shape
print Xte.shape

print "### Label Targets ###"
# target
targetencoder = LabelEncoder().fit(grtrain.group)
y = targetencoder.transform(grtrain.group)

print "#### Feature Extraction End ####"



#%%


#### test new feature with logistic regression
cs = np.logspace(-4, 0, 5)
c_grid = [{'C': cs}]
n_cv_folds = 3

clf = GridSearchCV(LogisticRegression(multi_class='multinomial',solver='lbfgs'),\
                   c_grid,scoring = "log_loss", cv = n_cv_folds,verbose=2)

clf.fit(Xtr, y)

clf.grid_scores_
# Out[36]:
# [mean: -2.40147, std: 0.00501, params: {'C': 0.0001},
#  mean: -2.34032, std: 0.02012, params: {'C': 0.001},
#  mean: -2.30648, std: 0.03998, params: {'C': 0.01},
#  mean: -2.38226, std: 0.06610, params: {'C': 0.10000000000000001},
#  mean: -2.59022, std: 0.04821, params: {'C': 1.0}]

clf.best_params_
# Out[37]: {'C': 0.01}

# more refined
cs = np.array([0.005,0.02,0.05])
c_grid = [{'C': cs}]
n_cv_folds = 3

clf = GridSearchCV(LogisticRegression(multi_class='multinomial',solver='lbfgs'),\
                   c_grid,scoring = "log_loss", cv = n_cv_folds,verbose=2)

clf.fit(Xtr, y)

clf.grid_scores_
# Out[38]: 
# [mean: -2.30580, std: 0.03326, params: {'C': 0.0050000000000000001},
#  mean: -2.31681, std: 0.04740, params: {'C': 0.02},
#  mean: -2.34370, std: 0.05681, params: {'C': 0.050000000000000003}]

clf.best_params_
# Out[39]: {'C': 0.0050000000000000001}

# more refined
cs = np.array([0.002,0.007])
c_grid = [{'C': cs}]
n_cv_folds = 3

clf = GridSearchCV(LogisticRegression(multi_class='multinomial',solver='lbfgs'),\
                   c_grid,scoring = "log_loss", cv = n_cv_folds,verbose=2)

clf.fit(Xtr, y)

clf.grid_scores_
# Out[40]:
# [mean: -2.32072, std: 0.02573, params: {'C': 0.002},
#  mean: -2.30479, std: 0.03634, params: {'C': 0.0070000000000000001}]

clf.best_params_
# Out[41]: {'C': 0.0070000000000000001}

### logistic: best param C = 0.01






#### test new feature with xgboost
param = {'objective': 'multi:softprob', 'num_class': 12, 'eval_metric': 'mlogloss', 'silent': 1}
param['eta'] = 0.1; param['max_depth'] = 3;
param['subsample'] = 0.8; param['colsample_bytree'] = 0.8;
num_round = 1000

msk = np.random.rand(Xtr.shape[0]) < 0.8
xg_cv_train = xgb.DMatrix( Xtr[msk], label=y[msk])
xg_cv_val = xgb.DMatrix( Xtr[~msk], label=y[~msk])

watchlist = [ (xg_cv_train,'train'), (xg_cv_val, 'test') ]
bst = xgb.train(param, xg_cv_train, num_round, watchlist)

# [999]	train-mlogloss:1.98379	test-mlogloss:2.24128

# pred 1 - xgboost
xg_test = xgb.DMatrix( Xte)
pred1 = bst.predict( xg_test ).reshape( Xte.shape[0], 12)






#### test new feature with keras
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

def batch_generatorp(X, batch_size):
    number_of_batches = X.shape[0]/np.ceil(X.shape[0]/batch_size)
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
    model.add(Dense(50, input_dim=Xtr.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model

X_train, X_val, y_train, y_val = train_test_split(Xtr, y, test_size=0.02, random_state=10)
model=baseline_model(Xtr)
fit= model.fit_generator(generator=batch_generator(X_train, y_train, 400, True),
                         nb_epoch=20,
                         samples_per_epoch=(X_train.shape[0]/400)*400,
                         validation_data=(X_val.todense(), y_val), verbose=2
                         )
# Epoch 20/20
# 16s - loss: 2.2195 - acc: 0.2211 - val_loss: 2.2577 - val_acc: 0.1929

scores = model.predict_generator(generator=batch_generatorp(Xte, 400), val_samples=Xte.shape[0])
scores2 = np.empty(scores.shape)
# Scaling to 1-0 probs
for i in xrange(Xte.shape[0]):
   scores2[i,]=scores[i,]/sum(scores[i,])







#### average
pred = (pred1+scores2)/2 # combined it score 2.22635

grtest = pd.read_csv(datadir+'gender_age_test.csv',index_col='device_id')
submit = pd.DataFrame(pred)
submit.columns = ['F23-','F24-26','F27-28','F29-32','F33-42','F43+','M22-','M23-26','M27-28','M29-31','M32-38','M39+']
submit['device_id'] = grtest.index
# reorder columns
cols = submit.columns.tolist()
cols = cols[-1:] + cols[:-1]
submit = submit[cols]
submit.to_csv('submission.csv',index=False)
