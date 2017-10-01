####### Features: phone_brand, device_model, app, label
####### Model: logistic (optimized parameter), XGboost (optimized parameter), Keras (parameter borrowed from forum)
####### Ensemble: simple blending -- take arithmetic average of predictions from the above three models



import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load data
datadir = '../Data/'
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



# Time info
events['hour'] = events.timestamp.dt.hour
events['day'] = events.timestamp.dt.dayofweek
device_evnt_hour = events.groupby(['device_id','hour'])['hour'].agg(['size'])\
                    .merge(grtrain[['trainrow']], how='left', left_index=True, right_index=True)\
                    .merge(grtest[['testrow']], how='left', left_index=True, right_index=True)\
                    .reset_index()
device_evnt_hour.head()

df = device_evnt_hour.dropna(subset=['trainrow'])
X_trhour = csr_matrix((df['size'], (df.trainrow, df.hour)),shape=(grtrain.shape[0],24))
df = device_evnt_hour.dropna(subset=['testrow'])
X_tehour = csr_matrix((df['size'], (df.testrow, df.hour)),shape=(grtest.shape[0],24))

print X_trhour.shape
print X_tehour.shape

device_evnt_day = events.groupby(['device_id','day'])['day'].agg(['size'])\
                    .merge(grtrain[['trainrow']], how='left', left_index=True, right_index=True)\
                    .merge(grtest[['testrow']], how='left', left_index=True, right_index=True)\
                    .reset_index()
device_evnt_hour.head()

df = device_evnt_day.dropna(subset=['trainrow'])
X_trday = csr_matrix((df['size'], (df.trainrow, df.day)),shape=(grtrain.shape[0],7))
df = device_evnt_day.dropna(subset=['testrow'])
X_teday = csr_matrix((df['size'], (df.testrow, df.day)),shape=(grtest.shape[0],7))

print X_trday.shape
print X_teday.shape



# combine features
Xtr = hstack((X_trbrand, X_trmodel, X_trapps, X_trlabels, X_trhour, X_trday), format='csr')
Xte = hstack((X_tebrand, X_temodel, X_teapps, X_telabels, X_tehour, X_teday), format='csr')

print Xtr.shape
print Xte.shape



# target
targetencoder = LabelEncoder().fit(grtrain.group)
y = targetencoder.transform(grtrain.group)



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
print('logloss val {}'.format(log_loss(y_val, scores_val)))
# logloss val 2.28906690657

### it's really interesting/sad that Time info seems to lower prediction power in every model we tried (logistic, Xgboost, Keras)


# combine features - drop time info
Xtr = hstack((X_trbrand, X_trmodel, X_trapps, X_trlabels), format='csr')
Xte = hstack((X_tebrand, X_temodel, X_teapps, X_telabels), format='csr')

print Xtr.shape
print Xte.shape



# Model 4: Random Forest
param_grid = {'n_estimators':[2000],
              'max_depth':[50]}
n_cv_folds = 2

clf = GridSearchCV(RandomForestClassifier(),param_grid,scoring = "log_loss", cv = n_cv_folds,verbose=2)

clf.fit(Xtr, y)

clf.grid_scores_
# [mean: -2.34449, std: 0.01663, params: {'n_estimators': 2000, 'max_depth': 50}]