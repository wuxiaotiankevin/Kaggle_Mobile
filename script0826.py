
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

import xgboost as xgb

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


# Model 1: Logistic
# tuning parameter optimization with cross-validation
cs = np.logspace(-3, -1, 3)
c_grid = [{'C': cs}]
n_cv_folds = 3

clf = GridSearchCV(LogisticRegression(multi_class='multinomial',solver='lbfgs'),\
                   c_grid,scoring = "log_loss", cv = n_cv_folds,verbose=2)

clf.fit(Xtr, y)

clf.grid_scores_

# Out[18]:
# [mean: -2.34180, std: 0.01992, params: {'C': 0.001},
# mean: -2.29083, std: 0.03385, params: {'C': 0.01},
# mean: -2.33015, std: 0.03266, params: {'C': 0.10000000000000001}]


# Model 1a: Logistic -- refined parameters
# tuning parameter optimization with cross-validation
cs = np.array([0.002,0.005,0.01,0.02,0.05])
c_grid = [{'C': cs}]
n_cv_folds = 3

clf = GridSearchCV(LogisticRegression(multi_class='multinomial',solver='lbfgs'),\
                   c_grid,scoring = "log_loss", cv = n_cv_folds,verbose=2)

clf.fit(Xtr, y)

clf.grid_scores_

# Out[19]:
# [mean: -2.32104, std: 0.02522, params: {'C': 0.002},
# mean: -2.29977, std: 0.03096, params: {'C': 0.0050000000000000001},
# mean: -2.29083, std: 0.03385, params: {'C': 0.01},
# mean: -2.28939, std: 0.03529, params: {'C': 0.02},
# mean: -2.30354, std: 0.03509, params: {'C': 0.050000000000000003}]


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


# add Time to features
Xtr_wtime = hstack((X_trbrand, X_trmodel, X_trapps, X_trlabels, X_trhour, X_trday), format='csr')
Xte_wtime = hstack((X_tebrand, X_temodel, X_teapps, X_telabels, X_tehour, X_teday), format='csr')


# Model 1b: Logistic -- with time info
# tuning parameter optimization with cross-validation
cs = np.logspace(-3, -1, 3)
c_grid = [{'C': cs}]
n_cv_folds = 3

clf = GridSearchCV(LogisticRegression(multi_class='multinomial',solver='lbfgs'),\
                   c_grid,scoring = "log_loss", cv = n_cv_folds,verbose=2)

clf.fit(Xtr_wtime, y)

clf.grid_scores_

# Out[25]:
# [mean: -2.34725, std: 0.01934, params: {'C': 0.001},
# mean: -2.32053, std: 0.03114, params: {'C': 0.01},
# mean: -2.32222, std: 0.02874, params: {'C': 0.10000000000000001}]

# not very good to include time info for logistic


# Pred 1: from logistic, without time info, cv mlogloss ~ 2.28939
clf = LogisticRegression(multi_class='multinomial',solver='lbfgs',C=0.02)
clf.fit(Xtr, y)
yprob = clf.predict_proba(Xte)

predict_logistic = yprob


# Model 2: Xgboost
# fixed parameters by setup
param = {'objective': 'multi:softprob', 'num_class': 12, 'eval_metric': 'mlogloss', 'silent': 1}

# parameters that might need tuning
param['eta'] = 0.1; param['max_depth'] = 5;
param['subsample'] = 0.8; param['colsample_bytree'] = 0.8;

msk = np.random.rand(Xtr.shape[0]) < 0.8
xg_cv_train = xgb.DMatrix( Xtr[msk], label=y[msk])
xg_cv_val = xgb.DMatrix( Xtr[~msk], label=y[~msk])

num_round = 500

watchlist = [ (xg_cv_train,'train'), (xg_cv_val, 'test') ]
bst = xgb.train(param, xg_cv_train, num_round, watchlist)

# [499]   train-mlogloss:1.97559  test-mlogloss:2.28


# Model 2-a: Xgboost -- parameter tuning
param_grid = {'max_depth':[3,5,7],'n_estimators':[300,500,700]}

bst_gsearch = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate = 0.1,
                                                         objective= 'multi:softprob',
                                                         subsample=0.8, colsample_bytree=0.8), 
                                   param_grid = param_grid, scoring='log_loss',cv = 2,verbose=2)
              
bst_gsearch.fit(Xtr,y)     

bst_gsearch.grid_scores_
        
# Out[15]:
# [mean: -2.31149, std: 0.02336, params: {'n_estimators': 300, 'max_depth': 3},
#  mean: -2.31009, std: 0.02408, params: {'n_estimators': 500, 'max_depth': 3},
#  mean: -2.31275, std: 0.02447, params: {'n_estimators': 700, 'max_depth': 3},
#  mean: -2.31350, std: 0.02347, params: {'n_estimators': 300, 'max_depth': 5},
#  mean: -2.32159, std: 0.02336, params: {'n_estimators': 500, 'max_depth': 5},
#  mean: -2.33223, std: 0.02327, params: {'n_estimators': 700, 'max_depth': 5},
#  mean: -2.32149, std: 0.02228, params: {'n_estimators': 300, 'max_depth': 7},
#  mean: -2.33633, std: 0.02240, params: {'n_estimators': 500, 'max_depth': 7},
#  mean: -2.35308, std: 0.02185, params: {'n_estimators': 700, 'max_depth': 7}]

bst_gsearch.best_params_
# Out[16]: {'max_depth': 3, 'n_estimators': 500}


# Model 2-b: Xgboost -- verify the chosen parameter
# fixed parameters by setup
param = {'objective': 'multi:softprob', 'num_class': 12, 'eval_metric': 'mlogloss', 'silent': 1}

# parameters that might need tuning
param['eta'] = 0.1; param['max_depth'] = 3;
param['subsample'] = 0.8; param['colsample_bytree'] = 0.8;

msk = np.random.rand(Xtr.shape[0]) < 0.8
xg_cv_train = xgb.DMatrix( Xtr[msk], label=y[msk])
xg_cv_val = xgb.DMatrix( Xtr[~msk], label=y[~msk])

num_round = 1000

watchlist = [ (xg_cv_train,'train'), (xg_cv_val, 'test') ]
bst = xgb.train(param, xg_cv_train, num_round, watchlist)
# [499]	train-mlogloss:2.12565	test-mlogloss:2.27499
# [800]	train-mlogloss:2.06813	test-mlogloss:2.27171


# Pred 2: from XGboost, without Time info, cv ~ 2.27171
param = {'objective': 'multi:softprob', 'num_class': 12, 'eval_metric': 'mlogloss', 'silent': 1,
         'eta' : 0.1, 'max_depth' : 3, 'subsample' : 0.8, 'colsample_bytree' : 0.8}
num_round = 800

xg_train = xgb.DMatrix( Xtr, label=y)
bst = xgb.train(param, xg_train, num_round)

xg_test = xgb.DMatrix( Xte)
yprob = bst.predict( xg_test ).reshape( Xte.shape[0], 12)

predict_xgboost = yprob


# Try simplest blending: take average of two predictions
predict_blending = (predict_xgboost + predict_logistic)/2

submit = pd.DataFrame(predict_blending)
submit.columns = ['F23-','F24-26','F27-28','F29-32','F33-42','F43+','M22-','M23-26','M27-28','M29-31','M32-38','M39+']
submit['device_id'] = grtest.index

# reorder columns
cols = submit.columns.tolist()
cols = cols[-1:] + cols[:-1]
submit = submit[cols]

submit.to_csv('submission.csv',index=False)


# Model 3: Random Forest

param_grid = {'n_estimators':[2000],
              'max_depth':[50]}
n_cv_folds = 2

clf = GridSearchCV(RandomForestClassifier(),param_grid,scoring = "log_loss", cv = n_cv_folds,verbose=2)

clf.fit(Xtr, y)

clf.grid_scores_

# Out[37]:
# [mean: -2.40121, std: 0.00534, params: {'n_estimators': 100, 'max_depth': 5},
#  mean: -2.40074, std: 0.00513, params: {'n_estimators': 500, 'max_depth': 5},
#  mean: -2.39325, std: 0.00610, params: {'n_estimators': 100, 'max_depth': 7},
#  mean: -2.39289, std: 0.00705, params: {'n_estimators': 500, 'max_depth': 7}]
# Out[38]:
# [mean: -2.36197, std: 0.01374, params: {'n_estimators': 1000, 'max_depth': 20},
#  mean: -2.36189, std: 0.01354, params: {'n_estimators': 2000, 'max_depth': 20}]
# Out[41]: [mean: -2.40177, std: 0.00473, params: {'max_features': 'log2', 'n_estimators': 2000, 'max_depth': 20}]
# Out[43]: [mean: -2.35028, std: 0.01594, params: {'n_estimators': 2000, 'max_depth': 30}]
