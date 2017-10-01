# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 21:38:31 2016

@author: yinghonglan
"""

#%%
import pandas as pd
import numpy as np
import xgboost as xgb


import os
os.chdir('C:/Brown/2016Summer/Kaggle/TalkingData Mobile User Demographics')


#%%

################################## SKIP ##################################

#############################################
############ Combine features ###############
#############################################

df = pd.read_csv("Features/group_device_train.csv",sep='\t')

# drop index
df = df.drop(df.columns[[0]], 1)

# load and merge features
names = ['device_id']+['installed_categ'+str(i) for i in range(1,31)]
f1 = pd.read_csv("Features/most_installed_app_category_train.csv",header=None,names=names)
f2 = pd.read_csv("Features/most_installed_app_category_train_2.csv",header=None,names=names)
most_installed_app_category = f1.append(f2)

names = ['device_id']+['active_categ'+str(i) for i in range(1,31)]
f1 = pd.read_csv("Features/most_active_app_category_train.csv",header=None,names=names)
f2 = pd.read_csv("Features/most_active_app_category_train_2.csv",header=None,names=names)
most_active_app_category = f1.append(f2)

names = ['device_id']+['hour'+str(i) for i in range(1,7)]
f1 = pd.read_csv("Features/most_active_hour_train.csv",header=None,names=names)
f2 = pd.read_csv("Features/most_active_hour_train_2.csv",header=None,names=names)
f2.columns = ['device_id']+['hour'+str(i) for i in range(1,7)]
most_active_hour = f1.append(f2)

names = ['device_id']+['day'+str(i) for i in range(1,4)]
f1 = pd.read_csv("Features/most_active_day_train.csv",header=None,names=names)
f2 = pd.read_csv("Features/most_active_day_train_2.csv",header=None,names=names)
most_active_day = f1.append(f2)

names = ['latitude','longitude']
f1 = pd.read_csv("Features/most_active_geo_location_train.csv",header=None,names=names)
f2 = pd.read_csv("Features/most_active_geo_location_train_2.csv",header=None,names=names)
most_active_geo_location = f1.append(f2)
most_active_geo_location['device_id'] = most_active_day.device_id

df = pd.merge(df, most_installed_app_category, on='device_id', how='left')
df = pd.merge(df, most_active_app_category, on='device_id', how='left')
df = pd.merge(df, most_active_hour, on='device_id', how='left')
df = pd.merge(df, most_active_day, on='device_id', how='left')
df = pd.merge(df, most_active_geo_location, on='device_id', how='left')

df.to_csv('Features/combined_features_train.csv')


########### test set ########
df1 = pd.read_csv("Features/group_device_test.csv",sep='\t')

# drop index
df1 = df1.drop(df1.columns[[0]], 1)

# load and merge features
names = ['device_id']+['installed_categ'+str(i) for i in range(1,31)]
f1 = pd.read_csv("Features/most_installed_app_category_test.csv",header=None,names=names)
f2 = pd.read_csv("Features/most_installed_app_category_test_2.csv",header=None,names=names)
f3 = pd.read_csv("Features/most_installed_app_category_test_3.csv",header=None,names=names)
f4 = f1.append(f2)
most_installed_app_category = f4.append(f3)

names = ['device_id']+['active_categ'+str(i) for i in range(1,31)]
f1 = pd.read_csv("Features/most_active_app_category_test.csv",header=None,names=names)
f2 = pd.read_csv("Features/most_active_app_category_test_2.csv",header=None,names=names)
f3 = pd.read_csv("Features/most_active_app_category_test_3.csv",header=None,names=names)
f4 = f1.append(f2)
most_active_app_category = f4.append(f3)

names = ['device_id']+['hour'+str(i) for i in range(1,7)]
f1 = pd.read_csv("Features/most_active_hour_test.csv",header=None,names=names)
f2 = pd.read_csv("Features/most_active_hour_test_2.csv",header=None,names=names)
f3 = pd.read_csv("Features/most_active_hour_test_3.csv",header=None,names=names)
f4 = f1.append(f2)
most_active_hour = f4.append(f3)

names = ['device_id']+['day'+str(i) for i in range(1,4)]
f1 = pd.read_csv("Features/most_active_day_test.csv",header=None,names=names)
f2 = pd.read_csv("Features/most_active_day_test_2.csv",header=None,names=names)
f3 = pd.read_csv("Features/most_active_day_test_3.csv",header=None,names=names)
f4 = f1.append(f2)
most_active_day = f4.append(f3)

names = ['device_id','latitude','longitude']
f1 = pd.read_csv("Features/most_active_geo_location_test.csv",header=None,names=names)
f2 = pd.read_csv("Features/most_active_geo_location_test_2.csv",header=None,names=names)
f3 = pd.read_csv("Features/most_active_geo_location_test_3.csv",header=None,names=names)
f4 = f1.append(f2)
most_active_geo_location= f4.append(f3)

df1 = pd.merge(df1, most_installed_app_category, on='device_id', how='left')
df1 = pd.merge(df1, most_active_app_category, on='device_id', how='left')
df1 = pd.merge(df1, most_active_hour, on='device_id', how='left')
df1 = pd.merge(df1, most_active_day, on='device_id', how='left')
df1 = pd.merge(df1, most_active_geo_location, on='device_id', how='left')

df1.to_csv('Features/combined_features_test.csv')


#%%

#############################################################
######################### Prep data #########################
#############################################################

###### train ######
df = pd.read_csv("Features/combined_features_train.csv")
# drop index
df = df.drop(df.columns[[0]], 1)

df.dtypes
df.shape

###### test ######
df1 = pd.read_csv("Features/combined_features_test.csv")
# drop index
df1 = df1.drop(df1.columns[[0]], 1)

df1.dtypes
df1.shape


################ Categorical -> dummy variables ################
#df2 = df.drop(df.columns[[1]], 1) # remove group to append
#df3 = df2.append(df1) # combine all data for cohesive dummy variable coding
#
#dummy1 = pd.get_dummies(df3.phone_brand)
#dummy2 = pd.get_dummies(df3.device_model)
#
#dummy1_train = dummy1.head(df.shape[0])
#dummy1_test= dummy1.tail(df1.shape[0])
#
#dummy2_train = dummy2.head(df.shape[0])
#dummy2_test= dummy2.tail(df1.shape[0])


################ Categorical variables ################
df2 = df.drop(df.columns[[1]], 1) # remove group to append
df3 = df2.append(df1) # combine all data for cohesive categorical variable coding

cat1 = df3.phone_brand.astype('category').cat.codes.values
cat2 = df3.device_model.astype('category').cat.codes.values


df_cat = np.column_stack((cat1[0:df.shape[0]],cat2[0:df.shape[0]]))
df1_cat = np.column_stack((cat1[df.shape[0]:],cat2[df.shape[0]:]))



#%%
####################################################################
######################### Cross validation #########################
####################################################################
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
val = df[~msk]


################ Cast variables ###################
## XGboost only takes numerical input
train_Y = train.group.astype('category').cat.codes.values
val_Y = val.group.astype('category').cat.codes.values

train_X = train.drop(train.columns[[0,1,2,3]], 1).values
train_X[np.isnan(train_X)]=-999
train_X = np.concatenate((train_X,df_cat[msk]),axis=1)

val_X = val.drop(val.columns[[0,1,2,3]], 1).values
val_X[np.isnan(val_X)]=-999
val_X = np.concatenate((val_X,df_cat[~msk]),axis=1)

xg_train = xgb.DMatrix( train_X, label=train_Y, missing=-999)
xg_val = xgb.DMatrix( val_X, label=val_Y, missing=-999)


#%%
################ Parameters ###################
param = {}

# use softmax multi-class classification
param['objective'] = 'multi:softprob'

# number of classes
param['num_class'] = 12

# evaluation: multiclass log loss
param['eval_metric'] = 'mlogloss'

# silent mode
param['silent'] = 1

############ Parameters that require further tuning #########
# scale weight of positive examples
# step size shrinkage
param['eta'] = 0.1

param['max_depth'] = 6

num_round = 100

# and many others ...
##########################################################

watchlist = [ (xg_train,'train'), (xg_val, 'test') ]
bst = xgb.train(param, xg_train, num_round, watchlist)


#%%
##############################################################
######################### Prediction #########################
##############################################################

################ Retrain with everything ###################

train_Y = df.group.astype('category').cat.codes.values
train_X = df.drop(df.columns[[0,1,2,3]], 1).values
train_X[np.isnan(train_X)]=-999
train_X = np.concatenate((train_X,df_cat),axis=1)
xg_train = xgb.DMatrix( train_X, label=train_Y, missing=-999)


test_X = df1.drop(df1.columns[[0,1,2]], 1).values
test_X[np.isnan(test_X)]=-999
test_X = np.concatenate((test_X,df1_cat),axis=1)
xg_test = xgb.DMatrix( test_X, missing=-999)

param = {}
param['objective'] = 'multi:softprob'
param['num_class'] = 12
param['eval_metric'] = 'mlogloss'
param['silent'] = 1

param['eta'] = 0.1
param['max_depth'] = 6
num_round = 100

bst = xgb.train(param, xg_train, num_round)


################ Predict ###################

# get prediction, this is in 1D array, need reshape to (ndata, nclass)
yprob = bst.predict( xg_test ).reshape( df1.shape[0], 12 )

# write to csv for submission
submit = pd.DataFrame(yprob)
submit.columns = ['F23-','F24-26','F27-28','F29-32','F33-42','F43+','M22-','M23-26','M27-28','M29-31','M32-38','M39+']
submit['device_id'] = df1.device_id
# reorder columns
cols = submit.columns.tolist()
cols = cols[-1:] + cols[:-1]
submit = submit[cols] 

submit.to_csv('submission_160816.csv',index=False)


#%%
# xgboost with cv
#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


target = 'Disbursed'
IDcol = 'ID'


#%%
# Define a function for cross validation
def modelfit(alg, train_x, train_y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train_x, label=train_y, missing=-999)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='mlogloss', early_stopping_rounds=early_stopping_rounds) #, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(train_x, train_y,eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(train_x)
    dtrain_predprob = alg.predict_proba(train_x)[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(train_y.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
#%%
#Choose all predictors except target & IDcols
# predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softprob',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train_X, train_Y)