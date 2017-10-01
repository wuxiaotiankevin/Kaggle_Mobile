# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 09:20:25 2016

@author: Xiaotian Wu
"""


#%%
# setup
import pandas as pd
import numpy as np

import csv
from sklearn.ensemble import RandomForestClassifier

import os
os.chdir('C:/Brown/2016Summer/Kaggle/TalkingData Mobile User Demographics')

#%%
# Preliminary prediction
# Using only phone brand and model to predict

df1 = pd.read_csv("Data/gender_age_train.csv")
# Drop gender and age
df2 = df1.drop(df1.columns[[1,2]], 1)
# order by device_id for convenience
df2 = df2.sort_values(by='device_id')
# merge device id and model
device = pd.read_csv("Data/phone_brand_device_model.csv")
df3 = pd.merge(df2, device, on='device_id', how='left')
# some duplicated rows may be created in merging
df3[df3.duplicated(subset="device_id",keep=False)]
# remove identical rows
df3 = df3.drop_duplicates() 
# one special case: identical device_id & group, two phones
df3[df3.duplicated(subset="device_id",keep=False)]
# remove the second entry (arbitrary choice)
df3 = df3.drop([58604])
# no absence of phone brand & model info
df3.isnull().sum()
# reset index
df3 = df3.reset_index(drop=True)

#%%

# make contigency table
df3.columns
df3.group.unique()
len(list(df3.phone_brand.unique()))
df3.device_model.unique()

# group vs. phone_brand
# Make table
group_phone_brand = pd.crosstab(index=df3["group"], columns=df3["phone_brand"])
colsum = group_phone_brand.sum(axis=0)
pred_table = group_phone_brand / colsum
# Predict
# 
test = pd.read_csv("Data/gender_age_test.csv")
test_device = pd.merge(test, device, on='device_id', how='left')
test_device = test_device.drop_duplicates()
# Drop arbitraryly chose one duplicate and drop
test_device[test_device['device_id'].isin([-6590454305031525112, 
            -7297178577997113203, -7059081542575379359, -3004353610608679970, 
            -5269721363279128080])]
test_device = test_device.drop([7884, 26451, 52876, 87752, 107243])
# Reset index
test_device = test_device.reset_index(drop=True)
# Merge prediction with pred_table
test_pred = pd.DataFrame()
for i in test_device.index:
    idx = pred_table.columns.values==test_device.loc[i, 'phone_brand']
    # if no matching, use average for each entry
    if idx.sum()==0:
        line = pd.DataFrame(np.append(test_device.loc[i, 'device_id'], np.repeat(1/12, 12))).T
    else:
        line = pd.DataFrame(np.append(test_device.loc[i, 'device_id'], pred_table.loc[:, idx].T.as_matrix())).T
    test_pred = test_pred.append(line, ignore_index=True)                      
    print(i)
submission = pd.read_csv("Data/sample_submission.csv")
test_pred.columns = submission.columns.values

#%%

# Set entries with 0 to 1/12
# As long as there is a 0 entry in the row, change the whole row to 1/12
tmp1 = test_pred.drop('device_id', axis=1)
yn = tmp1==0
yn = yn.sum(axis=1) > 0
tmp1[yn] = 1/12
tmp1[yn]

#%%
# write to csv
sub = pd.read_csv("Data/sample_submission.csv", dtype={'ID': object})
tmp = pd.concat([sub['device_id'], tmp1], axis=1)
tmp.to_csv('sumbission_160811.csv', index = False)








#%%
# 0815
# Read engineered files
most_active_app_category_train_1 = pd.read_csv("Data/Features_train/most_active_app_category_train.csv", header=None)
most_active_app_category_train_2 = pd.read_csv("Data/Features_train/most_active_app_category_train_2.csv", header=None)
most_active_app_category_train = most_active_app_category_train_1.append(most_active_app_category_train_2)
# len(set(most_active_app_category_train[0]))

#%%
# most_active_day_train.csv
# most_active_day_train = pd.read_csv("Data/Features_train/most_active_day_train.csv", header=None, skiprows=1)

# most_active_day_train
# file 1:
with open('Data/Features_train/most_active_day_train.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    device_id = []
    most_active_day = []
    for row in reader:
        # print(row[0])
        device_id.append(row[0])
        most_active_day.append(row[1])
most_active_day_train = pd.DataFrame([device_id, most_active_day]).transpose()
# file 2
with open('Data/Features_train/most_active_day_train_2.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    device_id = []
    most_active_day = []
    for row in reader:
        # print(row[0])
        device_id.append(row[0])
        most_active_day.append(row[1])
most_active_day_train = most_active_day_train.append(pd.DataFrame([device_id, most_active_day]).transpose())


#%%
# most_active_hour_train
# file 1
with open('Data/Features_train/most_active_hour_train.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    device_id = []
    most_active_hour = []
    for row in reader:
        # print(row[0])
        device_id.append(row[0])
        most_active_hour.append(row[1])
most_active_hour_train = pd.DataFrame([device_id, most_active_hour]).transpose()
# file 2
with open('Data/Features_train/most_active_day_train_2.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    device_id = []
    most_active_day = []
    for row in reader:
        # print(row[0])
        device_id.append(row[0])
        most_active_day.append(row[1])
most_active_hour_train = most_active_hour_train.append(pd.DataFrame([device_id, most_active_hour]).transpose())

#%%

# most_installed_app_category_train
most_installed_app_category_train_1= pd.read_csv("Data/Features_train/most_installed_app_category_train.csv", header=None)
most_installed_app_category_train = most_installed_app_category_train_1.append(pd.read_csv("Data/Features_train/most_installed_app_category_train_2.csv", header=None))


#%%
# Merge them all
dat = pd.concat([most_active_app_category_train[0], most_active_app_category_train[1], most_active_day_train[1], most_active_hour_train[1], most_active_app_category_train[1]], axis=1)
dat.columns = ['device_id', 'most_active_app_category_train', 'most_active_day_train', 'most_active_hour_train', 'most_installed_app_category_train']

# Merge with brand and age group
df = pd.merge(dat, df3.ix[:, 0:3], how='left', on=['device_id'])
#%%
cates = ['most_active_app_category_train', 'most_active_day_train', 'most_active_hour_train', 'most_installed_app_category_train', 'group', 'phone_brand']
x = ['most_active_app_category_train', 'most_active_day_train', 'most_active_hour_train', 'most_installed_app_category_train', 'phone_brand']
for i in cates:
    df[i] = df[i].astype('category')

# df['most_active_app_category_train'] = df['most_active_app_category_train'].astype('category')


#%%

rf = RandomForestClassifier(max_depth = 4)
rf.fit(df[x], df['group'])

#%%
# Merge them all
most_active_app_category_train
most_active_day_train
most_active_hour_train
most_installed_app_category_train









