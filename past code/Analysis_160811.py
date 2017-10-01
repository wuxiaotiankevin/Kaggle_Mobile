# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 09:20:25 2016

@author: Xiaotian Wu
"""


#%%
# setup
import pandas as pd
import numpy as np
import os

os.getcwd()
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
# Stopped here

# In[14]:

apps = pd.read_csv("Data/app_labels.csv")
apps.columns.values


# In[15]:

# order by app_id and label_id for convenience
apps = apps.sort_values(by=['app_id','label_id'])
apps


# In[16]:

categories = pd.read_csv("Data/label_categories.csv")
categories.columns.values


# In[17]:

# order by label_id for convenience
categories = categories.sort_values(by='label_id')
categories

#%%
# # X3 = Events
# 
# Each device records multiple events (events.csv). 
# Each event has time info (timestamp) and 
# geographical info (longitude, latitude). 
# 
# Event is what connects device with apps (app_events.csv). 
# As described above, each event represents using multiple apps. 
# 
# Because one device has many events & one event has many apps & 
# one app belongs to multiple categories, 
# we cannot simply merge app categories with event data with device data. 
# 
# Instead of directly using raw data, we need to do some
# ## feature engineering
# 
# One device -- multiple events, we can extract the following variable 
# for each device:
# 
# ### How active the device is used during different time periods?
# counts_of_events_during_each_timeperiod = [one count for each hour]  
# Note: these time periods are arbitrarily divided, 
# could be combined or further divided
# 
# ### How frequently the user is moving during different time periods?
# counts_of_different_geographical_location_during_each_timeperiod = 
# [one count for each hour]  
# Note: longitude==0.0 and/or latitude ==0.0 represents no info on 
# geographical location
# 
# one event -- multiple apps, we can extract the following variable for 
# each device:
# 
# ### Which apps are used during different time periods?
# five_most_used_app_category_during_each_timeperiod = 
# [one list for each hour]  
# Note1: five is just an arbitrary choice, could be three, could be ten  
# Note2: each time period would have a list of 5 integers
# 
# 
# Most important note:
# ### What are other useful features to extract from event and app data?
# For example, distances between different geographical locations could be 
# calculated based on longitude and latitude data. 
# Or, should we differentiate between activities on weekdays vs. on weekends?  
# Another example, as mentioned above, some natural language processing on 
# the app category names may provide extra information on what kinds of apps 
# are used.

# In[18]:

events = pd.read_csv("Data/events.csv")
events.columns.values


# In[19]:

# order by device_id and event_id for convenience
events = events.sort_values(by=['device_id','event_id'])
events


# In[20]:

events_apps = pd.read_csv("Data/app_events.csv")
events_apps.columns.values


# In[21]:

# order by event_id and app_id for convenience
events_apps = events_apps.sort_values(by=['event_id','app_id'])
events_apps


# ### Extract feature: counts_of_events_during_each_timeperiod

# In[22]:

# pandas has very cool tools for processing timestamp data

# as an example
# print(events.timestamp[0])

# show all features
# dir(pd.to_datetime(events.timestamp[0]))

pd.to_datetime(events.timestamp[0]).dayofweek

pd.to_datetime(events.timestamp[0]).hour


# In[23]:

# initialize the variable
counts_of_events_during_each_timeperiod = np.zeros([len(df3.index),24])
counts_of_events_during_each_timeperiod


# In[24]:

# as an example
tmp = events.loc[events['device_id'] == -9222956879900151005]
time = pd.to_datetime(tmp.timestamp)
#time.dt.dayofweek.value_counts()
time.dt.hour.value_counts()


# In[35]:

# scan the devices
for i in range(1000):
    d_events = events.loc[events['device_id'] == df3.device_id[i]]
    if len(d_events) > 0:
        d_time = pd.to_datetime(d_events.timestamp)
        for key, value in time.dt.hour.value_counts().iteritems():
            counts_of_events_during_each_timeperiod[i,key] = value
        print( df3.device_id[i])
        print( time.dt.hour.value_counts())


# In[36]:

# scan the devices
for i in range(1000):
    d_events = events.loc[events['device_id'] == df3.device_id[i]]
    if len(d_events) > 0:
        d_time = pd.to_datetime(d_events.timestamp)
        for key, value in time.dt.hour.value_counts().iteritems():
            counts_of_events_during_each_timeperiod[i,key] = value
        print (df3.device_id[i])
        print (time.dt.dayofweek.value_counts())


# # The first feature does not work!!! 
# 
# All devices that have event data share identical timestamp!!!




#%%
# Useful facts

# More than half of the devices have no events.



#%%

# See what all data sets are doing
submission = pd.read_csv("Data/sample_submission.csv")
sub = pd.read_csv("Data/sample_submission.csv", dtype={'ID': object})
# test = pd.read_csv("Data/gender_age_test.csv")
# test.columns.values

# 

