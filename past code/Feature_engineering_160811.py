
# coding: utf-8

# # Y = group
# 
# One device (identified by device_id) has one user, which belongs to one group 
# (gender, age, group). Each device's user group is what we want to predict. 

# In[1]:

import pandas as pd
import numpy as np
import os

os.getcwd()
os.chdir('C:/Brown/2016Summer/Kaggle/TalkingData Mobile User Demographics')

# In[2]:

df1 = pd.read_csv("Data/gender_age_train.csv")
df1.columns.values


# In[3]:

df1.shape


# In[4]:

df1


# In[5]:

# gender and age info are replicating group info, and they are obviously not 
# given in test dataset
# drop these two columns
df2 = df1.drop(df1.columns[[1,2]], 1)
df2.columns.values


# In[6]:

# order by device_id for convenience
df2 = df2.sort_values(by='device_id')


# # X1 = Device
# 
# Each device has one phone brand and one device model. 

# In[7]:

device = pd.read_csv("Data/phone_brand_device_model.csv")
device.columns.values


# In[8]:

df3 = pd.merge(df2, device, on='device_id', how='left')

# some duplicated rows may be created in merging
df3[df3.duplicated(subset="device_id",keep=False)]


# In[9]:

# remove identical rows
df3 = df3.drop_duplicates() 

df3


# In[10]:

# one special case: identical device_id & group, two phones
df3[df3.duplicated(subset="device_id",keep=False)]


# In[11]:

# remove the second entry (arbitrary choice)
df3 = df3.drop([58604])


# In[12]:

# no absence of phone brand & model info
df3.isnull().sum()


# In[13]:

# reset index
df3 = df3.reset_index(drop=True)


# # X2 = App
# 
# Each event represents using multiple apps, 
# some of which are being installed, some of which are being active. 
# Each app can belong to multiple category.
# 
# Will not merge category names into the dataset for now, 
# since the info is already contained in label_id.
# 
# Could do some interesting natural language processing on the actual names 
# in the future. For example, it seems that chunks of label_id's belong to 
# same larger category (e.g., game).

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

# 



#%%

# See what all data sets are doing
submission = pd.read_csv("Data/sample_submission.csv")
test = pd.read_csv("Data/gender_age_test.csv")
test.columns.values

# 

