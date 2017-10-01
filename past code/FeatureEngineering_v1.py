# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:41:30 2016

@author: yinghonglan
"""
#%%
import pandas as pd

#%%
directory = 'Data'

######## load data ########
# group
group_train = pd.read_csv(directory + '/gender_age_train.csv')
# drop gender and age
group_train = group_train.drop(group_train.columns[[1,2]], 1)
# order by device_id for convenience
group_train = group_train.sort('device_id')
group_test = pd.read_csv(directory + '/gender_age_test.csv')
# order by device_id for convenience
group_test = group_test.sort('device_id')

# device
device = pd.read_csv(directory + '/phone_brand_device_model.csv')
# order by device_id for convenience
device = device.sort('device_id')
device = device.drop_duplicates() 

# app
apps = pd.read_csv(directory + '/app_labels.csv')
# order by app_id and label_id for convenience
apps = apps.sort(['app_id','label_id'])

# categories
categories = pd.read_csv(directory + '/label_categories.csv')
# order by label_id for convenience
categories = categories.sort('label_id')

# events
events = pd.read_csv(directory + '/events.csv')
# order by device_id and event_id for convenience
events = events.sort(['device_id','event_id'])

# event apps
events_apps = pd.read_csv(directory + '/app_events.csv')
# order by event_id and app_id for convenience
events_apps = events_apps.sort(['event_id','app_id'])

#%%

######## features ########

###### count of events for each day: vector, length = 7 (days)
###### count of events for each hour: vector, length = 24 (hours)
events['hour'] = pd.to_datetime(events.timestamp).dt.hour
events['day'] = pd.to_datetime(events.timestamp).dt.dayofweek
f1 = events.groupby(['device_id','hour']).size().unstack()
f1.columns = ['hour'+str(i) for i in range(f1.shape[1])]
f1['device_id'] = f1.index

f2 = events.groupby(['device_id','day']).size().unstack()
f2.columns = ['day'+str(i) for i in range(f2.shape[1])]
f2['device_id'] = f2.index

##### count of events for each app: vector, length = # of different apps
tmp1 = events.merge(events_apps,on='event_id',how='left')
f3 = tmp1.groupby(['device_id','app_id']).size().unstack()
f3.columns = ['app'+str(i) for i in range(f3.shape[1])]
f3['device_id'] = f3.index

del tmp1

##### count of events for each category: vector, length = # of different categories of apps
tmp2 = pd.merge(events_apps,apps,on='app_id',how='left')
tmp3 = pd.merge(events,tmp2,on='event_id',how='left')
f4 = tmp3.groupby(['device_id','label_id']).size().unstack()
f4.columns = ['category'+str(i) for i in range(f4.shape[1])]
f4['device_id'] = f4.index

del tmp2
del tmp3

##### average geo location
tmp4 = events[(events.longitude * events.latitude) != 0.00 ]
f5 = tmp4.groupby('device_id').mean()[['longitude','latitude']]
f5['device_id'] = f5.index

#%%
######## put together ########
train = group_train.merge(device,on='device_id',how='left')\
        .merge(f1,on='device_id',how='left')\
        .merge(f2,on='device_id',how='left')\
        .merge(f3,on='device_id',how='left')\
        .merge(f4,on='device_id',how='left')\
        .merge(f5,on='device_id',how='left')
        
test =  group_test.merge(device,on='device_id',how='left')\
        .merge(f1,on='device_id',how='left')\
        .merge(f2,on='device_id',how='left')\
        .merge(f3,on='device_id',how='left')\
        .merge(f4,on='device_id',how='left')\
        .merge(f5,on='device_id',how='left')

train.to_csv('train.csv')
test.to_csv('test.csv')

