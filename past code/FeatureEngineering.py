# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:10:59 2016

@author: yinghonglan
"""
#%%
import pandas as pd
import numpy as np
import operator

import os
os.chdir('C:/Brown/2016Summer/Kaggle/TalkingData Mobile User Demographics')

#%%
##################################################################
#################### Prep training data ##########################
##################################################################

df1 = pd.read_csv("Data/gender_age_train.csv")

# drop gender and age
df2 = df1.drop(df1.columns[[1,2]], 1)

# order by device_id for convenience
df2 = df2.sort_values(by='device_id')


#################### Merge with Device ##########################

device = pd.read_csv("Data/phone_brand_device_model.csv")

df3 = pd.merge(df2, device, on='device_id', how='left')

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
##################################################################
###################### Prep App data ###########################
##################################################################

apps = pd.read_csv("Data/app_labels.csv")

# order by app_id and label_id for convenience
apps = apps.sort_values(by=['app_id','label_id'])

categories = pd.read_csv("Data/label_categories.csv")

# order by label_id for convenience
categories = categories.sort_values(by='label_id')



#%%
##################################################################
###################### Prep event data ###########################
##################################################################


events = pd.read_csv("Data/events.csv")

# order by device_id and event_id for convenience
events = events.sort_values(by=['device_id','event_id'])

events_apps = pd.read_csv("Data/app_events.csv")

# order by event_id and app_id for convenience
events_apps = events_apps.sort_values(by=['event_id','app_id'])


#################### Devices with event data ##########################

# all the device_id in the training data set and with event data
device_id_with_event = sorted(list(set(df3.device_id) & set(events.device_id)))

# only 23309 devices in the training data set, or 31.22%, have event data
#len(device_id_with_event)

#len(device_id_with_event)/(len(df3.device_id)*1.0)


# subset event based on device_id_with_event
events_for_train = events.loc[events['device_id'].isin(device_id_with_event)]

# subset events_apps accordingly
events_apps_for_train = events_apps.loc[events_apps['event_id'].isin(events_for_train.event_id)]

# get categories of each app
events_apps_for_train = pd.merge(events_apps_for_train, apps, on='app_id', how='left')

# Notice: one app can belong to multiple categories
# events_apps_for_train[events_apps_for_train.duplicated(subset=["event_id","app_id"],keep=False)]

# for convenience, merge device_id into events_apps_for_train
events_apps_for_train = pd.merge(events_for_train[['event_id','device_id']], events_apps_for_train, on='event_id', how='left')

# seperate is_installed vs. is_active
events_apps_for_train_is_installed = events_apps_for_train.loc[events_apps_for_train['is_installed'] == 1 ]
events_apps_for_train_is_active = events_apps_for_train.loc[events_apps_for_train['is_active'] == 1 ]
# for some cases, both is_installed and is_active equal 1
# all active is installed

#%%
##################################################################
###################### Extract event features ####################
##################################################################

## initialize features

# most installed app category
ten_most_installed_app_category = np.zeros([len(device_id_with_event),10])
# most active app category
ten_most_active_app_category = np.zeros([len(device_id_with_event),10])
# most active hour
five_most_active_hour = np.zeros([len(device_id_with_event),5])
# most active day
two_most_active_day = np.zeros([len(device_id_with_event),2])
# most active geo location
active_geo_location = np.zeros([len(device_id_with_event),2])


#%%
for i in range(len(device_id_with_event)):
    print(i)
    current_id = device_id_with_event[i]
    # most installed app category
    current_events_apps = events_apps_for_train_is_installed.loc[events_apps_for_train_is_installed['device_id'] == current_id]
    current_events_apps_counts = current_events_apps.label_id.value_counts()
    tmp1 = current_events_apps_counts.head(n=10).index.tolist()
    ten_most_installed_app_category[i,:] = tmp1 + [None]*(10-len(tmp1)) #in case there are < 10 categories
    # most active app category
    current_events_apps = events_apps_for_train_is_active.loc[events_apps_for_train_is_installed['device_id'] == current_id]
    current_events_apps_counts = current_events_apps.label_id.value_counts()
    tmp2 = current_events_apps_counts.head(n=10).index.tolist()
    ten_most_active_app_category[i,:] = tmp2 + [None]*(10-len(tmp2))
    # most active hour
    current_events = events_for_train.loc[events_for_train['device_id'] == current_id]
    current_timestamp = pd.to_datetime(current_events.timestamp)
    current_timestamp_counts_hour = current_timestamp.dt.hour.value_counts()
    tmp3 = current_timestamp_counts_hour.head(n=5).index.tolist()
    five_most_active_hour[i,:] = tmp3 + [None]*(5-len(tmp3))
    # most active day
    current_timestamp_counts_day = current_timestamp.dt.dayofweek.value_counts()
    tmp4 = current_timestamp_counts_day.head(n=2).index.tolist()
    two_most_active_day[i,:] = tmp4 + [None]*(2-len(tmp4))
    # most active geo location
    current_geolocation = current_events.loc[current_events['latitude'] != 0]
    if len(current_geolocation)>0:
        tmp5 = list(max(current_geolocation.groupby(['latitude','longitude']).size().to_dict().iteritems(), key=operator.itemgetter(1))[0])      
    else:
        tmp5 = []
    active_geo_location[i,:] = tmp5 + [None]*(2-len(tmp5))
    # dump to csv if necessary
    if (i>0) and (i%50 == 0):
        np.savetxt("ten_most_installed_app_category.csv", ten_most_installed_app_category[0:i+1,:], delimiter=",")
        np.savetxt("ten_most_active_app_category.csv", ten_most_active_app_category[0:i+1,:], delimiter=",")
        np.savetxt("five_most_active_hour.csv", five_most_active_hour[0:i+1,:], delimiter=",")
        np.savetxt("two_most_active_day.csv", two_most_active_day[0:i+1,:], delimiter=",")
        np.savetxt("active_geo_location.csv", active_geo_location[0:i+1,:], delimiter=",")
        
np.savetxt("ten_most_installed_app_category.csv", ten_most_installed_app_category, delimiter=",")
np.savetxt("ten_most_active_app_category.csv", ten_most_active_app_category, delimiter=",")
np.savetxt("five_most_active_hour.csv", five_most_active_hour, delimiter=",")
np.savetxt("two_most_active_day.csv", two_most_active_day, delimiter=",")
np.savetxt("active_geo_location.csv", active_geo_location, delimiter=",")


#%%

# Read engineered files
most_active_app_category_train_1 = pd.read_csv("Data/Features_train/most_active_app_category_train.csv", header=None)
most_active_app_category_train_2 = pd.read_csv("Data/Features_train/most_active_app_category_train_2.csv", header=None)
most_active_app_category_train = most_active_app_category_train_1.append(most_active_app_category_train_2)
# len(set(most_active_app_category_train[0]))

#%%
# most_active_day_train.csv
most_active_day_train = pd.read_csv("Data/Features_train/most_active_day_train.csv", header=None)


#%%
# Event locations
event_locations = events[['latitude', 'longitude']]
event_locations = event_locations.drop_duplicates()
event_locations.to_csv('event_locations.csv', sep=',', index=False, header=False)

#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

themap = Basemap(projection='gall',
              llcrnrlon = -15,              # lower-left corner longitude
              llcrnrlat = 28,               # lower-left corner latitude
              urcrnrlon = 45,               # upper-right corner longitude
              urcrnrlat = 73,               # upper-right corner latitude
              resolution = 'l',
              area_thresh = 100000.0,
              )
themap.drawcoastlines()
themap.drawcountries()
themap.fillcontinents(color = 'gainsboro')
themap.drawmapboundary(fill_color='steelblue')
x, y = themap(event_locations['lonitude'], event_locations['latitude'])
themap.plot(x, y, 
            'o',                    # marker shape
            color='Indigo',         # marker colour
            markersize=4            # marker size
            )

plt.show()


#%%
longitude_unique = pd.unique(events[['longitude']].values)
longitude_unique.sort()
longitude_unique






















        
        