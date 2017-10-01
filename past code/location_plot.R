# Kaggle Competition
# Deadline: 8/30/2016
# TalkingData Mobile User Demographics

setwd('C:/Brown/2016Summer/Kaggle/TalkingData Mobile User Demographics')
events <- read.csv('Data/events.csv')
loc <- events[, 4:5]
loc <- unique(loc)
loc <- loc[order(loc$latitude), ]
loc <- loc[loc$latitude!=0, ]

library(rworldmap)
# Worldwide
newmap <- getMap(resolution = "low")
plot(newmap, xlim = range(loc$longitude), ylim = range(loc$latitude), asp = 1)
points(loc$longitude, loc$latitude, cex = .6,
       col=rgb(red=0.2, green=0.2, blue=1.0, alpha=0.7))

# Around China
plot(newmap, xlim = c(90, 125), ylim = c(10, 60), asp = 1)
points(loc$longitude, loc$latitude, cex = .6, pch = 16,
       col=rgb(red=0.2, green=0.2, blue=1.0, alpha=0.05))
