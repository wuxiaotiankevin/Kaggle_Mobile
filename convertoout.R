# This file converts the output to submission file
setwd('C:/Brown/2016Summer/Kaggle/TalkingData Mobile User Demographics')

subsamp <- read.csv('Data/sample_submission.csv', colClasses = c(device_id = 'character'))
out1 <- read.csv('predict_keras.csv', header = F)
out2 <- read.csv('predict_logistic.csv', header = F)
out3 <- read.csv('predict_random_forest.csv', header = F)
out4 <- read.csv('predict_xgboost.csv', header = F)

out <- (out1 + out3 + out4 ) / 3

out <- cbind(subsamp$device_id, out)

colnames(out) <- c('device_id', 'F23-','F24-26','F27-28','F29-32','F33-42','F43+','M22-','M23-26','M27-28','M29-31','M32-38','M39+')

write.csv(out, 'submission.csv', row.names = F)


train <- read.csv('Data/gender_age_train.csv', colClasses = c(device_id = 'character'))
test <- read.csv('Data/')

id <- as.numeric(train$device_id)
id.sign <- as.numeric(id>0)
table(id.sign, train$gender)
