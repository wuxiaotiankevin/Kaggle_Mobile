# Kaggle Competition
# Deadline: 8/30/2016
# TalkingData Mobile User Demographics
setwd('C:/Brown/2016Summer/Kaggle/TalkingData Mobile User Demographics')
library(xgboost)
library(readr)
library(bigmemory)

# Load data
test <- read.csv('Features/test.csv', numerals='no.loss')
train <- read.csv('Features/train.csv') # , numerals='no.loss')
train <- read_csv('Features/train.csv')
test$X <- NULL
train$X <- NULL

# Make matrix
# Combine and make categorical to numerical
all <- rbind(train[, 3:4], test[, 2:3])
all$phone_brand <- as.numeric(all$phone_brand)
all$device_model <- as.numeric(all$device_model)
train[, 3:4] <- all[1:nrow(train), ]
test[, 2:3] <- all[(nrow(train)+1):nrow(all), ]
train$group <- as.numeric(train$group)-1


# Parameters Cross Validation
dtrain <- xgb.DMatrix(data.matrix(train[, -c(1:2)]), label=train$group, missing=NA)
xgbparams <- list(objective="multi:softprob", 
                  num_class=12, 
                  eval_metric="mlogloss", 
                  eta=0.1, 
                  max_depth=5, 
                  subsample=0.8, 
                  colsample_bytree=0.8,
                  min_child_weigth = 1,
                  gamma=0)

grid.search <- function(par1, par2, par1.val, par2.val, dat.train=dtrain, show.result=TRUE) {
  len1 <- length(par1.val)
  len2 <- length(par2.val)
  out <- NULL
  for (i in 1:len1) {
    for (j in 1:len2) {
      xgbparams[par1] <- par1.val[i]
      xgbparams[par2] <- par2.val[j]
      cv.result <- xgb.cv(xgbparams, dat.train, nrounds = 500, 
                          nfold = 5, early.stop.round = 20, 
                          verbose = show.result)
      idx <- which.min(cv.result$test.mlogloss.mean)
      out <- rbind(out, c(par1.val[i], par2.val[j], 
                          cv.result$test.mlogloss.mean[idx], 
                          cv.result$test.mlogloss.std[idx], 
                          idx))
      print(c(i, j))
    }
  }
  colnames(out) <- c(par1, par2, 'cv_mlogloss', 'cv_sd', 'round_num')
  return(out)
}

# Tune max_depth and min_child_weight
max_depth.min_child_weight <- grid.search('max_depth', 'min_child_weight', seq(3, 10, 2), seq(1, 6, 2)) 
max_depth.min_child_weight <- grid.search('max_depth', 'min_child_weight', seq(2, 3, 1), seq(5, 9, 2))
max_depth.min_child_weight <- grid.search('max_depth', 'min_child_weight', seq(2, 4, 1), seq(4, 6, 1)) 
max_depth.min_child_weight[which.min(max_depth.min_child_weight[, 3]), ]

xgbparams$max_depth <- 3
xgbparams$min_child_weigth <- 4


# Tune gamma
tune.gamma <- grid.search('max_depth', 'gamma', 3, seq(0, 0.5, 0.1)) 
# write.csv(tune.gamma, 'tune.gamma.csv', row.names = FALSE)
xgbparams$gamma <- tune.gamma[which.min(tune.gamma[, 3]), 2]

xgbparams$gamma <- 0.3


# Tune subsample and colsample_bytree
tune.subsample.colsample_bytree <- grid.search('subsample', 'colsample_bytree', seq(0.6, 0.9, 0.1), seq(0.6, 0.9, 0.1)) 
write.csv(tune.subsample.colsample_bytree, 'tune.subsample.colsample_bytree.csv', row.names = FALSE)
tune <- grid.search('subsample', 'colsample_bytree', 0.8, seq(0.4, 0.6, 0.1)) 
tune.subsample.colsample_bytree[which.min(tune.subsample.colsample_bytree[, 3]), ]

xgbparams$subsample <- 0.8
xgbparams$colsample_bytree <- 0.6


# Tuning Regularization Parameters
xgbparams$alpha <- 0
tune.reg_alpha <- grid.search('max_depth', 'alpha', 3, c(1e-5, 1e-2, 0.1, 1, 100)) 


# Reducing Learning Rate
xgbparams$eta <- 0.05
cv.result <- xgb.cv(xgbparams, dtrain, nrounds = 500, 
                    nfold = 5, early.stop.round = 20, 
                    verbose = TRUE)





# Train model with all data and tuned parameters
# dtrain <- xgb.DMatrix(data.matrix(train_train[, -c(1:3)]), label=train_train$group, missing=NA)
dtest <- xgb.DMatrix(data.matrix(test[, -1]), missing=NA)
train.gdbt <- xgb.train(params=xgbparams, data=dtrain, nrounds=500)


# Prediction and make submission file
xgbpred<-predict(train.gdbt,newdata=dtest)

submit <- t(matrix(xgbpred, nrow=12, ncol=length(xgbpred)/12))
colnames(submit) <- c('F23-','F24-26','F27-28','F29-32','F33-42','F43+','M22-','M23-26','M27-28','M29-31','M32-38','M39+')
submit <- cbind(as.character(test$device_id), submit)
colnames(submit)[1] <- 'device_id'

write.csv(submit, 'submission_160818.csv', row.names = FALSE)
