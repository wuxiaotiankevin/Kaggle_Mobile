options(stringsAsFactors=F,scipen=99)
rm(list=ls());gc()
require(data.table)

# setwd('C:/Brown/2016Summer/Kaggle/TalkingData Mobile User Demographics')
label_train <- fread("Data/gender_age_train.csv",
                     colClasses=c("character","character",
                                  "integer","character"))
label_test <- fread("Data/gender_age_test.csv",
                    colClasses=c("character"))
label_test$gender <- label_test$age <- label_test$group <- NA
label <- rbind(label_train,label_test)
setkey(label,device_id)
rm(label_test,label_train);gc()

brand <- fread("Data/phone_brand_device_model.csv",
               colClasses=c("character","character","character"))
setkey(brand,device_id)
brand0 <- unique(brand,by=NULL)
brand0 <- brand0[sample(nrow(brand0)),]
brand2 <- brand0[-which(duplicated(brand0$device_id)),]
label1 <- merge(label,brand2,by="device_id",all.x=T)
rm(brand,brand0,brand2);gc()

# apps
events <- fread("Data/events.csv",
                colClasses=c("character","character","character",
                             "numeric","numeric"))
setkeyv(events,c("device_id","event_id"))
event_app <- fread("Data/app_events.csv",
                   colClasses=rep("character",4))
setkey(event_app,event_id)

events <- unique(events[,list(device_id,event_id)],by=NULL)
event_apps <- event_app[,list(apps=paste(unique(app_id),collapse=",")),by="event_id"]
device_event_apps <- merge(events,event_apps,by="event_id")
rm(events,event_app,event_apps);gc()

f_split_paste <- function(z){paste(unique(unlist(strsplit(z,","))),collapse=",")}
device_apps <- device_event_apps[,list(apps=f_split_paste(apps)),by="device_id"]
rm(device_event_apps,f_split_paste);gc()

tmp <- strsplit(device_apps$apps,",")
device_apps <- data.table(device_id=rep(device_apps$device_id,
                                        times=sapply(tmp,length)),
                          app_id=unlist(tmp))
rm(tmp)

# dummy
d1 <- label1[,list(device_id,phone_brand)]
label1$phone_brand <- NULL
d2 <- label1[,list(device_id,device_model)]
label1$device_model <- NULL
d3 <- device_apps
rm(device_apps)
d1[,phone_brand:=paste0("phone_brand:",phone_brand)]
d2[,device_model:=paste0("device_model:",device_model)]
d3[,app_id:=paste0("app_id:",app_id)]
names(d1) <- names(d2) <- names(d3) <- c("device_id","feature_name")
dd <- rbind(d1,d2,d3)
rm(d1,d2,d3);gc()

require(Matrix)
ii <- unique(dd$device_id)
jj <- unique(dd$feature_name)
id_i <- match(dd$device_id,ii)
id_j <- match(dd$feature_name,jj)
id_ij <- cbind(id_i,id_j)
M <- Matrix(0,nrow=length(ii),ncol=length(jj),
            dimnames=list(ii,jj),sparse=T)
M[id_ij] <- 1
rm(ii,jj,id_i,id_j,id_ij,dd);gc()

x <- M[rownames(M) %in% label1$device_id,]
id <- label1$device_id[match(rownames(x),label1$device_id)]
y <- label1$group[match(rownames(x),label1$device_id)]
rm(M,label1)

# level reduction
x_train <- x[!is.na(y),]
tmp_cnt_train <- colSums(x_train)
x <- x[,tmp_cnt_train>0 & tmp_cnt_train<nrow(x_train)]
rm(x_train,tmp_cnt_train)

require(xgboost)
(group_name <- na.omit(unique(y)))
idx_train <- which(!is.na(y))
idx_test <- which(is.na(y))
train_data <- x[idx_train,]
test_data <- x[idx_test,]
train_label <- match(y[idx_train],group_name)-1
test_label <- match(y[idx_test],group_name)-1
dtrain <- xgb.DMatrix(train_data,label=train_label,missing=NA)
dtest <- xgb.DMatrix(test_data,label=test_label,missing=NA)

# param <- list(booster="gblinear",
#               num_class=length(group_name),
#               objective="multi:softprob",
#               eval_metric="mlogloss",
#               eta=0.01,
#               lambda=5,
#               lambda_bias=0,
#               alpha=2)
# watchlist <- list(train=dtrain)
# set.seed(2016)
# fit_cv <- xgb.cv(params=param,
#                  data=dtrain,
#                  nrounds=100000,
#                  watchlist=watchlist,
#                  nfold=5,
#                  early.stop.round=20,
#                  verbose=1)

# Tuning
xgbparams <- list(objective="multi:softprob", 
                  num_class=12, 
                  eval_metric="mlogloss", 
                  eta=0.2, 
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
      cv.result <- xgb.cv(xgbparams, dat.train, nrounds = 1000, 
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


# cv_output <- xgb.cv(xgbparams, dtrain, nrounds = 1000, nfold = 5, 
#                     early.stop.round = 20, verbose = TRUE)

################################################################
################################################################
# Tune max_depth and min_child_weight
max_depth.min_child_weight <- grid.search('max_depth', 'min_child_weight', seq(3, 10, 2), seq(1, 6, 2)) 
max_depth.min_child_weight <- grid.search('max_depth', 'min_child_weight', seq(2, 3, 1), seq(5, 9, 2))



# max_depth.min_child_weight <- grid.search('max_depth', 'min_child_weight', seq(2, 4, 1), seq(4, 6, 1)) 
# max_depth.min_child_weight[which.min(max_depth.min_child_weight[, 3]), ]
# 
# xgbparams$max_depth <- 3
# xgbparams$min_child_weigth <- 4
# 
# 
# # Tune gamma
# tune.gamma <- grid.search('max_depth', 'gamma', 3, seq(0, 0.5, 0.1)) 
# # write.csv(tune.gamma, 'tune.gamma.csv', row.names = FALSE)
# xgbparams$gamma <- tune.gamma[which.min(tune.gamma[, 3]), 2]
# 
# xgbparams$gamma <- 0.3
# 
# 
# # Tune subsample and colsample_bytree
# tune.subsample.colsample_bytree <- grid.search('subsample', 'colsample_bytree', seq(0.6, 0.9, 0.1), seq(0.6, 0.9, 0.1)) 
# write.csv(tune.subsample.colsample_bytree, 'tune.subsample.colsample_bytree.csv', row.names = FALSE)
# tune <- grid.search('subsample', 'colsample_bytree', 0.8, seq(0.4, 0.6, 0.1)) 
# tune.subsample.colsample_bytree[which.min(tune.subsample.colsample_bytree[, 3]), ]
# 
# xgbparams$subsample <- 0.8
# xgbparams$colsample_bytree <- 0.6
# 
# 
# # Tuning Regularization Parameters
# xgbparams$alpha <- 0
# tune.reg_alpha <- grid.search('max_depth', 'alpha', 3, c(1e-5, 1e-2, 0.1, 1, 100)) 
# 
# 
# # Reducing Learning Rate
# xgbparams$eta <- 0.05
# cv.result <- xgb.cv(xgbparams, dtrain, nrounds = 500, 
#                     nfold = 5, early.stop.round = 20, 
#                     verbose = TRUE)
# ################################################################
# ################################################################
# 
# 
# 
# 
# 
# 
# 
# 
# 
# ntree <- 440
# set.seed(2016)
# # fit_xgb <- xgb.train(params=param,
# #                      data=dtrain,
# #                      nrounds=ntree,
# #                      watchlist=watchlist,
# #                      verbose=1)
# fit_xgb <- xgb.train(params=xgbparams, data=dtrain, nrounds=ntree)
# pred <- predict(fit_xgb,dtest)
# pred_detail <- t(matrix(pred,nrow=length(group_name)))
# res_submit <- cbind(id=id[idx_test],as.data.frame(pred_detail))
# colnames(res_submit) <- c("device_id",group_name)
# write.csv(res_submit,file="submit_sparse_xgboost_tuned.csv",row.names=F,quote=F)
# 
