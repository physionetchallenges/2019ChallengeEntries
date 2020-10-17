##Tzvi Aviv
## xgb model for sepsis detection 
## open data 
##open the psv files in setA and setB and train an xgboost with selected variables
## impute NA's and train larger sets
## train the model to work with X that include a time series (1 Y per multiple X)

library(data.table)
library(zoo)
library(xgboost)
library(Metrics)
library(dplyr)
library(caret)
library(scales)
library(ggridges)
library(tidyr)
library(scales)

#start loading setA

temp = list.files("training_setA/training",pattern="*.psv")
files_df<-lapply(temp, function(i) {fread(paste("training_setA/training/",i,sep=""))})
df<-do.call(rbind.data.frame, files_df)

dim(df)

#check duplicate raws - do not remove ICULOS column

print(names(df))
dfA<-df %>% distinct()
dim(dfA)

#load setB

tempB = list.files("training_setB/training_setB",pattern="*.psv")
files_dfB<-lapply(tempB, function(i) {fread(paste("training_setB/training_setB/",i,sep=""))})
dfB<-do.call(rbind.data.frame, files_dfB)
print(names(dfB))
dim(dfB)
dfB<-dfB %>% distinct()
dim(dfB)

#combine the two df's

df<-rbind(dfA,dfB)

dim(df)
names(df)

#save(df, file="dfAB.rda")

#load(file="dfAB.rda")

df$SepsisLabel<-as.factor(df$SepsisLabel)
df$Gender<-as.factor(df$Gender)
df_select<-df[,c( "ICULOS" ,"Age", "HR", "MAP", "O2Sat", "SBP", "Resp", "Temp", "Potassium", "Hct")]

summary(df_select)

#remove all na's for training

#nonadf<-na.omit(df_select)

#impute with means
df_select[] <- lapply(df_select, function(x) { 
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  x
})

summary(df_select)

df_select %>%
  gather(variable, value, -SepsisLabel) %>%
  ggplot(aes(y = as.factor(variable), 
             fill = as.factor(SepsisLabel), 
             x = percent_rank(value))) +
  geom_density_ridges()

X<-as.matrix(df_select)

#scaling from 0 to 1 
#X <- apply(X, 2, function(x) rescale(x, to = c(0, 1)))

summary(X)

#genrating the label
as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}
Y<-as.numeric.factor(df$SepsisLabel)

N_FILES = nrow(X) #number of cases
SPLT = 0.8 #80% train, 20% test
b = floor(SPLT*N_FILES)
x_train = X[1:b,]
x_test = X[(b+1):N_FILES,]

y_train = Y[1:b]
y_test = Y[(b+1):N_FILES]

xgb_train <- xgb.DMatrix(data=x_train, label=y_train)
xgb_test <- xgb.DMatrix(data=x_test, label=y_test)

#xgb parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = xgb_train, nrounds = 200, nfold = 5, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 30, maximize = F)

xgb7 <- xgb.train (params = params, data = xgb_train, nrounds = 19, watchlist = list(val=xgb_test,train=xgb_train), print_every_n = 10, early_stoping_rounds = 10, maximize = F , eval_metric = "error")

save(xgb7, file="xgb7.rda")

xgb7

mat7 <- xgb.importance (feature_names = colnames(X),model = xgb7)

#plot order of importance 
xgb.plot.importance (importance_matrix = mat7[1:10]) 

#predict the test set with the model
xgbpred7 <- predict (xgb7,xgb_test)

summary(xgbpred7)

xgbpred7 <- ifelse (xgbpred7 > 0.05,1,0)

confusionMatrix (as.factor(xgbpred7), as.factor(y_test))

#accuracy 0.93
##balanced accuracy is 0.65

#save model
xgb.save(xgb6, "xgboost6_ta.model")

# save model to R's raw vector
rawVec <- xgb.save.raw(xgb5)

