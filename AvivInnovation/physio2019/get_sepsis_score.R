#!/usr/bin/Rscript
# Tzvi Aviv tzvika.aviv@gmail.com
#scoring with an 12-hr sliding window model trained in xgboost on setA and B centered July 29th
# centering will be done on the new data 
library(xgboost)

counter<-0
col_sums<-matrix(0,1,82)

get_sepsis_score = function(data, model){
counter <<- counter+1
#print(counter)   
temp<- matrix(data=NA,nrow=12,ncol=40)
num_row<-nrow(tail(data,12))    
temp[1:num_row,]<-tail(data[,1:40, drop=T],num_row)
mean<-apply(temp,2,mean,na.rm=T)
sd<-apply(temp,2,sd,na.rm=T)
sat_fi<-mean[2]/mean[11]
sat_fi[is.infinite(sat_fi)] = 5000
plat_wbc<-mean[34]/mean[32]
x<-c(mean,sat_fi,plat_wbc,sd)
x<-matrix(x,1,82)
col_sums <<- col_sums + x
colmeans<-col_sums/counter

x<-data.frame(x-colmeans)
#select top 20 variables
x<-x[,c('X40', 'X35', 'X39', 'X3', 'X1', 'X7', 'X4', 'X41' ,'X5', 'X2' ,'X43' ,'X6', 'X22', 'X44', 'X49', 'X46', 'X45', 'X47' ,'X32', 'X48')]

score = predict(model,as.matrix(x))
label = score > 0.03

    predictions = as.matrix(cbind(score, label)) 
    return(predictions)

}

load_sepsis_model = function(){
xgb.load("xgb_windowAB_center_July29")

}

