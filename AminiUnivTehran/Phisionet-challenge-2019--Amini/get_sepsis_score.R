#!/usr/bin/Rscript
source("dmvnorm.R")
source("forward-backward.R")
source("probs.R")
source("entry-prediction.R")
load_sepsis_model = function(){
	load("estimates.Rdata")
	model=list(thr=0.2,m=3,h=6,mu=muhat,sig=sigmahat,gam=gammahat)
	return(model)
}
get_sepsis_score=function(data,model){
	num_rows=nrow(data)
	num_cols=ncol(data)	
	x=matrix(data[,1:34],num_rows,34)
	x=cbind(x,0)
	pred<-entry.prediction(x,model$thr,model$mu,model$sig,model$gam,h=model$h)
	pred
}
