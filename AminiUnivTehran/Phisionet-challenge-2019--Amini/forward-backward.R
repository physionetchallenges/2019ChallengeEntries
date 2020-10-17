# -------------------------------------------------
# Forward and backward probabilities
# -------------------------------------------------
# -------------------------------------------------

norm.HMM.lalphabeta<-function(xx,m,mu,sigma,gamma,delta=NULL){
	if(is.null(delta)) delta<-solve(t(diag(m)-gamma+1),rep(1,m))
	xw <-xx
	p1 <- nrow(xx)
	p <- nrow(xx)-1
	w <- xx[p1,]
	xx <- xx[-p1,]
	if(is.null(dim(xx))) xx<-as.matrix(xx)
	n <- ncol(xx)
	wxmat<-matrix(c(rep(c(1,0),m-1),c(0,1)),2,m)
	lalpha  <- lbeta<-matrix(NA,m,n)
	lallprobs <- allprobs <-matrix(NA,n,m)
	no.miss<-which(apply(is.nan(xx),2,sum)==0)
	with.miss<-which(apply(is.nan(xx),2,sum)> 0 & apply(is.nan(xx),2,sum)< p)
	all.miss<-which(apply(is.nan(xx),2,sum)==p)
	for(i in no.miss){
		for(j in 1:m){
			lallprobs[i,j] = dmvnorm(xx[,i],mu[j,],diag(sigma[[j]]),log=TRUE)+log(wxmat[w[i]+1,j])
		}
	}
	for(i in all.miss){
		for(j in 1:m){
			dett=prod(sigma[[j]])
			if(dett<=0) dett=1e-12
			lallprobs[i,j] = -0.5*log(dett)-p/2*(1+log(2*pi))+log(wxmat[w[i]+1,j])
		}
	}
	for(i in with.miss){
		xx2=xx[,i]
		misxx=is.nan(xx2)
		xx.mis=xx2[misxx]
		xx.obs=xx2[!misxx]
		pm=length(xx.mis)
		for(j in 1:m){
			dett=prod(sigma[[j]][misxx])
			if(dett<=0) dett=1e-12
			a = -0.5*log(dett)
			if(length(sigma[[j]][!misxx])==1){
				b = dnorm(xx.obs,mu[j,!misxx],sigma[[j]][!misxx],log=TRUE)
			}else{
				b = dmvnorm(xx.obs,mu[j,!misxx],diag(sigma[[j]][!misxx]),log=TRUE)
			}
			lallprobs[i,j] = a-pm/2*(1+log(2*pi))+ b+log(wxmat[w[i]+1,j])
		}
	}
	allprobs<-exp(lallprobs)
	foo  <- delta*allprobs[1,]
	sumfoo <- sum(foo)
	lscale <- log(sumfoo)
	if(all(foo==0)){
		foo  <- delta
		lalpha[,1] <- log(foo)
		lscale=0
	}else{
		foo<-foo/sumfoo
		lalpha[,1] <- log(foo)+lscale
	}
if(n>1){
	for (i in 2:n){
		foo1 <- foo%*%gamma*allprobs[i,]
		if(all(foo1==0)){
			foo <- foo%*%gamma
		}else{
			foo <- foo1
		}
		sumfoo <- sum(foo)
		lscale <- lscale+log(sumfoo)
		foo  <- foo/sumfoo
		lalpha[,i] <- log(foo)+lscale
	}
}
	lbeta[,n]  <- rep(0,m)
	foo <- rep(1/m,m)
	lscale <- log(m)
if(n>1){
	for (i in (n-1):1){
		foo1<- gamma%*%(allprobs[i+1,]*foo)
		if(all(foo1==0)){
			foo <- gamma%*%foo
		}else{
			foo <- foo1
		}
		lbeta[,i]<- log(foo)+lscale
		sumfoo  <- sum(foo)
		foo  <- foo/sumfoo
		lscale <- lscale+log(sumfoo)
	}
}
	list(la=lalpha,lb=lbeta)
}
