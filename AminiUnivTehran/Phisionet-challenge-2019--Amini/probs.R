# -------------------------------------------------
# State prediction
# -------------------------------------------------
# This  function  computes  the  probability  Pr(Ct=i|X(n))  
# for t=n+1,...,n+H and i=1,...,m.
# -------------------------------------------------

predict.probs <- function(xx,m,mu,sigma,gamma,delta=NULL,H=1,...){
	if(!is.null(xx)){
	if(is.null(delta)) delta<-solve(t(diag(m)-gamma+1),rep(1,m))
	if(is.null(dim(xx))) xx=as.matrix(xx)
	n <- ncol(xx)
	fb <- norm.HMM.lalphabeta(xx,m,mu,sigma,gamma,delta=delta)
	la <- fb$la
	c <- max(la[,n])
	llk <- c+log(sum(exp(la[,n]-c)))
	statepreds <- matrix(NA,ncol=H,nrow=m)
	foo1 <- exp(la[,n]-llk)
	foo2 <- diag(m)
	for (i in 1:H){
		foo2 <- foo2%*%gamma
		statepreds[,i] <- foo1%*%foo2
	}
	statepreds
}else{
matrix(NA,ncol=H,nrow=m)
}
}