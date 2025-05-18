ELM <- function(xin, yin, p, par) {
  n <- dim(xin)[2]
  
  if(par == 1){
    xin <- cbind(1, xin)
    Z <- matrix(runif((n+1)*p, -0.5, 0.5), nrow = (n+1), ncol = p)
  } else {
    Z <- matrix(runif(n*p, -0.5, 0.5), nrow = n, ncol = p)
  }
  
  H <- tanh(xin %*% Z)
  Haug <- cbind(1, H)
  
  w <- pseudoinverse(Haug) %*% yin
  
  return( list(w, H, Z))
} 

## saída ELM para valores -1 e +1
YELM <- function(xin, Z, W, par){
  n <- dim(xin)[2]
  
  if (par == 1){
    xin <- cbind(1, xin)
  }
  
  H <- tanh(xin %*% Z)
  Haug <- cbind(1,H)
  Yhat <- sign(Haug %*% W)
  
  return(Yhat)
}