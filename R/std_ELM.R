ELM <- function(xin, yin, p, par_bias_input) {
  n <- dim(xin)[2]
  
  if(par_bias_input == 1){
    xin <- cbind(1, xin)
    Z <- matrix(runif((n+1)*p, -0.5, 0.5), nrow = (n+1), ncol = p)
  } else {
    Z <- matrix(runif(n*p, -0.5, 0.5), nrow = n, ncol = p)
  }
  
  H <- tanh(xin %*% Z)
  Haug <- cbind(1, H)
  
  w <- pseudoinverse(Haug) %*% yin
  
  p_final_val <- p # Para ELM padrão, o número de neurônios é o 'p' de entrada
  
  # Retorne a lista nomeada corretamente
  return(list(W = w, Z = Z, p_final = p_final_val))
} 

## saída ELM para valores -1 e +1
YELM <- function(xin, Z_final, W_final, par_bias_input){
  n <- dim(xin)[2]
  
  if (par_bias_input == 1){
    xin <- cbind(1, xin)
  }
  
  H <- tanh(xin %*% Z_final)
  Haug <- cbind(1,H)
  Yhat <- sign(Haug %*% W_final)
  
  return(Yhat)
}