OBD_ELM <- function(xin, yin, Z_current, par_bias_input) {
  n_samples <- nrow(xin)
  
  if (par_bias_input == 1) {
    xin_aug_input <- cbind(1, xin)
  } else {
    xin_aug_input <- xin
  }
  
  p_effective <- ncol(Z_current)
  if (p_effective == 0) { # Nenhum neurônio oculto
    H_current <- matrix(0, nrow = n_samples, ncol = 0)
  } else {
    H_current <- tanh(xin_aug_input %*% Z_current)
  }
  
  Haug_current <- cbind(1, H_current) # Adiciona coluna de bias para os pesos de saída
  num_cols_Haug <- ncol(Haug_current)
  
  if (num_cols_Haug == 0) { # Impossível treinar
    return(list(w = matrix(0,0,ncol(yin)), Haug = Haug_current, Z = Z_current, p_effective = 0))
  }
  
  # Cálculo dos pesos da camada de saída 'w' usando pseudo-inversa (sem regularização L2)
  # w = Haug^+ * yin
  w_current <- corpcor::pseudoinverse(Haug_current) %*% yin
  
  return(list(w = w_current, Haug = Haug_current, Z = Z_current, p_effective = p_effective))
}

calculate_saliencies_for_w <- function(Haug, w) {
  if (ncol(Haug) == 0 || length(w) == 0 || nrow(w) != ncol(Haug)) {
    return(numeric(0)) # Retorna vetor numérico vazio se não houver o que calcular
  }
  diag_Hessian_terms <- diag(t(Haug) %*% Haug)
  saliencies <- 0.5 * diag_Hessian_terms * (w^2)
  return(as.vector(saliencies)) # Retorna como vetor
}

YELM_OBD <- function(xin, Z_pruned, W_pruned, par_bias_input){
  if (par_bias_input == 1){
    xin_aug_input <- cbind(1, xin)
  } else {
    xin_aug_input <- xin
  }
  
  # Caso especial: Z_pruned não tem colunas (todos neurônios ocultos podados)
  if (is.null(Z_pruned) || ncol(Z_pruned) == 0) {
    # A predição dependerá apenas do peso de bias da camada de saída (primeiro elemento de W_pruned)
    if (!is.null(W_pruned) && nrow(W_pruned) >= 1) {
      # Assume que W_pruned[1] é o bias de saída
      # Cria uma coluna de 1s para multiplicar pelo bias
      Haug <- matrix(1, nrow = nrow(xin_aug_input), ncol = 1)
      # Usa apenas o peso de bias. Se W_pruned tem mais linhas, elas são ignoradas.
      Yhat <- sign(Haug %*% W_pruned[1, , drop = FALSE]) 
    } else {
      # Sem neurônios e sem bias, retorna predição neutra ou erro
      Yhat <- matrix(0, nrow = nrow(xin_aug_input), ncol = 1)
    }
    return(Yhat)
  }
  
  H <- tanh(xin_aug_input %*% Z_pruned)
  Haug <- cbind(1, H) # Adiciona coluna de bias para os pesos de saída
  Yhat <- sign(Haug %*% W_pruned)
  
  return(Yhat)
}
