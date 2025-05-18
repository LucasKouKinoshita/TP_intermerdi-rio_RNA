library("corpcor")

ELM_pruned <- function(xin, yin, p_initial, par, pruning_rate = 0.0, seed_val = NULL) {
  # Definir semente para reprodutibilidade da geração de Z e da poda, se fornecida
  if (!is.null(seed_val)) {
    set.seed(seed_val)
  }
  
  n_features <- dim(xin)[2]
  
  # Adicionar bias à camada de entrada (coluna de 1s em xin)
  if (par == 1) {
    xin_aug <- cbind(1, xin)
    # Z: pesos entre entrada (+bias) e camada oculta. Dim: (n_features+1) x p_initial
    Z_full <- matrix(runif((n_features + 1) * p_initial, -0.5, 0.5), nrow = (n_features + 1), ncol = p_initial)
  } else {
    xin_aug <- xin
    # Z: pesos entre entrada e camada oculta. Dim: n_features x p_initial
    Z_full <- matrix(runif(n_features * p_initial, -0.5, 0.5), nrow = n_features, ncol = p_initial)
  }
  
  # H_full: Matriz de saída da camada oculta (antes da poda). Dim: N_samples x p_initial
  H_full <- tanh(xin_aug %*% Z_full)
  
  # Poda Aleatória
  current_p <- p_initial
  Z_final <- Z_full
  H_final <- H_full
  
  if (pruning_rate > 0 && pruning_rate < 1) {
    num_neurons_to_keep <- round(p_initial * (1 - pruning_rate))
    # Garantir que pelo menos 1 neurônio seja mantido se p_initial > 0
    if (num_neurons_to_keep == 0 && p_initial > 0) num_neurons_to_keep <- 1
    if (num_neurons_to_keep > p_initial) num_neurons_to_keep <- p_initial # Caso improvável
    
    if (num_neurons_to_keep < p_initial) {
      # Selecionar aleatoriamente os neurônios a serem mantidos
      kept_neuron_indices <- sort(sample(1:p_initial, num_neurons_to_keep, replace = FALSE))
      
      H_final <- H_full[, kept_neuron_indices, drop = FALSE] # Matriz H podada
      Z_final <- Z_full[, kept_neuron_indices, drop = FALSE] # Matriz Z podada
      current_p <- num_neurons_to_keep
    }
  }
  
  # Haug: Adicionar bias à camada de saída (usando H podado)
  Haug <- cbind(1, H_final) # Dim: N_samples x (current_p + 1)
  
  # w: Pesos da camada de saída. Dim: (current_p + 1) x 1
  w <- pseudoinverse(Haug) %*% yin
  
  return(list(w = w, H = H_final, Z = Z_final, p_final = current_p))
}

## saída ELM para valores -1 e +1 (compatível com Z podado)
YELM_pruned <- function(xin, Z_final, W, par) { # Z_final é o Z após a poda
  # n <- dim(xin)[2] # Não é usado se Z_final já tem as dimensões corretas
  
  if (par == 1) {
    xin_aug <- cbind(1, xin)
  } else {
    xin_aug <- xin
  }
  
  H <- tanh(xin_aug %*% Z_final) # Usa Z_final (potencialmente podado)
  Haug <- cbind(1, H)
  Yhat <- sign(Haug %*% W)
  
  return(Yhat)
}