ELM_OBD <- function(xin, yin, Z_current, par_bias_input) {
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
  
  # Cálculo dos pesos da camada de saída 'w' usando pseudo-inversa
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

YELM_OBD <- function(xin, Z_final, W_final, par_bias_input){
  if (par_bias_input == 1){
    xin_aug_input <- cbind(1, xin)
  } else {
    xin_aug_input <- xin
  }
  
  # Caso especial: Z_pruned não tem colunas (todos neurônios ocultos podados)
  if (is.null(Z_final) || ncol(Z_final) == 0) {
    # A predição dependerá apenas do peso de bias da camada de saída (primeiro elemento de W_pruned)
    if (!is.null(W_final) && nrow(W_final) >= 1) {
      # Assume que W_pruned[1] é o bias de saída
      # Cria uma coluna de 1s para multiplicar pelo bias
      Haug <- matrix(1, nrow = nrow(xin_aug_input), ncol = 1)
      # Usa apenas o peso de bias. Se W_pruned tem mais linhas, elas são ignoradas.
      Yhat <- sign(Haug %*% W_final[1, , drop = FALSE]) 
    } else {
      # Sem neurônios e sem bias, retorna predição neutra ou erro
      Yhat <- matrix(0, nrow = nrow(xin_aug_input), ncol = 1)
    }
    return(Yhat)
  }
  
  H <- tanh(xin_aug_input %*% Z_final)
  Haug <- cbind(1, H) # Adiciona coluna de bias para os pesos de saída
  Yhat <- sign(Haug %*% W_final)
  
  return(Yhat)
}

run_obd_elm_process <- function(xin, yin, p_initial, par_bias_input, 
                                pruning_fraction, obd_max_iterations, min_neurons, 
                                seed_val = NULL) {
  
  if (!is.null(seed_val)) set.seed(seed_val) 
  
  n_features_xin <- ncol(xin)
  if (par_bias_input == 1) {
    Z_full <- matrix(runif((n_features_xin + 1) * p_initial, -0.5, 0.5), nrow = (n_features_xin + 1), ncol = p_initial)
  } else {
    Z_full <- matrix(runif(n_features_xin * p_initial, -0.5, 0.5), nrow = n_features_xin, ncol = p_initial)
  }
  
  active_neuron_indices_in_Z_full <- 1:p_initial
  Z_current_pruned <- Z_full
  W_final_model <- NULL 
  
  
  for (iter_obd_loop in 1:obd_max_iterations) {
    num_active_neurons <- length(active_neuron_indices_in_Z_full)
    
    if (num_active_neurons < min_neurons || num_active_neurons == 0) {
      # cat("      Número de neurônios OBD atingiu o limite. Parando poda interna.\n")
      break
    }
    
    # Passo de treino OBD (você chamou de ELM_OBD no seu arquivo)
    train_result <- ELM_OBD(xin, yin, Z_current_pruned, par_bias_input)
    w_current_iter <- train_result$w
    Haug_current_iter <- train_result$Haug
    W_final_model <- w_current_iter 
    
    saliencies_for_w <- calculate_saliencies_for_w(Haug_current_iter, w_current_iter)
    
    if (length(saliencies_for_w) <= 1 && num_active_neurons > 0) { break }
    if (length(saliencies_for_w) <= 1 && num_active_neurons == 0) { break }
    
    neuron_saliencies_values <- saliencies_for_w[-1] # Ignora saliência do bias de saída
    
    if (length(neuron_saliencies_values) == 0) { break }

    num_to_prune_this_iter <- ceiling(pruning_fraction * length(neuron_saliencies_values))
    
    if (length(neuron_saliencies_values) - num_to_prune_this_iter < min_neurons && 
        length(neuron_saliencies_values) > min_neurons) {
      num_to_prune_this_iter <- length(neuron_saliencies_values) - min_neurons
    } else if (length(neuron_saliencies_values) <= min_neurons) {
      num_to_prune_this_iter <- 0 
    }
    
    if (num_to_prune_this_iter <= 0) {
      if(num_active_neurons <= min_neurons) break; 
      next 
    }
    
    indices_to_prune_local <- order(neuron_saliencies_values)[1:num_to_prune_this_iter]
    indices_in_Z_full_to_remove <- active_neuron_indices_in_Z_full[indices_to_prune_local]
    active_neuron_indices_in_Z_full <- setdiff(active_neuron_indices_in_Z_full, indices_in_Z_full_to_remove)
    
    Z_current_pruned <- Z_full[, active_neuron_indices_in_Z_full, drop = FALSE]
  }
  
  # Retreino final com o Z_current_pruned definitivo
  # Se o loop parou, W_final_model pode ser do Z anterior à última tentativa de poda.
  final_train_output <- ELM_OBD(xin, yin, Z_current_pruned, par_bias_input)
  W_final_model <- final_train_output$w
  
  return(list(W = W_final_model, Z = Z_current_pruned, p_final = ncol(Z_current_pruned)))
}
