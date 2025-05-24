graphics.off() 
rm(list = ls()) 
library("mlbench") # Se for usar para outros datasets como BreastCancer
library("corpcor") # Para pseudoinverse
library("plot3D")  # Se for usar para plots
library("GA")      # Para ELM_GAP
library("caret")   # Para createFolds (validação cruzada)

train_elm_for_ga_fitness <- function(xin_f, yin_f, Z_subset_f, par_bias_input_f) {
  if (is.null(Z_subset_f) || ncol(Z_subset_f) == 0) {
    # Nenhum neurônio selecionado, apenas bias de saída.
    Haug_f <- matrix(1, nrow = nrow(xin_f), ncol = 1)
    # Verificar se yin_f tem as dimensões corretas para o produto matricial
    if(nrow(Haug_f) != nrow(yin_f) && ncol(Haug_f) == nrow(yin_f) && nrow(Haug_f) == ncol(yin_f)){ # Transpor yin_f se necessário e possível
      yin_f_corrected <- t(yin_f)
    } else {
      yin_f_corrected <- yin_f
    }
    
    if(nrow(Haug_f) == nrow(yin_f_corrected)){
      w_f <- tryCatch({
        corpcor::pseudoinverse(Haug_f) %*% yin_f_corrected
      }, error = function(e){
        # Em caso de erro com pseudoinversa (ex: yin_f vazio), retornar NULL ou matriz vazia
        matrix(0, nrow=ncol(Haug_f), ncol=if(is.matrix(yin_f_corrected)) ncol(yin_f_corrected) else 1)
      })
    } else {
      # Dimensões incompatíveis mesmo após tentativa de correção
      w_f <- matrix(0, nrow=ncol(Haug_f), ncol=if(is.matrix(yin_f_corrected)) ncol(yin_f_corrected) else 1)
    }
    return(list(w = w_f, Haug = Haug_f, y_pred = Haug_f %*% w_f))
  }
  
  if (par_bias_input_f == 1) {
    xin_aug_f <- cbind(1, xin_f)
  } else {
    xin_aug_f <- xin_f
  }
  
  H_f <- tanh(xin_aug_f %*% Z_subset_f)
  Haug_f <- cbind(1, H_f)
  
  w_f <- corpcor::pseudoinverse(Haug_f) %*% yin_f # ELM padrão
  y_pred_f <- Haug_f %*% w_f
  
  return(list(w = w_f, Haug = Haug_f, y_pred = y_pred_f))
}

YELM_GAP_predict <- function(xin_pred, Z_final, W_final, par_bias_input_pred){
  if (par_bias_input_pred == 1){
    xin_aug_pred <- cbind(1, xin_pred)
  } else {
    xin_aug_pred <- xin_pred
  }
  
  if (is.null(Z_final) || ncol(Z_final) == 0) { # Nenhum neurônio oculto
    if (!is.null(W_final) && nrow(W_final) == 1) { # Apenas peso de bias
      Haug_pred <- matrix(1, nrow = nrow(xin_aug_pred), ncol = 1)
      Yhat <- sign(Haug_pred %*% W_final) 
    } else { # Sem neurônios e sem bias, predição aleatória ou erro
      Yhat <- matrix(sample(c(-1,1), size=nrow(xin_aug_pred), replace=TRUE), ncol=1) 
    }
    return(Yhat)
  }
  
  H_pred <- tanh(xin_aug_pred %*% Z_final)
  Haug_pred <- cbind(1, H_pred)
  Yhat <- sign(Haug_pred %*% W_final)
  
  return(Yhat)
}

ELM_GAP <- function(xin, yin, p_initial, par_bias_input, # <--- MUDANÇA AQUI nos nomes
                    # Parâmetros do Algoritmo Genético
                    ga_pop_size = 50, 
                    ga_max_iter = 30, 
                    ga_pmutation = 0.1, 
                    ga_pcrossover = 0.8,
                    ga_elitism_fraction = 0.1, 
                    alpha_fitness = 0.99, 
                    use_press_loo = FALSE, 
                    seed_val = NULL) {
  
  if (!is.null(seed_val)) {
    set.seed(seed_val)
  }
  
  # Use 'xin' e 'yin' internamente onde antes usava 'xin_train' e 'yin_train'
  n_features <- ncol(xin) # <--- MUDANÇA AQUI
  n_samples_train <- nrow(yin) # <--- MUDANÇA AQUI
  
  # ... (resto da sua função ELM_GAP, garantindo que usa 'xin' e 'yin') ...
  # ... especialmente na chamada para fitness_fn_gap:
  # fitness = fitness_fn_gap,
  # Z_full_ref = Z_full_internal, 
  # xin_fit = xin,  # <--- MUDANÇA AQUI
  # yin_fit = yin,  # <--- MUDANÇA AQUI
  # ...
  
  # Exemplo de chamada dentro de ELM_GAP para a função de fitness:
  # (A sua definição de fitness_fn_gap já espera xin_fit, yin_fit, o que está bom,
  #  apenas garanta que ELM_GAP passe xin e yin para esses argumentos)
  #  ga_optim_results <- ga(..., xin_fit = xin, yin_fit = yin, ...) 
  
  # No final, ao treinar o modelo ELM final:
  # final_elm_output <- train_elm_for_ga_fitness(xin, yin, Z_final_model, par_bias_input) # <--- MUDANÇA AQUI
  
  # ... (o corpo da sua função ELM_GAP continua, assegurando o uso de 'xin' e 'yin')
  
  # --- CORPO DA SUA FUNÇÃO ELM_GAP CONTINUA ABAIXO COM AS ADAPTAÇÕES ---
  
  cat("Gerando Z_full com", p_initial, "neurônios iniciais...\n")
  if (par_bias_input == 1) {
    Z_full_internal <- matrix(runif((n_features + 1) * p_initial, -0.5, 0.5), nrow = (n_features + 1), ncol = p_initial)
  } else {
    Z_full_internal <- matrix(runif(n_features * p_initial, -0.5, 0.5), nrow = n_features, ncol = p_initial)
  }
  
  fitness_fn_gap <- function(chromosome_g, Z_full_ref, xin_fit, yin_fit, par_bias_fit, alpha_fit, use_press_fit, p_initial_ref) {
    chromosome_g <- as.integer(round(chromosome_g)) 
    selected_neuron_indices <- which(chromosome_g == 1)
    num_selected_neurons <- length(selected_neuron_indices)
    if (num_selected_neurons == 0) {
      return(alpha_fit * 1.0 + (1 - alpha_fit) * 0.0) 
    }
    Z_subset_fit <- Z_full_ref[, selected_neuron_indices, drop = FALSE]
    elm_res <- train_elm_for_ga_fitness(xin_fit, yin_fit, Z_subset_fit, par_bias_fit) # train_elm_for_ga_fitness já espera xin_f, yin_f
    y_pred_fit <- elm_res$y_pred
    if (use_press_fit) {
      # Implementação PRESS LOO aqui (atualmente simplificado)
      error_rate_val <- 1 - mean(sign(y_pred_fit) == yin_fit) 
      if(is.na(error_rate_val)) error_rate_val <- 1.0
    } else { 
      error_rate_val <- 1 - mean(sign(y_pred_fit) == yin_fit)
      if(is.na(error_rate_val)) error_rate_val <- 1.0
    }
    neuron_rate_val <- num_selected_neurons / p_initial_ref
    fitness_value <- alpha_fit * error_rate_val + (1 - alpha_fit) * neuron_rate_val
    return(fitness_value)
  }
  
  cat("Iniciando Algoritmo Genético...\n")
  ga_optim_results <- ga(type = "binary", 
                         fitness = fitness_fn_gap,
                         Z_full_ref = Z_full_internal, 
                         xin_fit = xin, # Passa 'xin' para 'xin_fit'
                         yin_fit = yin, # Passa 'yin' para 'yin_fit'
                         par_bias_fit = par_bias_input, 
                         alpha_fit = alpha_fitness,
                         use_press_fit = use_press_loo,
                         p_initial_ref = p_initial,
                         nBits = p_initial,
                         popSize = ga_pop_size,
                         maxiter = ga_max_iter,
                         pcrossover = ga_pcrossover,
                         pmutation = ga_pmutation,
                         elitism = max(1, round(ga_elitism_fraction * ga_pop_size)),
                         monitor = FALSE, 
                         seed = seed_val)
  
  best_chromosome <- as.integer(ga_optim_results@solution[1,])
  best_fitness_value <- ga_optim_results@fitnessValue
  cat("AG Concluído. Melhor fitness:", best_fitness_value, "\n")
  final_selected_indices <- which(best_chromosome == 1)
  if (length(final_selected_indices) == 0) {
    warning("GAP-ELM: Nenhum neurônio selecionado.")
    Z_final_model <- Z_full_internal[, c(), drop = FALSE] 
  } else {
    Z_final_model <- Z_full_internal[, final_selected_indices, drop = FALSE]
  }
  cat("Treinando modelo final ELM com", ncol(Z_final_model), "neurônios (GAP)...\n")
  # Usa 'xin' e 'yin' para o treino final
  final_elm_output <- train_elm_for_ga_fitness(xin, yin, Z_final_model, par_bias_input)
  W_final_model <- final_elm_output$w
  
  return(list(W = W_final_model, Z = Z_final_model, p_final = ncol(Z_final_model), 
              g_best = best_chromosome, Z_full = Z_full_internal, 
              best_fitness = best_fitness_value))
}

