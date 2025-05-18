graphics.off()
rm(list = ls())

library("mlbench")
library("corrplot")
library('corpcor')
library(plot3D)

source("D:/github/TP_intermerdi-rio_RNA/R/random_prunned_ELM.R")
source("D:/github/TP_intermerdi-rio_RNA/R/L2_ELM.R")
source("D:/github/TP_intermerdi-rio_RNA/R/std_ELM.R")
source("D:/github/TP_intermerdi-rio_RNA/R/OBD_ELM.R")
source("D:/github/TP_intermerdi-rio_RNA/R/GAP_ELM.R")

# Load statlog (Heart) 
# tratamento
df <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat")
colnames(df) <- c(
  "age", "sex", "chest_pain", "rest_bp", "chol", "fbs", "rest_ecg",
  "max_hr", "ex_angina", "oldpeak", "slope", "ca", "thal", "target"
)
df <- na.omit(df)
df$target <- ifelse(df$target == 1, -1, 1)

df[, 1:14] <- lapply(df[, 1:14], function(x) as.numeric(as.character(x)))

X <- as.matrix(df[, 1:14])
Y <- as.matrix(df$target)
N <- nrow(X)

########################################################################
ntrain <- round(nrow(df) * 0.7) # Usar round para garantir inteiro

reps <- 10
p_initial <- 100 # Número inicial de neurônios ocultos
pruning_rate <- 0.1 # Exemplo: podar 20% dos neurônios. Teste com 0.0 para ELM normal.
par <- 1 

acc_treino <- numeric(reps)
acc_teste <- numeric(reps)
final_p_values <- numeric(reps) # Para registrar o número de neurônios após a poda

# Gerar sementes individuais para cada repetição para a função ELM_pruned
# Isso garante que cada repetição tenha seu próprio Z aleatório e poda, mas o conjunto de repetições é o mesmo.
iteration_seeds <- sample(1:(100*reps), reps)


for (r in 1:reps) {
  current_iter_seed <- iteration_seeds[r]
  set.seed(current_iter_seed) # Semente para a amostragem de dados desta repetição
  
  idx <- sample(N)
  
  train_idx <- idx[1:ntrain]
  test_idx <- idx[(ntrain + 1):N]
  
  xin <- X[train_idx, ]
  yin <- Y[train_idx, , drop = FALSE] # Manter como matriz coluna
  xinteste <- X[test_idx, ]
  yteste <- Y[test_idx, , drop = FALSE] # Manter como matriz coluna
  
  # Treinar ELM com poda
  # A semente dentro de ELM_pruned (se você passar current_iter_seed ou uma nova)
  # controlará a geração de Z e a seleção de neurônios para poda.
  retlist <- ELM_pruned(xin, yin, p_initial, par,
                        pruning_rate = pruning_rate, seed_val = current_iter_seed + 1) # Semente para Z e poda
  
  w_final <- retlist$w
  Z_final <- retlist$Z # Z usado após a poda
  final_p_values[r] <- retlist$p_final
  
  # Avaliar no conjunto de treino
  yhat_train <- YELM_pruned(xin, Z_final, w_final, par)
  
  # Avaliar no conjunto de teste
  yhat_test <- YELM_pruned(xinteste, Z_final, w_final, par)
  
  acc_treino[r] <- mean(yhat_train == yin)
  acc_teste[r] <- mean(yhat_test == yteste)
  
  cat("Rep:", r, "/", reps,
      "| Neurônios Finais:", final_p_values[r],
      "| Acc Treino:", round(acc_treino[r], 4),
      "| Acc Teste:", round(acc_teste[r], 4), "\n")
}

# Resultados finais
cat("\n--- Resultados Finais ---\n")
cat("Taxa de poda alvo:", pruning_rate, "\n")
cat("Número inicial de neurônios:", p_initial, "\n")
cat("Número médio de neurônios após poda:", round(mean(final_p_values)),
    "±", round(sd(final_p_values)), "\n")

cat("\nAcurácia média (treino):", round(mean(acc_treino), 4),
    "±", round(sd(acc_treino), 4), "\n")
cat("Acurácia média (teste):", round(mean(acc_teste), 4),
    "±", round(sd(acc_teste), 4), "\n")


cat("\n\n############################# INICIANDO TESTES OBD-ELM ###########################\n")

obd_max_iterations <- 10 
min_neurons <- 5 

# Use nomes de variáveis distintos para os resultados do OBD se for comparar no mesmo script
acc_treino_obd_all_reps <- numeric(reps)
acc_teste_obd_all_reps <- numeric(reps)
final_p_values_obd_all_reps <- numeric(reps)

# iteration_seeds já foi definido e usado para ELM_pruned. Vamos reutilizá-lo.

for (r in 1:reps) {
  current_iter_seed <- iteration_seeds[r]
  set.seed(current_iter_seed) # Semente para a amostragem de dados desta repetição
  
  idx <- sample(N)
  train_idx <- idx[1:ntrain]
  test_idx <- idx[(ntrain + 1):N]
  
  # Dados para esta repetição r
  xin_r <- X[train_idx, ]
  yin_r <- Y[train_idx, , drop = FALSE]
  xinteste_r <- X[test_idx, ]
  yteste_r <- Y[test_idx, , drop = FALSE]
  
  cat("\n--- Repetição OBD:", r, "/", reps, "---\n")
  
  # --- Início do Processo OBD para esta repetição r ---
  # (Seu bloco de código OBD existente vai aqui, adaptado para usar xin_r, yin_r)
  
  # Inicialização de Z para OBD (dentro da repetição r)
  n_features_xin_r <- ncol(xin_r)
  # Usar uma semente diferente para Z_full para cada repetição, mas consistente entre execuções
  set.seed(current_iter_seed + reps + 1) # Adiciona um offset à semente da iteração
  
  if (par == 1) { # Usando par_bias_input_val que você já tem
    Z_full_r <- matrix(runif((n_features_xin_r + 1) * p_initial, -0.5, 0.5), 
                       nrow = (n_features_xin_r + 1), ncol = p_initial)
  } else {
    Z_full_r <- matrix(runif(n_features_xin_r * p_initial, -0.5, 0.5), 
                       nrow = n_features_xin_r, ncol = p_initial)
  }
  
  active_neuron_indices_in_Z_full_r <- 1:p_initial
  Z_current_pruned_r <- Z_full_r
  W_final_model_r <- NULL 
  
  cat("  Iniciando loop de poda OBD para repetição", r, "(P inicial =", p_initial, ")\n")
  
  for (iter_obd_loop in 1:obd_max_iterations) { # Usando obd_max_iterations que você já tem
    num_active_neurons_r <- length(active_neuron_indices_in_Z_full_r)
    # Opcional: cat(" Iteração de Poda Interna:", iter_obd_loop, "| Neurônios Ativos:", num_active_neurons_r, "\n")
    
    if (num_active_neurons_r < min_neurons || num_active_neurons_r == 0) {
      break
    }
    
    # Assumindo que OBD_ELM é sua função de treino do passo, e calculate_saliencies_for_w está disponível
    # (carregadas pelos seus `source` no início do script)
    train_result_r <- OBD_ELM(xin_r, yin_r, Z_current_pruned_r, par)
    w_current_iter_r <- train_result_r$w
    Haug_current_iter_r <- train_result_r$Haug
    W_final_model_r <- w_current_iter_r # Salva o modelo mais recente
    
    saliencies_for_w_r <- calculate_saliencies_for_w(Haug_current_iter_r, w_current_iter_r)
    
    if (length(saliencies_for_w_r) <= 1 && num_active_neurons_r > 0) { break }
    if (length(saliencies_for_w_r) <= 1 && num_active_neurons_r == 0) { break }
    
    neuron_saliencies_values_r <- saliencies_for_w_r[-1]
    
    if (length(neuron_saliencies_values_r) == 0) { break }
    
    num_to_prune_this_iter_r <- ceiling(pruning_rate * length(neuron_saliencies_values_r))
    
    if (length(neuron_saliencies_values_r) - num_to_prune_this_iter_r < min_neurons && length(neuron_saliencies_values_r) > min_neurons) {
      num_to_prune_this_iter_r <- length(neuron_saliencies_values_r) - min_neurons
    } else if (length(neuron_saliencies_values_r) <= min_neurons) {
      num_to_prune_this_iter_r <- 0 
    }
    
    if (num_to_prune_this_iter_r <= 0) {
      if(num_active_neurons_r <= min_neurons) break; 
      next 
    }
    
    indices_to_prune_local_r <- order(neuron_saliencies_values_r)[1:num_to_prune_this_iter_r]
    indices_in_Z_full_to_remove_r <- active_neuron_indices_in_Z_full_r[indices_to_prune_local_r]
    active_neuron_indices_in_Z_full_r <- setdiff(active_neuron_indices_in_Z_full_r, indices_in_Z_full_to_remove_r)
    
    Z_current_pruned_r <- Z_full_r[, active_neuron_indices_in_Z_full_r, drop = FALSE]
  } 
  
  cat("  Processo OBD para repetição", r, "concluído. Retreinando modelo final OBD...\n")
  final_train_result_r <- OBD_ELM(xin_r, yin_r, Z_current_pruned_r, par)
  W_final_model_r <- final_train_result_r$w
  
  if (!is.null(W_final_model_r)) {
    yhat_train_obd_r <- YELM_OBD(xin_r, Z_current_pruned_r, W_final_model_r, par)
    acc_treino_obd_all_reps[r] <- mean(yhat_train_obd_r == yin_r)
    
    yhat_test_obd_r <- YELM_pruned(xinteste_r, Z_current_pruned_r, W_final_model_r, par)
    acc_teste_obd_all_reps[r] <- mean(yhat_test_obd_r == yteste_r)
    
    final_p_values_obd_all_reps[r] <- if (is.null(Z_current_pruned_r) || ncol(Z_current_pruned_r) == 0) 0 else ncol(Z_current_pruned_r)
    
    cat("  Rep", r, "OBD Final: Neurônios:", final_p_values_obd_all_reps[r],
        "| Acc Treino:", round(acc_treino_obd_all_reps[r], 4),
        "| Acc Teste:", round(acc_teste_obd_all_reps[r], 4), "\n")
  } else {
    cat("  Rep", r, "OBD Final: Modelo final nulo.\n")
    acc_treino_obd_all_reps[r] <- NA
    acc_teste_obd_all_reps[r] <- NA
    final_p_values_obd_all_reps[r] <- 0
  }
} # Fim do loop de repetições (r) para OBD

# Resultados Finais Agregados do OBD-ELM
cat("\n\n--- Resultados Finais Agregados OBD-ELM ---\n")
# Use os mesmos nomes de parâmetros que você definiu no início da seção OBD
cat("Configurações: P inicial =", p_initial, # ou p_initial_obd se você usou um nome diferente
    ", Fração Poda/Iter =", pruning_rate, 
    ", Iterações OBD =", obd_max_iterations, 
    ", Neurônios Mínimos =", min_neurons, "\n")

cat("Número médio de neurônios após poda OBD:", round(mean(final_p_values_obd_all_reps, na.rm=TRUE)),
    "±", round(sd(final_p_values_obd_all_reps, na.rm=TRUE)), "\n")

cat("\nAcurácia média OBD (treino):", round(mean(acc_treino_obd_all_reps, na.rm=TRUE), 4),
    "±", round(sd(acc_treino_obd_all_reps, na.rm=TRUE), 4), "\n")
cat("Acurácia média OBD (teste):", round(mean(acc_teste_obd_all_reps, na.rm=TRUE), 4),
    "±", round(sd(acc_teste_obd_all_reps, na.rm=TRUE), 4), "\n")

