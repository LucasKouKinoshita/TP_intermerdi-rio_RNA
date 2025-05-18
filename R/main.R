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
pruning_rate_value <- 0.2 # Exemplo: podar 20% dos neurônios. Teste com 0.0 para ELM normal.

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
  retlist <- ELM_pruned(xin, yin, p_initial, par = 1,
                        pruning_rate = pruning_rate_value, seed_val = current_iter_seed + 1) # Semente para Z e poda
  
  w_final <- retlist$w
  Z_final <- retlist$Z # Z usado após a poda
  final_p_values[r] <- retlist$p_final
  
  # Avaliar no conjunto de treino
  yhat_train <- YELM_pruned(xin, Z_final, w_final, par = 1)
  
  # Avaliar no conjunto de teste
  yhat_test <- YELM_pruned(xinteste, Z_final, w_final, par = 1)
  
  acc_treino[r] <- mean(yhat_train == yin)
  acc_teste[r] <- mean(yhat_test == yteste)
  
  cat("Rep:", r, "/", reps,
      "| Neurônios Finais:", final_p_values[r],
      "| Acc Treino:", round(acc_treino[r], 4),
      "| Acc Teste:", round(acc_teste[r], 4), "\n")
}

# Resultados finais
cat("\n--- Resultados Finais ---\n")
cat("Taxa de poda alvo:", pruning_rate_value, "\n")
cat("Número inicial de neurônios:", p_initial, "\n")
cat("Número médio de neurônios após poda:", round(mean(final_p_values)),
    "±", round(sd(final_p_values)), "\n")

cat("\nAcurácia média (treino):", round(mean(acc_treino), 4),
    "±", round(sd(acc_treino), 4), "\n")
cat("Acurácia média (teste):", round(mean(acc_teste), 4),
    "±", round(sd(acc_teste), 4), "\n")


############################# ODB ###########################

p_initial <- 100          # Número inicial de neurônios ocultos
par_bias_input_val <- 1  # Se a camada de entrada tem uma coluna de bias para Z
pruning_fraction <- 0.2  # Fração dos neurônios *restantes* a podar em cada iteração
obd_max_iterations <- 10 # Máximo de iterações de poda
min_neurons <- 5         # Mínimo de neurônios ocultos a manter

# Inicialização de Z
n_features_xin <- ncol(xin)
if (par_bias_input_val == 1) {
  Z_full <- matrix(runif((n_features_xin + 1) * p_initial, -0.5, 0.5), nrow = (n_features_xin + 1), ncol = p_initial)
} else {
  Z_full <- matrix(runif(n_features_xin * p_initial, -0.5, 0.5), nrow = n_features_xin, ncol = p_initial)
}

active_neuron_indices_in_Z_full <- 1:p_initial
Z_current_pruned <- Z_full
W_final_model <- NULL 

cat("Iniciando OBD para ELM (sem regularização L2)...\n")
cat("Configurações: P inicial =", p_initial, "Fração Poda =", pruning_fraction, "Min Neurônios =", min_neurons, "\n")

for (iter_obd in 1:obd_max_iterations) {
  num_active_neurons <- length(active_neuron_indices_in_Z_full)
  cat("\nIteração OBD:", iter_obd, "| Neurônios Ativos:", num_active_neurons, "\n")
  
  if (num_active_neurons < min_neurons || num_active_neurons == 0) {
    cat("  Número de neurônios atingiu o limite mínimo ou zero. Parando a poda.\n")
    break
  }
  
  # Passo 2 do OBD (Adaptado): Treinar ELM com neurônios ativos (SEM L2)
  train_result <- OBD_ELM(xin, yin, Z_current_pruned, par_bias_input_val)
  w_current_iter <- train_result$w
  Haug_current_iter <- train_result$Haug
  W_final_model <- w_current_iter
  
  # (Opcional) Avaliar desempenho no treino/teste
  # yhat_train_iter <- YELM_predict(xin, Z_current_pruned, w_current_iter, par_bias_input_val)
  # acc_train_iter <- mean(yhat_train_iter == yin_train)
  # cat("    Acc Treino:", round(acc_train_iter, 4), "\n")
  
  # Passo 3 e 4 do OBD (Adaptado): Calcular Saliências
  saliencies_for_w <- calculate_saliencies_for_w(Haug_current_iter, w_current_iter)
  
  if (length(saliencies_for_w) <= 1 && num_active_neurons > 0) { 
    cat("  Problema ao calcular saliências ou apenas bias de saída. Parando.\n")
    break
  }
  if (length(saliencies_for_w) <=1 && num_active_neurons == 0){
    cat("  Nenhum neurônio para podar e apenas bias de saída.\n")
    break
  }
  
  neuron_saliencies_values <- saliencies_for_w[-1] # Ignora saliência do bias de saída
  
  if (length(neuron_saliencies_values) == 0) {
    cat("  Nenhum neurônio oculto para podar.\n")
    break
  }
  
  # Passo 5 do OBD (Adaptado): Ordenar e decidir quais neurônios podar
  num_to_prune_this_iter <- ceiling(pruning_fraction * length(neuron_saliencies_values))
  
  if (length(neuron_saliencies_values) - num_to_prune_this_iter < min_neurons && length(neuron_saliencies_values) > min_neurons) {
    num_to_prune_this_iter <- length(neuron_saliencies_values) - min_neurons
  } else if (length(neuron_saliencies_values) <= min_neurons) {
    num_to_prune_this_iter <- 0 
  }
  
  if (num_to_prune_this_iter <= 0) {
    cat("  Nenhum neurônio a ser podado nesta iteração ou limite mínimo atingido.\n")
    if (iter_obd == obd_max_iterations) cat("Poda finalizada após máximo de iterações.\n")
    if(num_active_neurons <= min_neurons) break; 
    next 
  }
  
  cat("    Podando", num_to_prune_this_iter, "neurônios de menor saliência.\n")
  indices_to_prune_local <- order(neuron_saliencies_values)[1:num_to_prune_this_iter]
  indices_in_Z_full_to_remove <- active_neuron_indices_in_Z_full[indices_to_prune_local]
  active_neuron_indices_in_Z_full <- setdiff(active_neuron_indices_in_Z_full, indices_in_Z_full_to_remove)
  
  Z_current_pruned <- Z_full[, active_neuron_indices_in_Z_full, drop = FALSE]
  
  if (iter_obd == obd_max_iterations) {
    cat("Poda finalizada após máximo de iterações. Retreinando modelo final...\n")
    train_result_final <- OBD_ELM(xin, yin, Z_current_pruned, par_bias_input_val)
    W_final_model <- train_result_final$w
    cat("Modelo final treinado com", length(active_neuron_indices_in_Z_full), "neurônios.\n")
  }
}
