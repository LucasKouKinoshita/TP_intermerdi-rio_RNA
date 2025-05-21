graphics.off() 
rm(list = ls()) 
library("mlbench")
library("corrplot") 
library('corpcor') 
library('GA')      
library("corpcor")

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

ELM_GAP <- function(xin_train, yin_train, p_initial, par_bias_input,
                    # Parâmetros do Algoritmo Genético
                    ga_pop_size = 50, 
                    ga_max_iter = 30, # Artigo menciona 300, mas pode ser custoso
                    ga_pmutation = 0.1, 
                    ga_pcrossover = 0.8,
                    ga_elitism_fraction = 0.1, # Fração para elitismo
                    alpha_fitness = 0.99, # Padrão do artigo para bons resultados
                    use_press_loo = FALSE, # Mudar para TRUE para tentar implementar PRESS
                    seed_val = NULL) {
  
  if (!is.null(seed_val)) {
    set.seed(seed_val)
  }
  
  n_features <- ncol(xin_train)
  n_samples_train <- nrow(yin_train)
  
  #  Gerar Z_full (matriz base de pesos da camada oculta)
  cat("Gerando Z_full com", p_initial, "neurônios iniciais...\n")
  if (par_bias_input == 1) {
    Z_full_internal <- matrix(runif((n_features + 1) * p_initial, -0.5, 0.5), nrow = (n_features + 1), ncol = p_initial)
  } else {
    Z_full_internal <- matrix(runif(n_features * p_initial, -0.5, 0.5), nrow = n_features, ncol = p_initial)
  }
  
  # Fitness function
  fitness_fn_gap <- function(chromosome_g, Z_full_ref, xin_fit, yin_fit, par_bias_fit, alpha_fit, use_press_fit, p_initial_ref) {
    # chromosome_g é um vetor binário fornecido pelo pacote GA
    chromosome_g <- as.integer(round(chromosome_g)) # Garantir 0s e 1s
    selected_neuron_indices <- which(chromosome_g == 1)
    
    num_selected_neurons <- length(selected_neuron_indices)
    
    if (num_selected_neurons == 0) {
      # Penalidade se nenhum neurônio for selecionado (fitness a ser minimizada)
      # Erro máximo (1.0) e taxa de neurônios zero.
      return(alpha_fit * 1.0 + (1 - alpha_fit) * 0.0) 
    }
    
    Z_subset_fit <- Z_full_ref[, selected_neuron_indices, drop = FALSE]
    
    # Treinar ELM base e obter predições no treino
    elm_res <- train_elm_for_ga_fitness(xin_fit, yin_fit, Z_subset_fit, par_bias_fit)
    y_pred_fit <- elm_res$y_pred # Predições no conjunto de treino
    
    # Calcular error_rate(g)
    if (use_press_fit) {
      error_rate_val <- 1 - mean(sign(y_pred_fit) == yin_fit) # Simplificação: erro de treino
      if(is.na(error_rate_val)) error_rate_val <- 1.0 # Caso de y_pred_fit ser problemático
    } else { # Usar erro de treino normal
      error_rate_val <- 1 - mean(sign(y_pred_fit) == yin_fit)
      if(is.na(error_rate_val)) error_rate_val <- 1.0
    }
    
    neuron_rate_val <- num_selected_neurons / p_initial_ref
    
    fitness_value <- alpha_fit * error_rate_val + (1 - alpha_fit) * neuron_rate_val
    return(fitness_value)
  }
  
  cat("Iniciando Algoritmo Genético...\n")
  # ga_ctrl <- gaControl(seed = seed_val) # Para reprodutibilidade do AG
  
  ga_optim_results <- ga(type = "binary", 
                         fitness = fitness_fn_gap,
                         # Passando argumentos adicionais para fitness_fn_gap
                         Z_full_ref = Z_full_internal, 
                         xin_fit = xin_train, 
                         yin_fit = yin_train, 
                         par_bias_fit = par_bias_input, 
                         alpha_fit = alpha_fitness,
                         use_press_fit = use_press_loo,
                         p_initial_ref = p_initial,
                         # Parâmetros do AG
                         nBits = p_initial,
                         popSize = ga_pop_size,
                         maxiter = ga_max_iter,
                         pcrossover = ga_pcrossover,
                         pmutation = ga_pmutation,
                         elitism = max(1, round(ga_elitism_fraction * ga_pop_size)),
                         monitor = FALSE, # Mudar para TRUE ou gaMonitor para ver progresso
                         #control = ga_ctrl
                         seed = seed_val
                         )
  
  best_chromosome <- as.integer(ga_optim_results@solution[1,])
  best_fitness_value <- ga_optim_results@fitnessValue
  
  cat("AG Concluído. Melhor fitness:", best_fitness_value, "\n")
  
  final_selected_indices <- which(best_chromosome == 1)
  
  if (length(final_selected_indices) == 0) {
    warning("GAP-ELM: Algoritmo Genético não selecionou nenhum neurônio. Verifique os parâmetros do AG e da função de fitness.")
    Z_final_model <- Z_full_internal[, c(), drop = FALSE] # Z vazio
  } else {
    Z_final_model <- Z_full_internal[, final_selected_indices, drop = FALSE]
  }
  
  cat("Treinando modelo final ELM com", ncol(Z_final_model), "neurônios selecionados pelo GAP...\n")
  final_elm_output <- train_elm_for_ga_fitness(xin_train, yin_train, Z_final_model, par_bias_input)
  W_final_model <- final_elm_output$w
  
  return(list(W = W_final_model, 
              Z = Z_final_model, 
              p_final = ncol(Z_final_model), 
              g_best = best_chromosome, 
              Z_full = Z_full_internal, 
              best_fitness = best_fitness_value))
}

# Load statlog (Heart)
# tratamento
df <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat")
colnames(df) <- c(
  "age", "sex", "chest_pain", "rest_bp", "chol", "fbs", "rest_ecg",
  "max_hr", "ex_angina", "oldpeak", "slope", "ca", "thal", "target"
)
df <- na.omit(df)
df$target <- ifelse(df$target == 1, -1, 1)

df[, 1:(ncol(df)-1)] <- lapply(df[, 1:(ncol(df)-1)], function(x) as.numeric(as.character(x))) # Ajustado para ncol(df)-1

X <- as.matrix(df[, 1:(ncol(df)-1)]) # Ajustado
Y <- as.matrix(df$target)
N <- nrow(X)

########################################################################
# Parâmetros Globais para o Teste
ntrain <- round(nrow(df) * 0.7)
reps <- 10 # Mantenha baixo para testes iniciais do GAP-ELM, pois pode ser lento

# Parâmetros para GAP-ELM
p_initial_gap <- 100      # Número inicial de neurônios para Z_full na GAP-ELM
par_gap <- 1               # Parâmetro 'par' (bias de entrada) para ELM base e predição
# Parâmetros do Algoritmo Genético (exemplos, ajuste conforme o artigo ou experimentação)
ga_pop_size_val <- 30     # Tamanho da população do AG (artigo usa 50)
ga_max_iter_val <- 50     # Número de gerações do AG (artigo usa 300, mas 50 para teste rápido)
ga_pmutation_val <- 0.1
ga_pcrossover_val <- 0.8
ga_elitism_fraction_val <- 0.1
alpha_fitness_val <- 0.99 # Ponderador da função de fitness [cite: 105, 110]
use_press_val <- FALSE    # Mude para TRUE se sua ELM_GAP tiver implementação robusta de PRESS LOO

# Vetores para armazenar resultados da GAP-ELM
acc_treino_gap_reps <- numeric(reps)
acc_teste_gap_reps <- numeric(reps)
final_p_values_gap_reps <- numeric(reps)
best_fitness_values_reps <- numeric(reps) # Para armazenar o melhor fitness de cada repetição

# Gerar sementes individuais para cada repetição (reutilizável)
master_seed_for_script <- 123 # Semente mestre para o script
set.seed(master_seed_for_script)
iteration_seeds <- sample(1:(1000 * reps), reps)

cat("\n\n############################# INICIANDO TESTES GAP-ELM ###########################\n")
cat("Configurações GAP-ELM: P inicial =", p_initial_gap,
    ", Pop AG =", ga_pop_size_val,
    ", Gerações AG =", ga_max_iter_val,
    ", Alpha Fitness =", alpha_fitness_val, "\n")

for (r in 1:reps) {
  current_loop_seed <- iteration_seeds[r]
  set.seed(current_loop_seed) # Semente para a amostragem de dados desta repetição
  
  idx <- sample(N)
  
  train_idx <- idx[1:ntrain]
  test_idx <- idx[(ntrain + 1):N]
  
  xin_loop <- X[train_idx, ]
  yin_loop <- Y[train_idx, , drop = FALSE]
  xinteste_loop <- X[test_idx, ]
  yteste_loop <- Y[test_idx, , drop = FALSE]
  
  cat("\n--- Repetição GAP-ELM:", r, "/", reps, "---\n")
  
  # Treinar com GAP-ELM
  # A semente para o AG e para a Z_full inicial dentro de ELM_GAP deve ser gerenciada
  # Se ELM_GAP usa o parâmetro seed_val, passe uma semente derivada.
  model_gap_info <- ELM_GAP(xin_train = xin_loop, 
                            yin_train = yin_loop, 
                            p_initial = p_initial_gap, 
                            par_bias_input = par_gap,
                            ga_pop_size = ga_pop_size_val, 
                            ga_max_iter = ga_max_iter_val, 
                            ga_pmutation = ga_pmutation_val,
                            ga_pcrossover = ga_pcrossover_val,
                            ga_elitism_fraction = ga_elitism_fraction_val,
                            alpha_fitness = alpha_fitness_val,
                            use_press_loo = use_press_val, 
                            seed_val = current_loop_seed + reps + 1) # Semente para o processo do AG e Z_full interno
  
  W_final_gap <- model_gap_info$W
  Z_final_gap <- model_gap_info$Z
  p_final_gap_val <- model_gap_info$p_final
  best_fitness_values_reps[r] <- model_gap_info$best_fitness # Assumindo que ELM_GAP retorna isso
  
  final_p_values_gap_reps[r] <- p_final_gap_val
  
  # Avaliar no conjunto de treino e teste
  if (!is.null(W_final_gap) && (!is.null(Z_final_gap) || p_final_gap_val == 0) ) { # Z pode ser nulo se p_final_gap_val = 0
    
    # Se Z_final_gap for NULL e p_final_gap_val for 0, crie uma matriz Z vazia compatível
    if(is.null(Z_final_gap) && p_final_gap_val == 0) {
      n_features_loop <- if(par_gap == 1) ncol(xin_loop) + 1 else ncol(xin_loop)
      Z_final_gap_eval <- matrix(0, nrow = n_features_loop, ncol = 0)
    } else {
      Z_final_gap_eval <- Z_final_gap
    }
    
    yhat_train_current_rep <- YELM_GAP_predict(xin_loop, Z_final_gap_eval, W_final_gap, par_gap)
    acc_treino_gap_reps[r] <- mean(yhat_train_current_rep == yin_loop)
    
    yhat_test_current_rep <- YELM_GAP_predict(xinteste_loop, Z_final_gap_eval, W_final_gap, par_gap)
    acc_teste_gap_reps[r] <- mean(yhat_test_current_rep == yteste_loop)
  } else {
    cat("  Aviso: Modelo GAP-ELM inválido ou não retornou Z/W para repetição", r, "\n")
    acc_treino_gap_reps[r] <- NA # Ou 0
    acc_teste_gap_reps[r] <- NA  # Ou 0
    # final_p_values_gap_reps[r] já seria 0 ou o valor de p_final_gap_val
  }
  
  cat("  Rep:", r, "/", reps, "GAP-ELM Concluído.",
      "| Neurônios Finais:", final_p_values_gap_reps[r],
      "| Melhor Fitness AG:", ifelse(!is.na(best_fitness_values_reps[r]), round(best_fitness_values_reps[r], 5), "N/A"),
      "| Acc Treino:", ifelse(!is.na(acc_treino_gap_reps[r]), round(acc_treino_gap_reps[r], 4), "N/A"),
      "| Acc Teste:", ifelse(!is.na(acc_teste_gap_reps[r]), round(acc_teste_gap_reps[r], 4), "N/A"), "\n")
}

# Resultados finais da GAP-ELM
cat("\n\n--- Resultados Finais Agregados GAP-ELM ---\n")
cat("Número inicial de neurônios (base para AG):", p_initial_gap, "\n")
cat("Parâmetros AG: PopSize=", ga_pop_size_val, ", MaxIter=", ga_max_iter_val, ", AlphaFitness=", alpha_fitness_val, "\n")

cat("Número médio de neurônios após poda GAP-ELM:", round(mean(final_p_values_gap_reps, na.rm=TRUE)),
    "±", round(sd(final_p_values_gap_reps, na.rm=TRUE)), "\n")

cat("Melhor fitness médio (AG) ao longo das repetições:", round(mean(best_fitness_values_reps, na.rm=TRUE), 5),
    "±", round(sd(best_fitness_values_reps, na.rm=TRUE), 5), "\n")

cat("\nAcurácia média GAP-ELM (treino):", round(mean(acc_treino_gap_reps, na.rm=TRUE), 4),
    "±", round(sd(acc_treino_gap_reps, na.rm=TRUE), 4), "\n")
cat("Acurácia média GAP-ELM (teste):", round(mean(acc_teste_gap_reps, na.rm=TRUE), 4),
    "±", round(sd(acc_teste_gap_reps, na.rm=TRUE), 4), "\n")