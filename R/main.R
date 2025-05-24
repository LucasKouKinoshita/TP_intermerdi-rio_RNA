graphics.off()
rm(list = ls())

library("corpcor") 
library("GA") 
library("caret")
####### AYO, QM ESTIVER BAIXANDO O CÓDIGO DO REPOSITÓRIO, MUDAR OS PATHS
####### PARA OS VALORES APROPRIADOS, POR ALGUMA RAZÃO SÓ 
####### 'R/file.R' NAO ESTAVA FUNCIONANDO


# modelos
model_files <- list.files("D:/github/TP_intermerdi-rio_RNA/R/", pattern = "^(random_prunned_ELM|L2_ELM|std_ELM|OBD_ELM|GAP_ELM)\\.R$", full.names = TRUE)
sapply(model_files, source)

# Utils 
source("D:/github/TP_intermerdi-rio_RNA/R/utils_data_preprocessing.R")
source("D:/github/TP_intermerdi-rio_RNA/R/utils_evaluation.R")
source("D:/github/TP_intermerdi-rio_RNA/R/utils_experiment_helpers.R")


# parametros globais
global_cv_config <- list(
  k_folds = 10,
  master_fold_seed = 2024 # Semente mestre para gerar as 'iteration_seeds'
)

# Datasets
# Cada item é uma função que, quando chamada, retorna list(X, Y, name, type)
datasets_config <- list(
  function() load_and_preprocess_heart(),
  function() load_and_preprocess_breast_cancer(),
  function() load_and_preprocess_iris(binary_class_target = "setosa"), # Exemplo com um parâmetro
  function() generate_polynomial_data(degree = 2), # Aproximação quadrática
  function() generate_polynomial_data(degree = 0)  # Aproximação sinusoidal (degree=0 no meu exemplo é sin)
)

# 4. Lista de Modelos e Seus Parâmetros Específicos
# Os nomes das funções ('train_fn_name', 'predict_fn_name') devem corresponder
# exatamente aos nomes das funções definidas/carregadas.
models_config <- list(
  list(name = "StdELM",
       train_fn_name = "ELM",       # Sua função ELM(xin, yin, p, par)
       predict_fn_name = "YELM",    # Sua função YELM(xin, Z, W, par)
       params = list(p = 150, par = 1) # Estes nomes (p, par) devem bater com ELM
  ),
  list(name = "RandomPrunedELM",
       train_fn_name = "ELM_pruned", 
       predict_fn_name = "YELM_pruned", 
       params = list(p_initial = 150, par = 1, pruning_rate = 0.2, seed_val_offset = 100) 
  ),
  list(name = "L2_ELM",
       train_fn_name = "ELM_L2", # Nome da sua função
       predict_fn_name = "YELM_L2", # Nome da sua função
       params = list(p = 150, par = 1, L = 0.01) # 'L' para lambda
  ),
  list(name = "OBD_ELM",
       train_fn_name = "run_obd_elm_process", # <-- MUDE AQUI
       predict_fn_name = "YELM_OBD", 
       params = list(p_initial = 100, par_bias_input = 1, pruning_fraction = 0.1, 
                     obd_max_iterations = 10, min_neurons = 5, 
                     seed_val_offset = 300) # Adicionado seed_val_offset
  ),
  list(name = "GAP_ELM",
       train_fn_name = "ELM_GAP", 
       predict_fn_name = "YELM_GAP_predict", 
       params = list(p_initial = 50, # Reduzido para GAP ser mais rápido em testes
                     par_bias_input = 1, 
                     ga_pop_size = 20, # Reduzido 
                     ga_max_iter = 30, # Reduzido
                     alpha_fitness = 0.99,
                     ga_pmutation = 0.1, ga_pcrossover = 0.8, ga_elitism_fraction = 0.1,
                     use_press_loo = FALSE,
                     seed_val_offset = 200) # Offset para a semente interna do AG
  )
)
# Certifique-se que as funções de treino retornam uma lista com W, Z, p_final
# e para GAP_ELM, opcionalmente, best_fitness.
# As funções de predição devem aceitar (xin, Z_final, W_final, par)

all_cv_summaries <- list()
all_cv_detailed_fold_results <- list()

for (i in 1:length(datasets_config)) {
  dataset_loader_fn <- datasets_config[[i]]
  current_data_list <- dataset_loader_fn() 
  
  cat(paste0("\n\n================== DATASET: ", current_data_list$name, " ==================\n"))
  
  for (j in 1:length(models_config)) {
    model_setup <- models_config[[j]]
    
    # Verifica se as funções existem (essencial para depuração)
    if (!exists(model_setup$train_fn_name, mode = "function")) {
      stop(paste("Função de treino '", model_setup$train_fn_name, "' não encontrada para o modelo '", model_setup$name, "'."))
    }
    if (!exists(model_setup$predict_fn_name, mode = "function")) {
      stop(paste("Função de predição '", model_setup$predict_fn_name, "' não encontrada para o modelo '", model_setup$name, "'."))
    }
    
    # Chamar a função run_single_experiment_cv (definida em utils_experiment_helpers.R)
    experiment_cv_result <- run_single_experiment_cv(
      X_data = current_data_list$X,
      Y_data = current_data_list$Y,
      dataset_name = current_data_list$name,
      problem_type = current_data_list$type,
      model_name = model_setup$name,
      train_fn = get(model_setup$train_fn_name), 
      predict_fn = get(model_setup$predict_fn_name), 
      model_params = model_setup$params, # Parâmetros específicos do modelo
      global_cv_params = global_cv_config # Parâmetros globais da CV (k_folds, master_fold_seed)
    )
    
    summary_key <- paste(current_data_list$name, model_setup$name, sep = "_")
    all_cv_summaries[[summary_key]] <- experiment_cv_result$summary
    all_cv_detailed_fold_results[[summary_key]] <- experiment_cv_result$detailed_folds
  }
}


cat("\n\n================== RESUMO FINAL DOS EXPERIMENTOS (", global_cv_config$k_folds, "-Fold CV) ==================\n")
summary_cv_table <- do.call(rbind, all_cv_summaries)
print(summary_cv_table)

results_dir_cv <- "results_cv"
if (!dir.exists(results_dir_cv)) dir.create(results_dir_cv)
save(all_cv_summaries, all_cv_detailed_fold_results, global_cv_config, 
     file = file.path(results_dir_cv, "all_cv_experiment_results.RData"))
write.csv(summary_cv_table, file.path(results_dir_cv, "summary_table_cv_experiments.csv"), row.names = FALSE)

cat("\nResultados da Validação Cruzada salvos no diretório:", results_dir_cv, "\n")