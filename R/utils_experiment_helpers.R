run_single_experiment_cv <- function(X_data, Y_data, dataset_name, problem_type,
                                     model_name, train_fn, predict_fn, model_params,
                                     global_cv_params) {
  
  N <- nrow(X_data)
  k_folds <- global_cv_params$k_folds
  
  set.seed(global_cv_params$master_fold_seed) 
  fold_assignments <- caret::createFolds(Y_data, k = k_folds, list = TRUE, returnTrain = FALSE)
  
  fold_results_df <- data.frame(
    fold_id = 1:k_folds,
    metric_train = numeric(k_folds),
    metric_test = numeric(k_folds),
    final_neurons = numeric(k_folds),
    best_fitness_ga = numeric(k_folds) 
  )
  fold_results_df$best_fitness_ga <- NA 
  
  cat(paste0("  Iniciando ", k_folds, "-fold CV para [", model_name, "] em [", dataset_name, "]...\n"))
  
  for (k_idx in 1:k_folds) {
    test_indices <- fold_assignments[[k_idx]]
    train_indices <- setdiff(1:N, test_indices)
    
    xin_cv_train <- X_data[train_indices, , drop = FALSE]
    yin_cv_train <- Y_data[train_indices, , drop = FALSE]
    xin_cv_test <- X_data[test_indices, , drop = FALSE]
    ytest_cv_test <- Y_data[test_indices, , drop = FALSE]
    
    current_model_internal_seed <- global_cv_params$master_fold_seed + k_idx + 
      (model_params$seed_val_offset %||% 1000)
    
    current_model_params_for_fold <- model_params
    if ("seed_val" %in% names(formals(train_fn)) || 
        "seed_val" %in% names(current_model_params_for_fold)) { # Verifica se o modelo aceita/precisa de seed_val
      current_model_params_for_fold$seed_val <- current_model_internal_seed
    }
    current_model_params_for_fold$seed_val_offset <- NULL 
    
    train_call_args <- c(list(xin = xin_cv_train, yin = yin_cv_train), current_model_params_for_fold)
    
    trained_model_output <- do.call(train_fn, train_call_args)
    
    W_fold_model <- trained_model_output$W
    Z_fold_model <- trained_model_output$Z
    p_fold_final <- trained_model_output$p_final
    
    fold_results_df$final_neurons[k_idx] <- ifelse(is.null(p_fold_final), NA, p_fold_final)
    if ("best_fitness" %in% names(trained_model_output)) { # Para GAP-ELM
      fold_results_df$best_fitness_ga[k_idx] <- trained_model_output$best_fitness
    }
    
    if (!is.null(W_fold_model)) {
      par_for_predict <- current_model_params_for_fold$par %||% current_model_params_for_fold$par_bias_input %||% 1
      
      Z_eval <- Z_fold_model
      if( (is.null(p_fold_final) || is.na(p_fold_final) || p_fold_final == 0) && is.null(Z_fold_model)){
        n_feat_pred_fold <- if(par_for_predict == 1) ncol(xin_cv_train) + 1 else ncol(xin_cv_train)
        Z_eval <- matrix(0, nrow = n_feat_pred_fold, ncol = 0)
      }
      
      predict_par_name <- "par" # Default
      if (grepl("OBD", deparse(substitute(predict_fn))) || grepl("GAP", deparse(substitute(predict_fn)))) {
        predict_par_name <- "par_bias_input"
      }
      
      predict_call_args_train <- list(xin = xin_cv_train, Z_final = Z_eval, W_final = W_fold_model)
      predict_call_args_train[[predict_par_name]] <- par_for_predict # Adiciona 'par' ou 'par_bias_input'
      
      predict_call_args_test <- list(xin = xin_cv_test, Z_final = Z_eval, W_final = W_fold_model)
      predict_call_args_test[[predict_par_name]] <- par_for_predict
      
      
      yhat_fold_train <- do.call(predict_fn, predict_call_args_train)
      metrics_fold_train <- calculate_metrics(yin_cv_train, yhat_fold_train, problem_type)
      fold_results_df$metric_train[k_idx] <- metrics_fold_train$main_metric
      
      yhat_fold_test <- do.call(predict_fn, predict_call_args_test)
      metrics_fold_test <- calculate_metrics(ytest_cv_test, yhat_fold_test, problem_type)
      fold_results_df$metric_test[k_idx] <- metrics_fold_test$main_metric
      
      metric_name_for_print <- metrics_fold_test$main_metric_name
    } else {
      # ... (tratamento de erro se W_fold_model for nulo)
      fold_results_df$metric_train[k_idx] <- NA
      fold_results_df$metric_test[k_idx] <- NA
      metric_name_for_print <- "N/A (Erro no Modelo)"
    }
    
    cat(paste0("    Fold ", k_idx, ": ", metric_name_for_print, " Teste = ", ifelse(is.na(fold_results_df$metric_test[k_idx]), "NA", round(fold_results_df$metric_test[k_idx], 4)), 
               ", Neurônios = ", ifelse(is.na(fold_results_df$final_neurons[k_idx]), "NA",fold_results_df$final_neurons[k_idx]), "\n"))
  }
  
  summary_cv_results <- data.frame( # ... (como definido antes) ... )
    Dataset = dataset_name,
    Model = model_name,
    ProblemType = problem_type,
    MetricName = ifelse(exists("metric_name_for_print") && !is.na(fold_results_df$metric_test[1]),
                        metric_name_for_print, 
                        ifelse(problem_type=="classification", "Accuracy", "RMSE")),
    Avg_Metric_Train_CV = mean(fold_results_df$metric_train, na.rm = TRUE),
    Std_Metric_Train_CV = sd(fold_results_df$metric_train, na.rm = TRUE),
    Avg_Metric_Test_CV = mean(fold_results_df$metric_test, na.rm = TRUE),
    Std_Metric_Test_CV = sd(fold_results_df$metric_test, na.rm = TRUE),
    Avg_Final_Neurons_CV = mean(fold_results_df$final_neurons, na.rm = TRUE),
    Std_Final_Neurons_CV = sd(fold_results_df$final_neurons, na.rm = TRUE),
    Avg_Best_Fitness_CV = mean(fold_results_df$best_fitness_ga, na.rm = TRUE) # Será NA se não aplicável
  )
  
  cat(paste0("  CV Concluída para [", model_name, "]. Média Teste ", summary_cv_results$MetricName, ": ", 
             round(summary_cv_results$Avg_Metric_Test_CV, 4), "\n"))
  
  return(list(summary = summary_cv_results, detailed_folds = fold_results_df))
}

`%||%` <- function(a, b) if (!is.null(a)) a else b
