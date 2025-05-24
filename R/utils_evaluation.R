calculate_metrics <- function(y_true, y_pred, problem_type) {
  if (problem_type == "classification") {
    # Assegurar que y_pred seja -1 ou 1 para acurácia
    y_pred_class <- sign(y_pred) 
    accuracy <- mean(y_true == y_pred_class)
    return(list(accuracy = accuracy, main_metric = accuracy, main_metric_name = "Accuracy"))
  } else if (problem_type == "regression") {
    mse <- mean((y_true - y_pred)^2)
    rmse <- sqrt(mse)
    return(list(mse = mse, rmse = rmse, main_metric = rmse, main_metric_name = "RMSE"))
  } else {
    stop("Tipo de problema desconhecido. Use 'classification' ou 'regression'.")
  }
}