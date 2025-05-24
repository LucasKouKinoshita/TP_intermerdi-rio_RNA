# Função para carregar e pré-processar Statlog Heart
load_and_preprocess_heart <- function() {
  df <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat")
  colnames(df) <- c("age", "sex", "chest_pain", "rest_bp", "chol", "fbs", "rest_ecg",
                    "max_hr", "ex_angina", "oldpeak", "slope", "ca", "thal", "target")
  df <- na.omit(df)
  df$target <- ifelse(df$target == 1, -1, 1) # Mapeia 1->-1, 2->1
  
  # Converte todas as colunas (exceto target) para numérico
  feature_cols <- setdiff(colnames(df), "target")
  df[, feature_cols] <- lapply(df[, feature_cols], function(x) as.numeric(as.character(x)))
  
  X <- as.matrix(df[, feature_cols])
  Y <- as.matrix(df$target)
  
  # Opcional: Normalização/Padronização das features X
  # X <- scale(X)
  
  return(list(X = X, Y = Y, name = "StatlogHeart", type = "classification"))
}

# Função para carregar e pré-processar Breast Cancer Wisconsin (Original)
load_and_preprocess_breast_cancer <- function() {
  if (!requireNamespace("mlbench", quietly = TRUE)) install.packages("mlbench")
  data(BreastCancer, package = "mlbench")
  df <- BreastCancer
  df$Id <- NULL # Remover coluna de ID
  df <- na.omit(df) # Remover linhas com NA
  
  # Converter classes para -1 e 1
  df$Class <- ifelse(df$Class == "benign", -1, 1)
  
  feature_cols <- setdiff(colnames(df), "Class")
  # Converter todas as features para numérico (já são fatores, precisam de as.character primeiro)
  for (col in feature_cols) {
    df[[col]] <- as.numeric(as.character(df[[col]]))
  }
  
  X <- as.matrix(df[, feature_cols])
  Y <- as.matrix(df$Class)
  
  # X <- scale(X) # Opcional
  
  return(list(X = X, Y = Y, name = "BreastCancer", type = "classification"))
}

# Função para carregar e pré-processar Iris
load_and_preprocess_iris <- function(binary_class_target = "setosa") {
  data(iris)
  df <- iris
  
  # Para ELM binária, converter para problema binário.
  # Ex: 'setosa' vs 'não-setosa'
  if (binary_class_target == "setosa") {
    df$Species <- ifelse(df$Species == "setosa", 1, -1)
    name <- "Iris (Setosa vs Rest)"
  } else if (binary_class_target == "versicolor") {
    df$Species <- ifelse(df$Species == "versicolor", 1, -1)
    name <- "Iris (Versicolor vs Rest)"
  } else { # virginica
    df$Species <- ifelse(df$Species == "virginica", 1, -1)
    name <- "Iris (Virginica vs Rest)"
  }
  # Nota: Para uma análise completa do Iris, você precisaria de uma abordagem multiclasse
  # ou treinar 3 classificadores binários.
  
  feature_cols <- colnames(df)[1:4] # Sepal.Length, Sepal.Width, Petal.Length, Petal.Width
  X <- as.matrix(df[, feature_cols])
  Y <- as.matrix(df$Species)
  
  # X <- scale(X) # Opcional
  
  return(list(X = X, Y = Y, name = name, type = "classification"))
}

# Função para gerar dados de aproximação polinomial
generate_polynomial_data <- function(n_samples = 200, degree = 2, noise_sd = 0.5, x_range = c(-5, 5)) {
  set.seed(42) # Para reprodutibilidade
  x <- runif(n_samples, min = x_range[1], max = x_range[2])
  
  # Exemplo de polinômio: a*x^2 + b*x + c (ou mais complexo)
  # Para este exemplo, vamos usar y = x^2 - 2x + 1 + noise
  if (degree == 1) { # Linear
    y <- 2*x -1 + rnorm(n_samples, 0, noise_sd)
    name <- "PolyApprox_Linear"
  } else if (degree == 2) { # Quadrático
    y <- x^2 - 2*x + 1 + rnorm(n_samples, 0, noise_sd)
    name <- "PolyApprox_Quadratic"
  } else { # Sinusoidal como exemplo alternativo
    y <- sin(x) + rnorm(n_samples, 0, noise_sd / 2) # Menos ruído para sin
    name <- "PolyApprox_Sin"
  }
  
  X <- as.matrix(x)
  colnames(X) <- "x"
  Y <- as.matrix(y)
  colnames(Y) <- "y"
  
  return(list(X = X, Y = Y, name = name, type = "regression"))
}