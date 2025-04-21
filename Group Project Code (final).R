library(readxl)

library(tidyverse)
library(GGally)
library(entropy)
library(writexl)
library(officer)
library(flextable)
install.packages(c("keras", "ISLR2", "glmnet", "Matrix", "reticulate", "dplyr", "tidyr", "ggplot2"))
library(keras)
library(ISLR2)
library(glmnet)
library(Matrix)
library(reticulate)
library(dplyr)
library(tidyr)
library(ggplot2)

# Set up Python environment
virtualenv_create("r-tensorflow")
use_virtualenv("r-tensorflow", required = TRUE)
py_install("tensorflow", envname = "r-tensorflow")

# Convert data for Keras
keras <- import("keras")
np <- import("numpy")
tuple <- import_builtins()$tuple

df <- read.csv("gym_members_exercise_tracking.csv")

df$Workout_Type <- factor(df$Workout_Type, levels = c("Cardio", "HIIT", "Strength", "Yoga"), labels = c(1,2,3,4))
df$Workout_Type <- as.numeric(df$Workout_Type)

df$Gender <- factor(df$Gender, levels = c("Female", "Male"), labels = c(1,2))
df$Gender <- as.numeric(df$Gender)

df$Fat_Mass <- df$Weight..kg. * (df$Fat_Percentage / 100) 
df$LBM <- df$Weight..kg. - df$Fat_Mass 
df$FFMI <- df$LBM / (df$Height..m. ^ 2) 

df <- df |>  mutate( natural_status = case_when(
  Gender == 2 & (LBM > 90 | FFMI > 25) ~ "Juice", 
  Gender == 2 & (LBM > 85 | FFMI > 23) ~ "Suspicious", 
  
  # Females (balanced criteria — only LBM & FFMI) 
  Gender == 1 & (LBM > 70 | FFMI > 24) ~ "Juice", 
  Gender == 1 & (LBM > 65 | FFMI > 22) ~ "Suspicious", 
  
  # Everyone else 
  TRUE ~ "Natural" )
  )

df$natural_status <- factor(df$natural_status, levels = c("Natural", "Juice", "Suspicious"), labels = c(1,2,3))
df$natural_status <- as.numeric(df$natural_status)


#Target Variable Analysis: 
# Examine the distribution of the target variable (class labels). 
# Look for class imbalance, as this may require special handling 
# (e.g., oversampling, under sampling, or using class weights). 

table(df$Workout_Type)

255+221+258+239

probs <- prop.table(table(df$Workout_Type))

ggplot(df, aes(x = Workout_Type)) +
  geom_bar() +
  theme_minimal()

target_entropy <- entropy::entropy(probs, unit = "log2")
print(target_entropy)
## High level of randomness/uncertainty might not be good for classification :(
## In decision trees (like in rpart, CART), entropy is used to decide where to 
## split — lower entropy means purer splits.
## Entropy in classification measures the uncertainty or randomness in the 
## distribution of probabilities across classes. With 4 classes, the maximum 
## entropy value is 2 (logarithm base 2 of 4). A value of 1.99 indicates that 
## the probabilities are nearly evenly distributed among the classes, meaning 
## the model is highly uncertain about its predictions. This could suggest that 
## the classes are difficult to distinguish or that the model needs improvement.

#Feature Distribution: 
# Analyse the distribution of each feature. Are they skewed or normally 
# distributed? Would transformations help?

plot(df$Workout_Type)
ggplot(df, aes(x = Workout_Type)) + geom_bar(fill = "lightblue") + 
  labs(x = "Workout Type", y = "Frequency") + theme(panel.grid = element_blank())
ggplot(df, aes(x = natural_status)) + geom_bar(fill = "lightblue") + 
  labs(x = "Natural Status", y = "Frequency") + theme(panel.grid = element_blank())
hist(df$Age)
hist(df$Weight..kg., main = "", xlab = "Weight (kg)", ylab = "Frequency",
     col = "lightblue")
## Positive skew
hist(df$Height..m., main = "", xlab = "Height (m)", ylab = "Frequency",
     col = "lightblue")
## Positive skew
hist(df$Max_BPM)
hist(df$Avg_BPM, main = "", xlab = "Average BPM", ylab = "Frequency",
     col = "lightblue")
hist(df$Resting_BPM)
hist(df$Session_Duration..hours., main = "", xlab = "Session Duration", ylab = "Frequency",
     col = "lightblue")
hist(df$Calories_Burned)
hist(df$Calories_Burned)
hist(df$Fat_Percentage, main = "", xlab = "Fat Percentage", ylab = "Frequency",
     col = "lightblue")
## Negative Skew
hist(df$Water_Intake..liters.)
hist(df$BMI)
hist(df$natural_status)

### Might need to PCA and/or regularise (ridge and/or lasso (good for mulitcolinearity))

#Feature Correlation: 
# Check for correlations between features and the target variable to identify 
# important predictors. 
ggpairs(df)

table(df$natural_status)

## Correlation not as important as classification methods do not need a linear 
## relationship between the IVs and DV.

# Examine multicollinearity between features, as highly correlated features may 
# need to be dropped or combined. 

### Calories burned and session duration

cor.test(df$Calories_Burned, df$Session_Duration..hours.)

### Expierence level and frequency

cor.test(df$Experience_Level, df$Workout_Frequency..days.week.)

### expereince level and duration

cor.test(df$Experience_Level, df$Session_Duration..hours.)

### Expereince level and fat percentage

cor.test(df$Experience_Level, df$Fat_Percentage)

### Reducent vairable: water intake

cor.test(df$Gender, df$Water_Intake..liters.)

### Drop: EL and CB


cor.test(df$LBM, df$Weight..kg.)

cor.test(df$FFMI, df$BMI)

library(GGally)

ggpairs(df)

#Relationship Exploration: 
# Use visualizations to study the relationships between features and their 
# relation to the target variable (e.g., scatterplots, boxplots, or violin plots). 

cor_matrix <- round(cor(df, use = "pairwise.complete.obs"), 3)

write_xlsx(as.data.frame(cor_matrix), "correlation_matrix.xlsx")

cor_matrix <- as.data.frame(cor_matrix)

# Create Word document with the correlation matrix as a table
doc <- read_docx() |> 
  body_add_flextable(flextable(cor_matrix)) |> 
  body_add_par("Generated in R", style = "Normal")

# Save the document
print(doc, target = "correlation_matrix.docx")

## Regluarisation to handle multicollinearity
# Seperating the data to be used in reglurisation and PCA
x = as.matrix(df[, c("Age", "Gender", "Weight..kg.", "Height..m.", "Max_BPM", "Avg_BPM", 
                     "Resting_BPM", "Session_Duration..hours.", "Fat_Percentage", 
                     "Workout_Frequency..days.week.", "natural_status", "Fat_Mass",
                     "BMI")])
y = as.matrix(df$Workout_Type) # This is the dependent variable

# Ridge regression
ridge_df = cv.glmnet(x, y, alpha=0)
plot(ridge_df, xvar = "lambda", label = TRUE)
dim(coef(ridge_df))
# 14 variables and 1 lambda

best_lambda <- ridge_df$lambda.min
print(best_lambda)
library(glmnet)
# Refitting the model with the best lambda
final_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)

# The coefficients of the final model
coef(final_model)

predictions <- predict(final_model, newx = x)
mse <- mean((predictions - y)^2)  # Mean Squared Error
print(mse)
# MSE of 1.26

## Lasso regression
lasso_df = cv.glmnet(x, y, alpha=1)
plot(lasso_df, xvar = "lambda", label = TRUE)
dim(coef(lasso_df))
# 14 variables and 1 lambda

best_lambda <- lasso_df$lambda.min
print(best_lambda)
# Best lambda is 0.06

# Refitting the model with the best lambda
final_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)

# The coefficients of the final model
coef(final_model)
# There are full stops in the coefficients results, due to there being spaces, 
# this is not probalamatic for further analysis

predictions <- predict(final_model, newx = x)
mse <- mean((predictions - y)^2)  # Mean Squared Error
print(mse)
# MSE of 1.27

cv_model_lasso <- cv.glmnet(x, y, alpha = 1)
best_lambda_lasso <- cv_model_lasso$lambda.min
lasso_model <- glmnet(x, y, alpha = 1, lambda = best_lambda_lasso)
lasso_coefs <- as.matrix(coef(lasso_model))
print(lasso_coefs)
# Lasso has over reglularised the model, and has removed all variables

# Fixing the lasso model
adjusted_lambda <- best_lambda / 2  # Reduce by half
lasso_model <- glmnet(x, y, alpha = 1, lambda = adjusted_lambda)

# The coefficients of the final model
coef(lasso_model)

predictions <- predict(lasso_model, newx = x)
mse <- mean((predictions - y)^2)  # Mean Squared Error
print(mse)
# MSE of 1.26

lasso_coefs <- as.matrix(coef(lasso_model))
print(lasso_coefs)
# Age, Gender, Workout Frequency, and BMI are the most important variables

# Neither are appropriate as Ridge and Lasso cannot handle categorical data which
# is not ordered. PCA will be applied instead. 

##################################################
remove.packages("MASS")
library(caret)

# Creating PCA data on training/test data
set.seed(123) 
train_index <- caret::createDataPartition(df$Workout_Type, p = 0.7, list = FALSE)

train_data <- df[train_index, ]
test_data <- df[-train_index, ]

pca_model <- prcomp(train_data[, -which(names(train_data) == "Workout_Type")], scale. = TRUE)

train_pca <- data.frame(predict(pca_model, train_data[, -which(names(train_data) == "Workout_Type")]))
train_pca$Workout_Type <- train_data$Workout_Type 

test_pca <- data.frame(predict(pca_model, test_data[, -which(names(test_data) == "Workout_Type")]))
test_pca$Workout_Type <- test_data$Workout_Type 

# Checking the PCA model and CUM PVE
summary(pca_model)

pca_var <- pca_model$sdev^2
pve <- pca_var / sum(pca_var)
cumsum_pve <- cumsum(pve)

# Plotting the PVE
plot(pve, xlab = "Principal Component", ylab = "Proportion of Variance Explained",
     ylim = c(0, 1), type = "b", main = "PVE of Principal Components")
# Cumulative PVE
plot(cumsum(pve), xlab = "Principal Component", 
     ylab = "Cumulative Proportion of Variance Explained",
     ylim = c(0.1, 1.0), 
     type = "b",
     yaxt = "n")
axis(2, at = seq(0.1, 1.0, by = 0.1), labels = seq(0.1, 1.0, by = 0.1))
# Add horizontal line at 0.8
abline(h = 0.88, col = "red", lty = 2, lwd = 2)
# Add custom line for cumulative PVE
lines(cumsum(pve), col = "blue", lwd = 2, type = "b")
# Cumulative PVE shows that the first 7 components explain about 88% of the variance.

print(pca_model$rotation[, 1:7])

sum(is.na(pca_model))
# No missing values

#################################

#### Base data
library(nnet)

sample_index <- sample(seq_len(nrow(df)), size = 0.7 * nrow(df))

train <- df[sample_index,]
test <-df[-sample_index, ]

df$Workout_Type <- as.factor(df$Workout_Type)

model <- multinom(Workout_Type ~ ., data = train)

predicted <- predict(model, newdata = test)

table(Predicted = predicted, Actual = test$Workout_Type)
mean(predicted == test$Workout_Type)

## Model predicting only 25% of the time
df$Workout_Type <- as.numeric(df$Workout_Type)

df$natural_status <- as.factor(df$natural_status)

model <- multinom(natural_status ~ ., data = train)

predicted <- predict(model, newdata = test)

table(Predicted = predicted, Actual = test$natural_status)
mean(predicted == test$natural_status)
## Model predicting 94% of the time

#### PCA transformed data
predictors <- c("Age", "Gender", "Weight..kg.", "Height..m.", "Max_BPM", "Avg_BPM", 
                "Resting_BPM", "Session_Duration..hours.", "Fat_Percentage", 
                "Workout_Frequency..days.week.", "Fat_Mass", "BMI")

x <- df[, predictors]
y <- df$Workout_Type

sample_index <- sample(seq_len(nrow(df)), size = 0.7 * nrow(df))

x_train <- x[sample_index, ]
x_test <- x[-sample_index, ]

y_train <- y[sample_index]
y_test <- y[-sample_index]

pca_model <- prcomp(x_train, scale. = TRUE)

train_pca <- predict(pca_model, newdata = x_train)[, 1:7]
test_pca  <- predict(pca_model, newdata = x_test)[, 1:7]

train_pca <- data.frame(train_pca)
train_pca$target <- as.factor(y_train)

test_pca <- data.frame(test_pca)
test_pca$target <- as.factor(y_test)

model <- multinom(target ~ ., data = train_pca)

predicted <- predict(model, newdata = test_pca)
table(Predicted = predicted, Actual = test_pca$target)
mean(predicted == test_pca$target)

predictors <- c("Age", "Gender", "Weight..kg.", "Height..m.", "Max_BPM", "Avg_BPM", 
                "Resting_BPM", "Session_Duration..hours.", "Fat_Percentage", 
                "Workout_Frequency..days.week.", "Fat_Mass", "BMI")

x <- df[, predictors]
y <- df$natural_status

sample_index <- sample(seq_len(nrow(df)), size = 0.7 * nrow(df))

x_train <- x[sample_index, ]
x_test <- x[-sample_index, ]

y_train <- y[sample_index]
y_test <- y[-sample_index]

pca_model <- prcomp(x_train, scale. = TRUE)

train_pca <- predict(pca_model, newdata = x_train)[, 1:7]
test_pca  <- predict(pca_model, newdata = x_test)[, 1:7]

train_pca <- data.frame(train_pca)
train_pca$target <- as.factor(y_train)

test_pca <- data.frame(test_pca)
test_pca$target <- as.factor(y_test)

model <- multinom(target ~ ., data = train_pca)

predicted <- predict(model, newdata = test_pca)
table(Predicted = predicted, Actual = test_pca$target)
mean(predicted == test_pca$target)

############################

# Neural Network and LDA

library(tidyverse)
library(GGally)


library(readxl)


# Load data
df <- read.csv("gym_members_exercise_tracking.csv")

# Convert character variables to factors and then numeric
df$Workout_Type <- factor(df$Workout_Type, levels = c("Cardio", "HIIT", "Strength", "Yoga"), labels = c(1, 2, 3, 4))
df$Workout_Type <- as.numeric(df$Workout_Type)
df$Gender <- factor(df$Gender, levels = c("Female", "Male"), labels = c(1, 2))
df$Gender <- as.numeric(df$Gender)

# Calculate LBM, FFMI, and natural status
df$Fat_Mass <- df$Weight..kg. * (df$Fat_Percentage / 100)
df$LBM <- df$Weight..kg. - df$Fat_Mass
df$FFMI <- df$LBM / (df$Height..m.^2)

df <- df %>%
  mutate(natural_status = case_when(
    Gender == 2 & (LBM > 90 | FFMI > 25) ~ "PEDs",
    Gender == 2 & (LBM > 85 | FFMI > 23) ~ "Suspicious",
    Gender == 1 & (LBM > 70 | FFMI > 24) ~ "PEDs",
    Gender == 1 & (LBM > 65 | FFMI > 22) ~ "Suspicious",
    TRUE ~ "Natural"
  ))

df$natural_status <- factor(df$natural_status, levels = c("PEDs", "Suspicious", "Natural"), labels = c(1, 2, 3))
df$natural_status <- as.numeric(df$natural_status)
df <- df %>% dplyr::select(-LBM, -FFMI)

# Drop unnecessary variables
df <- df %>% dplyr::select(-Calories_Burned, -Water_Intake..liters., -Experience_Level)

# ======================
#  NEW TARGET: Workout_Type
# ======================

# Prepare data for PCA — Exclude Workout_Type from X
x <- as.matrix(df[, c("Age", "Gender", "Weight..kg.", "Height..m.", "Max_BPM", "Avg_BPM",
                      "Resting_BPM", "Session_Duration..hours.", "Fat_Percentage",
                      "Workout_Frequency..days.week.", "natural_status", "Fat_Mass", "BMI")])

y <- df$Workout_Type  # Set dependent variable to Workout_Type

# PCA
pca_model <- prcomp(x, scale. = TRUE)
pca_model$rotation <- -pca_model$rotation
pca_model$x <- -pca_model$x

# Prepare PCA dataset
pca_data <- data.frame(pca_model$x[, 1:7])
pca_data$target <- y
pca_data$target <- as.factor(pca_data$target)
pca_data$target_index <- as.integer(pca_data$target) - 1

# Prepare data for keras classification
X_pca <- as.matrix(pca_data[, paste0("PC", 1:7)])
y_pca <- pca_data$target_index

set.seed(13)
n <- nrow(X_pca)
train_size <- floor(0.6 * n)
val_size <- floor(0.2 * n)
test_size <- n - train_size - val_size

indices <- sample(1:n)
train_idx <- indices[1:train_size]
val_idx <- indices[(train_size + 1):(train_size + val_size)]
test_idx <- indices[(train_size + val_size + 1):n]
virtualenv_create("r-tensorflow")
use_virtualenv("r-tensorflow", required = TRUE)
py_install("tensorflow", envname = "r-tensorflow")
reticulate::py_available()
reticulate::py_config()
library(keras)
keras <- import("keras")
np <- import("numpy")
tuple <- import_builtins()$tuple
shape <- tuple(list(ncol(X_pca)))

x_train <- r_to_py(X_pca[train_idx, , drop = FALSE])
y_train <- r_to_py(np$reshape(y_pca[train_idx], tuple(list(length(train_idx), 1L))))

x_val <- r_to_py(X_pca[val_idx, , drop = FALSE])
y_val <- r_to_py(np$reshape(y_pca[val_idx], tuple(list(length(val_idx), 1L))))
val_data <- tuple(list(x_val, y_val))

x_test <- r_to_py(X_pca[test_idx, , drop = FALSE])
y_test <- r_to_py(np$reshape(y_pca[test_idx], tuple(list(length(test_idx), 1L))))

# Define model
model_pca <- keras$models$Sequential()
model_pca$add(keras$layers$Dense(units = 64L, activation = "relu", input_shape = shape))
model_pca$add(keras$layers$Dense(units = 32L, activation = "relu"))
model_pca$add(keras$layers$Dense(units = 4L, activation = "softmax"))  # 4 classes for workout type

# Compile
model_pca$compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = keras$optimizers$Adam(),
  metrics = list("accuracy", "mae")
)

# Train
history_pca <- model_pca$fit(
  x = x_train,
  y = y_train,
  epochs = 150L,
  batch_size = 16L,
  validation_data = val_data
)

# Evaluate
pred_probs <- model_pca$predict(x_test)
pred_classes <- np$argmax(pred_probs, axis = 1L)
true_classes <- as.integer(py_to_r(y_test))

accuracy <- mean(pred_classes == true_classes)
mae_manual <- mean(abs(pred_classes - true_classes))

cat("PCA Test Accuracy:", round(accuracy * 100, 2), "%\n") #29.59%
cat("PCA Test MAE:", round(mae_manual, 4), "\n") # 1.3163
print(table(Predicted = pred_classes, Actual = true_classes))
# Prepare data for keras classification
X_pca <- as.matrix(pca_data[, paste0("PC", 1:7)])
y_pca <- pca_data$target_index

set.seed(13)
n <- nrow(X_pca)
train_size <- floor(0.6 * n)
val_size <- floor(0.2 * n)
test_size <- n - train_size - val_size

indices <- sample(1:n)
train_idx <- indices[1:train_size]
val_idx <- indices[(train_size + 1):(train_size + val_size)]
test_idx <- indices[(train_size + val_size + 1):n]

keras <- import("keras")
np <- import("numpy")
tuple <- import_builtins()$tuple
shape <- tuple(list(ncol(X_pca)))

x_train <- r_to_py(X_pca[train_idx, , drop = FALSE])
y_train <- r_to_py(np$reshape(y_pca[train_idx], tuple(list(length(train_idx), 1L))))

x_val <- r_to_py(X_pca[val_idx, , drop = FALSE])
y_val <- r_to_py(np$reshape(y_pca[val_idx], tuple(list(length(val_idx), 1L))))
val_data <- tuple(list(x_val, y_val))

x_test <- r_to_py(X_pca[test_idx, , drop = FALSE])
y_test <- r_to_py(np$reshape(y_pca[test_idx], tuple(list(length(test_idx), 1L))))

# Define model
model_pca <- keras$models$Sequential()
model_pca$add(keras$layers$Dense(units = 64L, activation = "relu", input_shape = shape))
model_pca$add(keras$layers$Dense(units = 32L, activation = "relu"))
model_pca$add(keras$layers$Dense(units = 4L, activation = "softmax"))  # 4 classes for workout type

# Compile
model_pca$compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = keras$optimizers$Adam(),
  metrics = list("accuracy", "mae")
)

# Train
history_pca <- model_pca$fit(
  x = x_train,
  y = y_train,
  epochs = 150L,
  batch_size = 16L,
  validation_data = val_data
)

# Evaluate
pred_probs <- model_pca$predict(x_test)
pred_classes <- np$argmax(pred_probs, axis = 1L)
true_classes <- as.integer(py_to_r(y_test))

accuracy <- mean(pred_classes == true_classes)
mae_manual <- mean(abs(pred_classes - true_classes))

cat("PCA Test Accuracy:", round(accuracy * 100, 2), "%\n") #29.59%
cat("PCA Test MAE:", round(mae_manual, 4), "\n") # 1.3163
print(table(Predicted = pred_classes, Actual = true_classes))

library(tidyverse)
library(GGally)
library(glmnet)
library(dplyr)
library(readxl)
library(reticulate)
library(keras)
library(MASS)

# Load data
df <- read.csv("gym_members_exercise_tracking.csv")

# Convert character variables to factors and then numeric
df$Workout_Type <- factor(df$Workout_Type, levels = c("Cardio", "HIIT", "Strength", "Yoga"), labels = c(1, 2, 3, 4))
df$Workout_Type <- as.numeric(df$Workout_Type)
df$Gender <- factor(df$Gender, levels = c("Female", "Male"), labels = c(1, 2))
df$Gender <- as.numeric(df$Gender)

# Calculate LBM, FFMI, and natural status
df$Fat_Mass <- df$Weight..kg. * (df$Fat_Percentage / 100)
df$LBM <- df$Weight..kg. - df$Fat_Mass
df$FFMI <- df$LBM / (df$Height..m.^2)

df <- df %>%
  mutate(natural_status = case_when(
    Gender == 2 & (LBM > 90 | FFMI > 25) ~ "PEDs",
    Gender == 2 & (LBM > 85 | FFMI > 23) ~ "Suspicious",
    Gender == 1 & (LBM > 70 | FFMI > 24) ~ "PEDs",
    Gender == 1 & (LBM > 65 | FFMI > 22) ~ "Suspicious",
    TRUE ~ "Natural"
  ))

df$natural_status <- factor(df$natural_status, levels = c("PEDs", "Suspicious", "Natural"), labels = c(1, 2, 3))
df$natural_status <- as.numeric(df$natural_status)
df <- df %>% dplyr::select(-LBM, -FFMI)

# Drop unnecessary variables
df <- df %>% dplyr::select(-Calories_Burned, -Water_Intake..liters., -Experience_Level)
df
# Prepare data (no PCA)
x <- as.matrix(df[, c("Age", "Gender", "Weight..kg.", "Height..m.", "Max_BPM", "Avg_BPM",
                      "Resting_BPM", "Session_Duration..hours.", "Fat_Percentage",
                      "Workout_Frequency..days.week.", "natural_status", "Fat_Mass", "BMI")])



y <- df$Workout_Type

set.seed(13)
n <- nrow(x)
train_size <- floor(0.6 * n)
val_size <- floor(0.2 * n)
test_size <- n - train_size - val_size

indices <- sample(1:n)
train_idx <- indices[1:train_size]
val_idx <- indices[(train_size + 1):(train_size + val_size)]
test_idx <- indices[(train_size + val_size + 1):n]

keras <- import("keras")
np <- import("numpy")
tuple <- import_builtins()$tuple
shape <- tuple(list(ncol(x)))

x_train <- r_to_py(x[train_idx, , drop = FALSE])
y_train <- r_to_py(np$reshape(y[train_idx] - 1L, tuple(list(length(train_idx), 1L))))

x_val <- r_to_py(x[val_idx, , drop = FALSE])
y_val <- r_to_py(np$reshape(y[val_idx] - 1L, tuple(list(length(val_idx), 1L))))
val_data <- tuple(list(x_val, y_val))

x_test <- r_to_py(x[test_idx, , drop = FALSE])
y_test <- r_to_py(np$reshape(y[test_idx] - 1L, tuple(list(length(test_idx), 1L))))

# Define model
model <- keras$models$Sequential()
model$add(keras$layers$Dense(units = 64L, activation = "relu", input_shape = shape))
model$add(keras$layers$Dense(units = 32L, activation = "relu"))
model$add(keras$layers$Dense(units = 4L, activation = "softmax"))

# Compile
model$compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = keras$optimizers$Adam(),
  metrics = list("accuracy", "mae")
)

# Train
history <- model$fit(
  x = x_train,
  y = y_train,
  epochs = 150L,
  batch_size = 16L,
  validation_data = val_data
)

# Evaluate
pred_probs <- model$predict(x_test)
pred_classes <- np$argmax(pred_probs, axis = 1L)
true_classes <- as.integer(py_to_r(y_test))

accuracy <- mean(pred_classes == true_classes)
mae_manual <- mean(abs(pred_classes - true_classes))

cat("Test Accuracy:", round(accuracy * 100, 2), "%\n")
cat("Test MAE:", round(mae_manual, 4), "\n")
print(table(Predicted = pred_classes, Actual = true_classes))


library(tidyverse)
library(GGally)
library(glmnet)
library(dplyr)
library(readxl)
library(reticulate)
library(keras)
library(MASS)

# Load data
df <- read.csv("gym_members_exercise_tracking.csv")

# Convert character variables to factors and then numeric
df$Workout_Type <- factor(df$Workout_Type, levels = c("Cardio", "HIIT", "Strength", "Yoga"), labels = c(1, 2, 3, 4))
df$Workout_Type <- as.numeric(df$Workout_Type)
df$Gender <- factor(df$Gender, levels = c("Female", "Male"), labels = c(1, 2))
df$Gender <- as.numeric(df$Gender)

# Calculate LBM, FFMI, and natural status
df$Fat_Mass <- df$Weight..kg. * (df$Fat_Percentage / 100)
df$LBM <- df$Weight..kg. - df$Fat_Mass
df$FFMI <- df$LBM / (df$Height..m.^2)

df <- df %>%
  mutate(natural_status = case_when(
    Gender == 2 & (LBM > 90 | FFMI > 25) ~ "PEDs",
    Gender == 2 & (LBM > 85 | FFMI > 23) ~ "Suspicious",
    Gender == 1 & (LBM > 70 | FFMI > 24) ~ "PEDs",
    Gender == 1 & (LBM > 65 | FFMI > 22) ~ "Suspicious",
    TRUE ~ "Natural"
  ))

df$natural_status <- factor(df$natural_status, levels = c("PEDs", "Suspicious", "Natural"), labels = c(1, 2, 3))
df$natural_status <- as.numeric(df$natural_status)
df <- df %>% dplyr::select(-LBM, -FFMI)


# Drop unnecessary variables
df <- df %>% dplyr::select(-Calories_Burned, -Water_Intake..liters., -Experience_Level)




# Prepare data for PCA — Exclude Workout_Type from X
x <- as.matrix(df[, c("Age", "Gender", "Weight..kg.", "Height..m.", "Max_BPM", "Avg_BPM",
                      "Resting_BPM", "Session_Duration..hours.", "Fat_Percentage",
                      "Workout_Frequency..days.week.", "Fat_Mass", "BMI")])

y <- df$natural_status  # Set dependent variable to Workout_Type

# PCA
pca_model <- prcomp(x, scale. = TRUE)
pca_model$rotation <- -pca_model$rotation
pca_model$x <- -pca_model$x

# Prepare PCA dataset
pca_data <- data.frame(pca_model$x[, 1:7])
pca_data$target <- y
pca_data$target <- as.factor(pca_data$target)
pca_data$target_index <- as.integer(pca_data$target) - 1

# Prepare data for keras classification
X_pca <- as.matrix(pca_data[, paste0("PC", 1:7)])
y_pca <- pca_data$target_index

set.seed(13)
n <- nrow(X_pca)
train_size <- floor(0.6 * n)
val_size <- floor(0.2 * n)
test_size <- n - train_size - val_size

indices <- sample(1:n)
train_idx <- indices[1:train_size]
val_idx <- indices[(train_size + 1):(train_size + val_size)]
test_idx <- indices[(train_size + val_size + 1):n]

keras <- import("keras")
np <- import("numpy")
tuple <- import_builtins()$tuple
shape <- tuple(list(ncol(X_pca)))

x_train <- r_to_py(X_pca[train_idx, , drop = FALSE])
y_train <- r_to_py(np$reshape(y_pca[train_idx], tuple(list(length(train_idx), 1L))))

x_val <- r_to_py(X_pca[val_idx, , drop = FALSE])
y_val <- r_to_py(np$reshape(y_pca[val_idx], tuple(list(length(val_idx), 1L))))
val_data <- tuple(list(x_val, y_val))

x_test <- r_to_py(X_pca[test_idx, , drop = FALSE])
y_test <- r_to_py(np$reshape(y_pca[test_idx], tuple(list(length(test_idx), 1L))))

# Define model
model_pca <- keras$models$Sequential()
model_pca$add(keras$layers$Dense(units = 64L, activation = "relu", input_shape = shape))
model_pca$add(keras$layers$Dense(units = 32L, activation = "relu"))
model_pca$add(keras$layers$Dense(units = 4L, activation = "softmax"))  # 4 classes for workout type

# Compile
model_pca$compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = keras$optimizers$Adam(),
  metrics = list("accuracy", "mae")
)

# Train
history_natural <- model_pca$fit(
  x = x_train,
  y = y_train,
  epochs = 150L,
  batch_size = 16L,
  validation_data = val_data
)

# Evaluate
pred_probs <- model_pca$predict(x_test)
pred_classes <- np$argmax(pred_probs, axis = 1L)
true_classes <- as.integer(py_to_r(y_test))

accuracy <- mean(pred_classes == true_classes)
mae_manual <- mean(abs(pred_classes - true_classes))

cat("PCA Test Accuracy:", round(accuracy * 100, 2), "%\n") #29.59%
cat("PCA Test MAE:", round(mae_manual, 4), "\n") # 1.3163
print(table(Predicted = pred_classes, Actual = true_classes))
# Prepare data for keras classification
X_pca <- as.matrix(pca_data[, paste0("PC", 1:7)])
y_pca <- pca_data$target_index

set.seed(13)
n <- nrow(X_pca)
train_size <- floor(0.6 * n)
val_size <- floor(0.2 * n)
test_size <- n - train_size - val_size

indices <- sample(1:n)
train_idx <- indices[1:train_size]
val_idx <- indices[(train_size + 1):(train_size + val_size)]
test_idx <- indices[(train_size + val_size + 1):n]

keras <- import("keras")
np <- import("numpy")
tuple <- import_builtins()$tuple
shape <- tuple(list(ncol(X_pca)))

x_train <- r_to_py(X_pca[train_idx, , drop = FALSE])
y_train <- r_to_py(np$reshape(y_pca[train_idx], tuple(list(length(train_idx), 1L))))

x_val <- r_to_py(X_pca[val_idx, , drop = FALSE])
y_val <- r_to_py(np$reshape(y_pca[val_idx], tuple(list(length(val_idx), 1L))))
val_data <- tuple(list(x_val, y_val))

x_test <- r_to_py(X_pca[test_idx, , drop = FALSE])
y_test <- r_to_py(np$reshape(y_pca[test_idx], tuple(list(length(test_idx), 1L))))

# Define model
model_pca <- keras$models$Sequential()
model_pca$add(keras$layers$Dense(units = 64L, activation = "relu", input_shape = shape))
model_pca$add(keras$layers$Dense(units = 32L, activation = "relu"))
model_pca$add(keras$layers$Dense(units = 4L, activation = "softmax"))  # 4 classes for workout type

# Compile
model_pca$compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = keras$optimizers$Adam(),
  metrics = list("accuracy", "mae")
)

# Train
history_pca <- model_pca$fit(
  x = x_train,
  y = y_train,
  epochs = 150L,
  batch_size = 16L,
  validation_data = val_data
)

# Evaluate
pred_probs <- model_pca$predict(x_test)
pred_classes <- np$argmax(pred_probs, axis = 1L)
true_classes <- as.integer(py_to_r(y_test))

accuracy <- mean(pred_classes == true_classes)
mae_manual <- mean(abs(pred_classes - true_classes))

cat("PCA Test Accuracy:", round(accuracy * 100, 2), "%\n") 
cat("PCA Test MAE:", round(mae_manual, 4), "\n") 
print(table(Predicted = pred_classes, Actual = true_classes))


library(tidyverse)
library(GGally)
library(glmnet)
library(dplyr)
library(readxl)
library(reticulate)
library(keras)
library(MASS)

# Load data
df <- read.csv("gym_members_exercise_tracking.csv")

# Convert character variables to factors and then numeric
df$Workout_Type <- factor(df$Workout_Type, levels = c("Cardio", "HIIT", "Strength", "Yoga"), labels = c(1, 2, 3, 4))
df$Workout_Type <- as.numeric(df$Workout_Type)
df$Gender <- factor(df$Gender, levels = c("Female", "Male"), labels = c(1, 2))
df$Gender <- as.numeric(df$Gender)

# Calculate LBM, FFMI, and natural status
df$Fat_Mass <- df$Weight..kg. * (df$Fat_Percentage / 100)
df$LBM <- df$Weight..kg. - df$Fat_Mass
df$FFMI <- df$LBM / (df$Height..m.^2)

df <- df %>%
  mutate(natural_status = case_when(
    Gender == 2 & (LBM > 90 | FFMI > 25) ~ "PEDs",
    Gender == 2 & (LBM > 85 | FFMI > 23) ~ "Suspicious",
    Gender == 1 & (LBM > 70 | FFMI > 24) ~ "PEDs",
    Gender == 1 & (LBM > 65 | FFMI > 22) ~ "Suspicious",
    TRUE ~ "Natural"))
    
    library(tidyverse)
    library(GGally)
    library(glmnet)
    library(dplyr)
    library(readxl)
    library(reticulate)
    library(keras)
    library(MASS)
    
    # Load data
    df <- read.csv("gym_members_exercise_tracking.csv")
    
    # Convert character variables to factors and then numeric
    df$Workout_Type <- factor(df$Workout_Type, levels = c("Cardio", "HIIT", "Strength", "Yoga"), labels = c(1, 2, 3, 4))
    df$Workout_Type <- as.numeric(df$Workout_Type)
    df$Gender <- factor(df$Gender, levels = c("Female", "Male"), labels = c(1, 2))
    df$Gender <- as.numeric(df$Gender)
    
    # Calculate LBM, FFMI, and natural status
    df$Fat_Mass <- df$Weight..kg. * (df$Fat_Percentage / 100)
    df$LBM <- df$Weight..kg. - df$Fat_Mass
    df$FFMI <- df$LBM / (df$Height..m.^2)
    
    df <- df %>%
      mutate(natural_status = case_when(
        Gender == 2 & (LBM > 90 | FFMI > 25) ~ "PEDs",
        Gender == 2 & (LBM > 85 | FFMI > 23) ~ "Suspicious",
        Gender == 1 & (LBM > 70 | FFMI > 24) ~ "PEDs",
        Gender == 1 & (LBM > 65 | FFMI > 22) ~ "Suspicious",
        TRUE ~ "Natural"
      ))
    
    df$natural_status <- factor(df$natural_status, levels = c("PEDs", "Suspicious", "Natural"), labels = c(1, 2, 3))
    df$natural_status <- as.numeric(df$natural_status)
    df <- df %>% dplyr::select(-LBM, -FFMI)
    
    # Drop unnecessary variables
    df <- df %>% dplyr::select(-Calories_Burned, -Water_Intake..liters., -Experience_Level)
    df
    # Prepare data (no PCA)
    x <- as.matrix(df[, c("Age", "Gender", "Weight..kg.", "Height..m.", "Max_BPM", "Avg_BPM",
                          "Resting_BPM", "Session_Duration..hours.", "Fat_Percentage",
                          "Workout_Frequency..days.week.", "natural_status", "Fat_Mass", "BMI")])
    
    
    
    y <- df$Workout_Type
    
    set.seed(13)
    n <- nrow(x)
    train_size <- floor(0.6 * n)
    val_size <- floor(0.2 * n)
    test_size <- n - train_size - val_size
    
    indices <- sample(1:n)
    train_idx <- indices[1:train_size]
    val_idx <- indices[(train_size + 1):(train_size + val_size)]
    test_idx <- indices[(train_size + val_size + 1):n]
    
    keras <- import("keras")
    np <- import("numpy")
    tuple <- import_builtins()$tuple
    shape <- tuple(list(ncol(x)))
    
    x_train <- r_to_py(x[train_idx, , drop = FALSE])
    y_train <- r_to_py(np$reshape(y[train_idx] - 1L, tuple(list(length(train_idx), 1L))))
    
    x_val <- r_to_py(x[val_idx, , drop = FALSE])
    y_val <- r_to_py(np$reshape(y[val_idx] - 1L, tuple(list(length(val_idx), 1L))))
    val_data <- tuple(list(x_val, y_val))
    
    x_test <- r_to_py(x[test_idx, , drop = FALSE])
    y_test <- r_to_py(np$reshape(y[test_idx] - 1L, tuple(list(length(test_idx), 1L))))
    
    # Define model
    model <- keras$models$Sequential()
    model$add(keras$layers$Dense(units = 64L, activation = "relu", input_shape = shape))
    model$add(keras$layers$Dense(units = 32L, activation = "relu"))
    model$add(keras$layers$Dense(units = 4L, activation = "softmax"))
    
    # Compile
    model$compile(
      loss = "sparse_categorical_crossentropy",
      optimizer = keras$optimizers$Adam(),
      metrics = list("accuracy", "mae")
    )
    
    # Train
    history <- model$fit(
      x = x_train,
      y = y_train,
      epochs = 150L,
      batch_size = 16L,
      validation_data = val_data
    )
    
    # Evaluate
    pred_probs <- model$predict(x_test)
    pred_classes <- np$argmax(pred_probs, axis = 1L)
    true_classes <- as.integer(py_to_r(y_test))
    
    accuracy <- mean(pred_classes == true_classes)
    mae_manual <- mean(abs(pred_classes - true_classes))
    
    cat("Test Accuracy:", round(accuracy * 100, 2), "%\n")
    cat("Test MAE:", round(mae_manual, 4), "\n")
    print(table(Predicted = pred_classes, Actual = true_classes))
    
    # Evaluate
    pred_probs <- model$predict(x_test)
    pred_classes <- np$argmax(pred_probs, axis = 1L)
    true_classes <- as.integer(py_to_r(y_test))
    
    accuracy <- mean(pred_classes == true_classes)
    mae_manual <- mean(abs(pred_classes - true_classes))
    
    cat("Test Accuracy:", round(accuracy * 100, 2), "%\n")
    cat("Test MAE:", round(mae_manual, 4), "\n")
    print(table(Predicted = pred_classes, Actual = true_classes))
    
    
    library(tidyverse)
    library(GGally)
    library(glmnet)
    library(dplyr)
    library(readxl)
    library(reticulate)
    library(keras)
    library(MASS)
    
    # Load data
    df <- read.csv("gym_members_exercise_tracking.csv")
    
    # Convert character variables to factors and then numeric
    df$Workout_Type <- factor(df$Workout_Type, levels = c("Cardio", "HIIT", "Strength", "Yoga"), labels = c(1, 2, 3, 4))
    df$Workout_Type <- as.numeric(df$Workout_Type)
    df$Gender <- factor(df$Gender, levels = c("Female", "Male"), labels = c(1, 2))
    df$Gender <- as.numeric(df$Gender)
    
    # Calculate LBM, FFMI, and natural status
    df$Fat_Mass <- df$Weight..kg. * (df$Fat_Percentage / 100)
    df$LBM <- df$Weight..kg. - df$Fat_Mass
    df$FFMI <- df$LBM / (df$Height..m.^2)
    
    df <- df %>%
      mutate(natural_status = case_when(
        Gender == 2 & (LBM > 90 | FFMI > 25) ~ "PEDs",
        Gender == 2 & (LBM > 85 | FFMI > 23) ~ "Suspicious",
        Gender == 1 & (LBM > 70 | FFMI > 24) ~ "PEDs",
        Gender == 1 & (LBM > 65 | FFMI > 22) ~ "Suspicious",
        TRUE ~ "Natural"
      ))
    
    df$natural_status <- factor(df$natural_status, levels = c("PEDs", "Suspicious", "Natural"), labels = c(1, 2, 3))
    df$natural_status <- as.numeric(df$natural_status)
    df <- df %>% dplyr::select(-LBM, -FFMI)
    
    # Drop unnecessary variables
    df <- df %>% dplyr::select(-Calories_Burned, -Water_Intake..liters., -Experience_Level)
    df
    # Prepare data (no PCA)
    x <- as.matrix(df[, c("Age", "Gender", "Weight..kg.", "Height..m.", "Max_BPM", "Avg_BPM",
                          "Resting_BPM", "Session_Duration..hours.", "Fat_Percentage",
                          "Workout_Frequency..days.week.", "Fat_Mass", "BMI")])
    
    
    y <- df$natural_status
    
    set.seed(13)
    n <- nrow(x)
    train_size <- floor(0.6 * n)
    val_size <- floor(0.2 * n)
    test_size <- n - train_size - val_size
    
    indices <- sample(1:n)
    train_idx <- indices[1:train_size]
    val_idx <- indices[(train_size + 1):(train_size + val_size)]
    test_idx <- indices[(train_size + val_size + 1):n]
    
    keras <- import("keras")
    np <- import("numpy")
    tuple <- import_builtins()$tuple
    shape <- tuple(list(ncol(x)))
    
    x_train <- r_to_py(x[train_idx, , drop = FALSE])
    y_train <- r_to_py(np$reshape(y[train_idx] - 1L, tuple(list(length(train_idx), 1L))))
    
    x_val <- r_to_py(x[val_idx, , drop = FALSE])
    y_val <- r_to_py(np$reshape(y[val_idx] - 1L, tuple(list(length(val_idx), 1L))))
    val_data <- tuple(list(x_val, y_val))
    
    x_test <- r_to_py(x[test_idx, , drop = FALSE])
    y_test <- r_to_py(np$reshape(y[test_idx] - 1L, tuple(list(length(test_idx), 1L))))
    
    # Define model
    model <- keras$models$Sequential()
    model$add(keras$layers$Dense(units = 64L, activation = "relu", input_shape = shape))
    model$add(keras$layers$Dense(units = 32L, activation = "relu"))
    model$add(keras$layers$Dense(units = 4L, activation = "softmax"))
    
    # Compile
    model$compile(
      loss = "sparse_categorical_crossentropy",
      optimizer = keras$optimizers$Adam(),
      metrics = list("accuracy", "mae")
    )
    
    # Train
    history <- model$fit(
      x = x_train,
      y = y_train,
      epochs = 150L,
      batch_size = 16L,
      validation_data = val_data
    )
    
    # Evaluate
    pred_probs <- model$predict(x_test)
    pred_classes <- np$argmax(pred_probs, axis = 1L)
    true_classes <- as.integer(py_to_r(y_test))
    
    accuracy <- mean(pred_classes == true_classes)
    mae_manual <- mean(abs(pred_classes - true_classes))
    
    cat("Test Accuracy:", round(accuracy * 100, 2), "%\n")
    cat("Test MAE:", round(mae_manual, 4), "\n")
    print(table(Predicted = pred_classes, Actual = true_classes))
    
    install.packages("MASS")  # if not already installed
    library(MASS)
    library(dplyr)
    
    # Defining formula of features
    lda_formula <- natural_status ~ Age + Gender + Weight..kg. + Height..m. + 
      Max_BPM + Avg_BPM + Resting_BPM + 
      Session_Duration..hours. + Fat_Percentage + 
      Workout_Frequency..days.week. + Fat_Mass + BMI
    
    set.seed(13)
    train_indices <- sample(1:nrow(df), size = 0.7 * nrow(df))  # 70% train
    df_train <- df[train_indices, ]
    df_test <- df[-train_indices, ]
    
    lda_model <- lda(lda_formula, data = df_train)
    
    lda_pred <- predict(lda_model, newdata = df_test)
    
    confusion_matrix <- table(Predicted = lda_pred$class, Actual = df_test$natural_status)
    print(confusion_matrix)
    
    accuracy <- mean(lda_pred$class == df_test$natural_status)
    print(paste("Test Accuracy:", round(accuracy * 100, 2), "%")) 
    #94.51
    
    
    # PCA classification using natural_status as target
    x <- as.matrix(df[, c("Age", "Gender", "Weight..kg.", "Height..m.", "Max_BPM", "Avg_BPM",
                          "Resting_BPM", "Session_Duration..hours.", "Fat_Percentage",
                          "Workout_Frequency..days.week.", "Fat_Mass", "BMI")])
    y <- df$natural_status - 1L  # Convert to 0-indexed
    
    # PCA
    pca_model <- prcomp(x, scale. = TRUE)
    pca_model$rotation <- -pca_model$rotation
    pca_model$x <- -pca_model$x
    
    # Prepare PCA dataset
    pca_data <- data.frame(pca_model$x[, 1:7])
    pca_data$target <- y
    
    set.seed(13)
    n <- nrow(pca_data)
    train_size <- floor(0.7 * n)
    train_indices <- sample(1:n, size = train_size)
    
    df_train <- pca_data[train_indices, ]
    df_test <- pca_data[-train_indices, ]
    
    # LDA with PCA features
    lda_model <- lda(target ~ ., data = df_train)
    lda_pred <- predict(lda_model, newdata = df_test)
    
    confusion_matrix <- table(Predicted = lda_pred$class, Actual = df_test$target)
    print(confusion_matrix)
    
    accuracy <- mean(lda_pred$class == df_test$target)
    print(paste("Test Accuracy:", round(accuracy * 100, 2), "%"))
    
    # Defining formula of features
    lda_formula <- Workout_Type ~ Age + Gender + Weight..kg. + Height..m. + 
      Max_BPM + Avg_BPM + Resting_BPM + 
      Session_Duration..hours. + Fat_Percentage + 
      Workout_Frequency..days.week. + Fat_Mass + BMI
    
    set.seed(13)
    train_indices <- sample(1:nrow(df), size = 0.7 * nrow(df))  # 70% train
    df_train <- df[train_indices, ]
    df_test <- df[-train_indices, ]
    
    lda_model <- lda(lda_formula, data = df_train)
    
    lda_pred <- predict(lda_model, newdata = df_test)
    
    confusion_matrix <- table(Predicted = lda_pred$class, Actual = df_test$Workout_Type)
    print(confusion_matrix)
    
    accuracy <- mean(lda_pred$class == df_test$Workout_Type)
    print(paste("Test Accuracy:", round(accuracy * 100, 2), "%")) #25%
    
    
    
    set.seed(13)
    train_indices <- sample(1:nrow(df), size = 0.7 * nrow(df))  # 70% train
    df_train <- df[train_indices, ]
    df_test <- df[-train_indices, ]
    
    lda_model <- lda(lda_formula, data = df_train)
    
    lda_pred <- predict(lda_model, newdata = df_test)
    
    confusion_matrix <- table(Predicted = lda_pred$class, Actual = df_test$natural_status)
    print(confusion_matrix)
    
    accuracy <- mean(lda_pred$class == df_test$natural_status)
    print(paste("Test Accuracy:", round(accuracy * 100, 2), "%")) 

    