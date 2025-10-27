setwd("C:/Users/fxgoh/Desktop/FX/DSA1101")
df = read.csv("C:/Users/fxgoh/Desktop/FX/DSA1101/insurance.csv")
head(df)
attach(df)

library(ggplot2)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)

### Exploratory Data Analysis ###

str(df)
colSums(is.na(df))    # no missing values


# PART I - Check distribution of each variables

hist(df$age, col = blues9, xlab = "Age", main = "Histogram of Age")
hist(df$bmi, col = blues9, xlab = "BMI", main = "Histogram of BMI")
hist(df$charges, col = blues9, xlab = "Charges", main = "Histogram of Charges")
df$logcharges <- log(df$charges, exp(1))
# since charges is right skewed, we treat it by taking log of the charges

table(df$sex)
table(df$children)
table(df$smoker)
table(df$region)


# PART II - Check association between each variable and response

df$sex <- as.factor(df$sex)
df$children <- as.factor(df$children)
df$smoker <- as.factor(df$smoker)
df$region <- as.factor(df$region)

ggplot(df, aes(x = sex, y = logcharges, fill = sex)) +
  geom_boxplot() +
  labs(title = "Boxplot of Sex",
       x = "Sex",
       y = "log(Charges)")

ggplot(df, aes(x = smoker, y = logcharges, fill = smoker)) +
  geom_boxplot() +
  labs(title = "Boxplot of Smoker",
       x = "Smoker",
       y = "log(Charges)")

ggplot(df, aes(x = region, y = logcharges, fill = region)) +
  geom_boxplot() +
  labs(title = "Boxplot of Region",
       x = "Region",
       y = "log(Charges)")

ggplot(df, aes(x = children, y = logcharges, fill = children)) +
  geom_boxplot() +
  labs(title = "Boxplot of Children",
       x = "Children",
       y = "log(Charges)")

ggplot(df, aes(x = age, y = logcharges)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Scatterplot of Age vs log(Charges)",
       x = "Age",
       y = "log(Charges)")
cor(df$age, df$logcharges)  # 0.527834

ggplot(df, aes(x = bmi, y = logcharges)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Scatterplot of BMI vs log(Charges)",
       x = "BMI",
       y = "log(Charges)")
cor(df$bmi, df$logcharges)  # 0.1326694



### Baseline Modelling using Linear Regression ###

set.seed(42)
df$children <- as.numeric(as.character(df$children))
n <- nrow(df)

idx <- sample.int(n, size = floor(0.8 * n))   # 80% train

train <- df[idx, ]
test  <- df[-idx, ]

# Fit on train
model_tt <- lm(charges ~ age + sex + bmi + children + smoker + region, data = train)

# Predict on test
pred_test <- predict(model_tt, newdata = test)
res_test  <- test$charges - pred_test

# Test metrics
MSE_test  <- mean(res_test^2)
RMSE_test <- sqrt(MSE_test)

# Test R^2 (computed against test mean)
SS_res <- sum(res_test^2)
SS_tot <- sum( (test$charges - mean(test$charges))^2 )
R2_test <- 1 - SS_res/SS_tot

cat("Test MSE:", MSE_test, "\nTest RMSE:", RMSE_test, "\nTest R^2:", R2_test, "\n")
summary(model_tt)



### Advanced Modelling ###

# PART I - Decision Trees
set.seed(123)

# Create training (80%) and testing (20%) indices
train_index <- sample(1:nrow(df), 0.8 * nrow(df))

# Split the dataset
train_data <- df[train_index, ]
test_data  <- df[-train_index, ]

# Fit the tree again
tree_model <- rpart(charges ~ smoker + age + sex + bmi + children + region,
                    data = train_data, method = "anova")
# Fancy plot
fancyRpartPlot(tree_model)
title(main = "Decision Tree for Predicting Charges", cex.main = 1.4, col.main = "black")

pred <- predict(tree_model, newdata = test_data)

# Compute R-squared
SST <- sum((test_data$charges - mean(test_data$charges))^2)
SSE <- sum((test_data$charges - pred)^2)
R2_test <- 1 - SSE/SST
R2_test   # 0.7965152

# Compute RMSE
RMSE_test <- sqrt(mean((test_data$charges - pred)^2))
RMSE_test  # 5019.467

mean(df$charges)  # 13270.42



# PART I - Random Forest

rf_model <- randomForest(
  charges ~ age + sex + bmi + children + region + smoker,
  data = train_data,
  ntree = 1000,
  mtry = 3,
  importance = TRUE
)
predictions <- predict(rf_model, newdata = test_data)

# Calculate RMSE
rmse <- sqrt(mean((predictions - test_data$charges)^2))
rmse   # 4293.665
print(rf_model)
varImpPlot(rf_model)





