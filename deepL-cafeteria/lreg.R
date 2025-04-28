library(readr)
library(dplyr)
library(tidyr)
library(caret)


data <- read_csv("daily_sort.csv")

# add up the data to get total counts for each product
product_counts <- data %>%
  pivot_longer(cols = starts_with("ProductName_"), 
               names_to = "ProductType", values_to = "ProductName") %>%
  bind_cols(
    data %>%
      pivot_longer(cols = starts_with("count_"),
                   names_to = "CountType", values_to = "Count") %>%
      select(Count)
  ) %>%
  mutate(NormalizedCount = Count / total_count)

# Calculate average popularity for each product
product_popularity <- product_counts %>%
  group_by(ProductName) %>%
  summarise(NormalizedCount = mean(NormalizedCount, na.rm = TRUE)) %>%
  ungroup()

print(product_popularity)

# Create all possible pairs of products
product_pairs <- combn(product_popularity$ProductName, 2, simplify = FALSE)

# Create pair data with popularity difference
pair_data <- lapply(product_pairs, function(pair) {
  product1 <- pair[1]
  product2 <- pair[2]
  pop1 <- product_popularity %>% filter(ProductName == product1) %>% pull(NormalizedCount)
  pop2 <- product_popularity %>% filter(ProductName == product2) %>% pull(NormalizedCount)
  data.frame(Product1 = product1, Product2 = product2, PopularityDiff = pop1 - pop2)
})

pair_df <- bind_rows(pair_data)

# Encode product names as categorical variables (one-hot encoding)
X <- pair_df %>% select(Product1, Product2)
y <- pair_df$PopularityDiff

# Split data into training and test sets
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# One-hot encode
X_train_encoded <- model.matrix(~ Product1 + Product2 - 1, data = X_train) %>% as.data.frame()
X_test_encoded <- model.matrix(~ Product1 + Product2 - 1, data = X_test) %>% as.data.frame()

# Align columns
missing_cols <- setdiff(names(X_train_encoded), names(X_test_encoded))
X_test_encoded[missing_cols] <- 0
X_test_encoded <- X_test_encoded[, names(X_train_encoded)]

# Train linear regression model
model <- lm(y_train ~ ., data = X_train_encoded)

# Predict and evaluate
y_pred <- predict(model, X_test_encoded)
mse <- mean((y_test - y_pred)^2)
cat("Mean Squared Error:", mse, "\n")

# Define prediction function
predict_relative_demand <- function(product1, product2) {
  input_data <- data.frame(Product1 = product1, Product2 = product2)
  
  #to match factor levels from training data
  input_data$Product1 <- factor(input_data$Product1, levels = unique(X$Product1))
  input_data$Product2 <- factor(input_data$Product2, levels = unique(X$Product2))
  
  input_encoded <- model.matrix(~ Product1 + Product2 - 1, data = input_data) %>% as.data.frame()
  missing <- setdiff(names(X_train_encoded), names(input_encoded))
  input_encoded[missing] <- 0
  input_encoded <- input_encoded[, names(X_train_encoded)]
  
  diff <- predict(model, newdata = input_encoded)
  
  if (diff > 0) {
    sprintf("%s is more popular by %.2f%%", product1, diff * 100)
  } else {
    sprintf("%s is more popular by %.2f%%", product2, abs(diff) * 100)
  }
}

# Test the function
predict_relative_demand("Japanese Katsu Curry", "Palak Paneer")
