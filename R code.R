# Load Libraries
suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(caret)
  library(pROC)
  library(rpart)
  library(rpart.plot)
  library(glmnet)
  library(Matrix)
  library(ggplot2)
  library(broom)
})


#extra
listings <- read.csv("C:\\Users\\Nandini\\OneDrive\\Documents\\listings.csv")


# --- 1. DATA CLEANING & TYPE CONVERSION ---
# Q4: Cleaning & Conversion
listings$price = as.numeric(gsub("[\\$,]","",listings$price))
listings$bathrooms = as.numeric(sub("([0-9.]+).*","\\1",listings$bathrooms_text))

num_cols = c("host_response_rate","host_acceptance_rate")
for (col in intersect(num_cols, names(listings))) {
  listings[[col]] = suppressWarnings(as.numeric(gsub("%","",listings[[col]])))
}

vars_to_factor = c("host_is_superhost","instant_bookable","has_availability")
for (col in intersect(vars_to_factor, names(listings))) {
  listings[[col]] = factor(listings[[col]])
}

if ("room_type" %in% names(listings)) {
  listings$room_type = factor(listings$room_type)
  levels(listings$room_type) = c(levels(listings$room_type),"Entire home")
  listings$room_type[listings$room_type == "Entire home/apt"] = "Entire home"
  listings$room_type = droplevels(listings$room_type)
}


# --- 2. TRAIN/TEST SPLIT ---
# Q4: Train/Test Split (Chronological)
set.seed(123)

if ("host_since" %in% names(listings)) {
  listings$host_since_date = as.Date(listings$host_since)
  listings = listings %>% arrange(host_since_date)
  
  split_point = floor(0.8 * nrow(listings))
  train_df = listings[1:split_point,]
  test_df = listings[(split_point + 1):nrow(listings),]
} else {
  train_index = sample(seq_len(nrow(listings)), size = floor(0.8 * nrow(listings)))
  train_df = listings[train_index,]
  test_df = listings[-train_index,]
}


# --- 3. PREPROCESSING PARAMETERS (Trained on Training Set) ---
# Q4: Preprocessing Parameters (Imputation & Capping)
impute_cols = intersect(c("bedrooms","beds","bathrooms","number_of_reviews_ltm",
                          "reviews_per_month","minimum_minimum_nights","maximum_minimum_nights",
                          "minimum_maximum_nights","maximum_maximum_nights","minimum_nights_avg_ntm",
                          "maximum_nights_avg_ntm","host_listings_count",
                          "host_response_rate","host_acceptance_rate"), names(train_df))
impute_values = sapply(train_df[,impute_cols], median, na.rm=TRUE)

outlier_cols = intersect(c("price","minimum_nights","bedrooms","beds","bathrooms"), names(train_df))
outlier_bounds = list()
for (col in outlier_cols) {
  Q3 = quantile(train_df[[col]], 0.75, na.rm=TRUE)
  IQR_val = IQR(train_df[[col]], na.rm=TRUE)
  outlier_bounds[[col]] = Q3 + 1.5 * IQR_val
}

# Q3: O_score Min/Max Parameters
O_min_train = min(train_df$number_of_reviews_ltm, na.rm = TRUE)
O_max_train = max(train_df$number_of_reviews_ltm, na.rm = TRUE)


# --- 4. PREPROCESSING & TARGET DEFINITION FUNCTION ---
# Q3/Q4: Target Definition and Preprocessing Function
apply_preprocessing = function(df, impute_values, outlier_bounds,
                               O_min, O_max, is_train) {
  # Imputation
  for (col in names(impute_values)) {
    med = impute_values[col]
    df[[col]][is.na(df[[col]])] = med
  }
  
  # Outlier Capping
  for (col in names(outlier_bounds)) {
    upper_bound = outlier_bounds[[col]]
    df[[col]][df[[col]] > upper_bound] = upper_bound
  }
  
  # New Safe Derived Features (Feature Engineering)
  df = df %>% mutate(capacity = beds + bedrooms + bathrooms)
  
  # 1. Standardize Occupancy component (O_score)
  df = df %>%
    mutate(
      O_score = (number_of_reviews_ltm - O_min) / (O_max - O_min)
    )
  
  # 2. Define the Categorical Target (Q3 classification)
  S_Q3 = 0
  if (is_train) {
    S_Q3 = quantile(df$O_score, 0.75, na.rm = TRUE)
    assign("S_Q3_train", S_Q3, envir = .GlobalEnv)
  } else {
    if (exists("S_Q3_train")) {
      S_Q3 = get("S_Q3_train", envir = .GlobalEnv)
    } else {
      stop("S_Q3_train threshold not found. Run training set preprocessing first.")
    }
  }
  
  df <- df %>%
    mutate(listing_success_new = factor(
      ifelse(df$O_score >= S_Q3, "Good", "Bad"),
      levels = c("Bad", "Good")
    ))
  
  # Clean up the intermediate columns
  df = df %>% select(-O_score)
  
  return(df)
}

# Apply Preprocessing
# Q3/Q4: Apply Preprocessing & Define Target
train_df = apply_preprocessing(train_df, impute_values, outlier_bounds,
                               O_min_train, O_max_train, is_train = TRUE)
test_df = apply_preprocessing(test_df, impute_values, outlier_bounds,
                              O_min_train, O_max_train, is_train = FALSE)

y_train = train_df$listing_success_new
y_test = test_df$listing_success_new


# --- 5. PREPARE INPUTS FOR MODELING ---
safe_predictors = c(
  "accommodates", "minimum_nights", "capacity", "price",
  "host_response_rate", "host_acceptance_rate",
  "instant_bookable", "room_type", "host_is_superhost",
  "bathrooms", "beds", "bedrooms"
)

safe_predictors = intersect(safe_predictors, names(train_df))

char_cols = sapply(train_df[, safe_predictors], is.character)
train_df[, safe_predictors][, char_cols] = lapply(train_df[, safe_predictors][, char_cols], factor)
test_df[, safe_predictors][, char_cols] = lapply(test_df[, safe_predictors][, char_cols], factor)

# Q4: Input Preparation (One-Hot Encoding)
dummies = dummyVars(~ ., data = train_df[, safe_predictors], fullRank = TRUE)
x_train = predict(dummies, newdata = train_df[, safe_predictors])
x_test = predict(dummies, newdata = test_df[, safe_predictors])

x_train_dense = as.matrix(x_train); x_train_dense[!is.finite(x_train_dense)] = 0
x_test_dense = as.matrix(x_test); x_test_dense[!is.finite(x_test_dense)] = 0

x_train_sparse = Matrix(x_train_dense, sparse = TRUE)
x_test_sparse = Matrix(x_test_dense, sparse = TRUE)

x_train_df = as.data.frame(x_train_dense)
x_test_df = as.data.frame(x_test_dense)

# Helper function
# Q6: Model Setup (Threshold & CV Control)
find_optimal_threshold = function(roc_obj) {
  coords_df = coords(roc_obj, "all", ret=c("threshold","specificity","sensitivity"))
  coords_df$J = coords_df$specificity + coords_df$sensitivity - 1
  best_coords = coords_df[which.max(coords_df$J),]
  return(best_coords$threshold)
}

# Setup cross-validation for caret
ctrl = trainControl(method = "repeatedcv", number = 5, repeats = 2, classProbs = TRUE, summaryFunction = twoClassSummary)

results_list = list()


# --- 6. MODEL BUILDING ---

### 1. Elastic Net (Logistic Regression)

## 1A. Baseline Elastic Net
# Q6: Model Building and Tuning
set.seed(123)
cv_lognet_base = cv.glmnet(x_train_sparse, y_train, family = "binomial", alpha = 0.5)
best_lambda_base = cv_lognet_base$lambda.min
lognet_base_model = glmnet(x_train_sparse, y_train, family = "binomial", alpha = 0.5, lambda = best_lambda_base)

lognet_base_probs = predict(lognet_base_model, newx = x_test_sparse, type = "response")
roc_lognet_base = roc(y_test, as.numeric(lognet_base_probs))
auc_lognet_base = auc(roc_lognet_base)
opt_lognet_base_thresh = find_optimal_threshold(roc_lognet_base)
lognet_base_class_opt = factor(ifelse(lognet_base_probs >= opt_lognet_base_thresh, "Good", "Bad"), levels = c("Bad", "Good"))
cm_lognet_base = confusionMatrix(lognet_base_class_opt, y_test)

results_list[["Elastic Net (Baseline)"]] = c(Accuracy = cm_lognet_base$overall["Accuracy"], AUC = auc_lognet_base)

## 1B. Tuned Elastic Net
set.seed(123)
lognet_fit_tuned = train(
  x = x_train_df, y = y_train, method = "glmnet", trControl = ctrl,
  tuneLength = 10, metric = "ROC"
)
best_alpha = lognet_fit_tuned$bestTune$alpha

cv_lognet_tuned = cv.glmnet(x_train_sparse, y_train, family = "binomial", alpha = best_alpha)
lognet_tuned_model = glmnet(x_train_sparse, y_train, family = "binomial", alpha = best_alpha, lambda = cv_lognet_tuned$lambda.min)

lognet_tuned_probs = predict(lognet_tuned_model, newx = x_test_sparse, type = "response")
roc_lognet_tuned = roc(y_test, as.numeric(lognet_tuned_probs))
auc_lognet_tuned = auc(roc_lognet_tuned)
opt_lognet_tuned_thresh = find_optimal_threshold(roc_lognet_tuned)
lognet_tuned_class_opt = factor(ifelse(lognet_tuned_probs >= opt_lognet_tuned_thresh, "Good", "Bad"), levels = c("Bad", "Good"))
cm_lognet_tuned = confusionMatrix(lognet_tuned_class_opt, y_test)

results_list[["Elastic Net (Tuned)"]] = c(Accuracy = cm_lognet_tuned$overall["Accuracy"], AUC = auc_lognet_tuned)

### 2. Decision Tree 

## 2A. Baseline Decision Tree
set.seed(123)
tree_grid_base = expand.grid(cp = seq(0.0001, 0.05, by = 0.001))
tree_fit_base = train(
  x = x_train_df, y = y_train, method = "rpart", trControl = ctrl, tuneGrid = tree_grid_base, metric = "ROC"
)
tree_model_base = tree_fit_base$finalModel
tree_probs_base = predict(tree_model_base, newdata = x_test_df, type = "prob")[,"Good"]
roc_tree_base = roc(y_test, tree_probs_base)
auc_tree_base = auc(roc_tree_base)
opt_tree_base_thresh = find_optimal_threshold(roc_tree_base)
tree_class_opt_base = factor(ifelse(tree_probs_base >= opt_tree_base_thresh, "Good", "Bad"), levels = c("Bad", "Good"))
cm_tree_base = confusionMatrix(tree_class_opt_base, y_test)

results_list[["Decision Tree (Baseline)"]] = c(Accuracy = cm_tree_base$overall["Accuracy"], AUC = auc_tree_base)

## 2B. Tuned Decision Tree
set.seed(123)
tree_grid_tuned = expand.grid(cp = seq(0.0005, 0.01, by = 0.0005))
tree_fit_tuned = train(
  x = x_train_df, y = y_train, method = "rpart", trControl = ctrl, tuneGrid = tree_grid_tuned, metric = "ROC"
)
tree_model_tuned = tree_fit_tuned$finalModel
tree_probs_tuned = predict(tree_model_tuned, newdata = x_test_df, type = "prob")[,"Good"]
roc_tree_tuned = roc(y_test, tree_probs_tuned)
auc_tree_tuned = auc(roc_tree_tuned)
opt_tree_tuned_thresh = find_optimal_threshold(roc_tree_tuned)
tree_class_opt_tuned = factor(ifelse(tree_probs_tuned >= opt_tree_tuned_thresh, "Good", "Bad"), levels = c("Bad", "Good"))
cm_tree_tuned = confusionMatrix(tree_class_opt_tuned, y_test)

results_list[["Decision Tree (Tuned)"]] = c(Accuracy = cm_tree_tuned$overall["Accuracy"], AUC = auc_tree_tuned)


# --- ADDITIONAL CODE FOR REPORTING ---

# Code for Q5: Hypothesis Support (vs. Plots/Descriptive Stats)
cat("\n\n--- CODE FOR Q5: 6 NEW HYPOTHESIS SUPPORT PLOTS/TABLES ---\n")

# H1: Room Type Dominance: 'Entire home' listings will have a significantly higher proportion of success than 'Private room' listings.
# Q5: Hypothesis Support (Descriptive Stats)
cat("\n\n-- H1: Entire Homes have a higher success rate than Private Rooms. --\n")
q5_h1_data = train_df %>%
  filter(room_type %in% c("Entire home", "Private room")) %>%
  group_by(room_type) %>%
  summarise(
    Total_Listings = n(),
    Good_Count = sum(listing_success_new == "Good"),
    Proportion_Good = mean(listing_success_new == "Good") * 100, .groups = 'drop')
print(q5_h1_data)

# H2: Perfect Acceptance: Listings where host_acceptance_rate is 100 will have a higher success rate than others.
# Q5: Hypothesis Support (Descriptive Stats)
cat("\n\n-- H2: Listings with a 100% Host Acceptance Rate are more successful. --\n")
q5_h2_data = train_df %>%
  mutate(Perfect_Acceptance = factor(ifelse(host_acceptance_rate == 100, "100%", "Below 100%"))) %>%
  group_by(Perfect_Acceptance) %>%
  summarise(Proportion_Good = mean(listing_success_new == "Good") * 100, .groups = 'drop')
print(q5_h2_data)

# H3: Short Stay Preference: Listings with a minimum night stay of 1 or 2 nights will have a higher success rate than those with 3+ nights.
# Q5: Hypothesis Support (Descriptive Stats)
cat("\n\n-- H3: Short Minimum Stays (1-2 nights) lead to higher success. --\n")
q5_h3_data = train_df %>%
  mutate(Min_Nights_Group = factor(ifelse(minimum_nights <= 2, "1-2 Nights", "3+ Nights"))) %>%
  group_by(Min_Nights_Group) %>%
  summarise(Proportion_Good = mean(listing_success_new == "Good") * 100, .groups = 'drop')
print(q5_h3_data)

# H4: Superhost/Instant Booking Synergy: Listings that are Superhost (t) AND Instant Bookable (t) will have the highest success rate.
# Q5: Hypothesis Support (Descriptive Stats)
cat("\n\n-- H4: Superhost(t) AND Instant Bookable(t) Synergy - Proportion of Success --\n")
q5_h4_data = train_df %>%
  mutate(Synergy_Group = interaction(host_is_superhost, instant_bookable, sep = " + ")) %>%
  group_by(Synergy_Group) %>%
  summarise(Proportion_Good = mean(listing_success_new == "Good") * 100, .groups = 'drop') %>%
  arrange(desc(Proportion_Good))
print(q5_h4_data)

# H5: Optimal Price Point: The median price of "Good" listings will be in the second or third quartile of the overall price distribution.
# Q5: Hypothesis Support (Descriptive Stats)
cat("\n\n-- H5: Optimal Price Point Analysis --\n")
# Calculate overall price quartiles on the training set
price_quartiles = quantile(train_df$price, probs = c(0.25, 0.50, 0.75), na.rm = TRUE)
cat(paste("Overall Price Quartiles (Q1, Q2, Q3): $", paste(round(price_quartiles, 2), collapse=", $"), "\n", sep=""))

q5_h5_data = train_df %>%
  mutate(Price_Quartile = cut(price,
                              breaks = c(0, price_quartiles[1], price_quartiles[2], price_quartiles[3], Inf),
                              labels = c("Q1 (Low)", "Q2", "Q3", "Q4 (High)"),
                              include.lowest = TRUE)) %>%
  group_by(Price_Quartile, listing_success_new) %>%
  summarise(Count = n(), .groups = 'drop_last') %>%
  mutate(Proportion = Count / sum(Count) * 100) %>%
  filter(listing_success_new == "Good") %>%
  select(-listing_success_new, -Count)

print(q5_h5_data)

# H6: Bathroom Advantage: The median bathrooms count will be significantly higher for "Good" listings compared to "Bad" listings.
# Q5: Hypothesis Support (Descriptive Stats)
cat("\n\n-- H6: Median Bathrooms (Good vs Bad) --\n")
q5_h6_stats = train_df %>%
  group_by(listing_success_new) %>%
  summarise(
    Mean_Bathrooms = mean(bathrooms, na.rm = TRUE),
    Median_Bathrooms = median(bathrooms, na.rm = TRUE)
  )
print(q5_h6_stats)

# --- 3. PRUNE AND PLOT FOR INTERPRETABILITY (The final requested output) ---
cat("\nPruning and Plotting Final Interpretable Tree...\n")
# FIX: Replaced undefined 'tree_model_retuned' with 'tree_model_tuned'
pruned_tree = prune(tree_model_tuned, cp = 0.002) 

# 1. Save current graphics parameters safely
original_par = par(no.readonly = TRUE) 
# Filter out non-restorable parameters like "pin", "pty", and "mfrow"
original_par = original_par[!names(original_par) %in% c("pin", "pty", "mfrow", "fig", "usr")]

# 2. Set small margins: c(bottom, left, top, right) to prevent the figure margins too large error
par(mar=c(1, 1, 1, 1)) 

# Plotting the pruned tree with key decision metrics displayed
rpart.plot(pruned_tree,
           type = 2,          
           extra = 104,       
           under = TRUE,     
           cex = 0.65,        
           main = "Pruned Decision Tree (CP=0.002) for Interpretation")

# 3. Restore original graphics parameters safely
par(original_par) 

# Clean up global variable
if (exists("S_Q3_train")) {
  rm("S_Q3_train", envir = .GlobalEnv)
}