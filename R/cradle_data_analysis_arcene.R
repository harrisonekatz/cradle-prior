###############################################################################
# arcene_analysis.R
#
# Purpose:
#   Demonstrate the Cradle, Horseshoe, and Lasso priors on the Arcene dataset,
#   illustrating high-dimensional logistic regression with global–local shrinkage.
#   The script downloads (or reads) Arcene data, splits into train/test,
#   fits all three models in Stan, evaluates performance via classification metrics,
#   and produces summary figures.
#
#   NOTE: Arcene has ~10,000 features and only 100 training examples, which is
#         ideal for showcasing moderate-signal detection in a high-dimensional
#         environment.
###############################################################################
install.packages("qs")
# ========================
# 0) Libraries & Options
# ========================
if (!requireNamespace("rstan", quietly=TRUE)) {
  install.packages("rstan", repos="https://cloud.r-project.org")
}
if (!requireNamespace("loo", quietly=TRUE)) {
  install.packages("loo", repos="https://cloud.r-project.org")
}
if (!requireNamespace("pROC", quietly=TRUE)) {
  install.packages("pROC", repos="https://cloud.r-project.org")
}
if (!requireNamespace("ggplot2", quietly=TRUE)) {
  install.packages("ggplot2", repos="https://cloud.r-project.org")
}


library(qs)
library(rstan)
library(loo)
library(pROC)
library(ggplot2)

rstan_options(auto_write = TRUE)  # Speed up Stan usage
options(mc.cores = parallel::detectCores())

###############################################################################
# 1) Download / Load Arcene Data
###############################################################################
# Arcene data is available through the UCI ML repository:
#   https://archive.ics.uci.edu/ml/datasets/Arcene
# Typically, we have these files:
#   arcene_train.data, arcene_train.labels,
#   arcene_valid.data, arcene_valid.labels
#
# For demonstration, assume these files are locally available in a path or
# already downloaded. If needed, adjust to read them from a URL or other path.
#
# Arcene has 100 training samples (50 positives, 50 negatives)
# and an additional 100 "validation" samples. Each sample has ~10,000 features.
###############################################################################

arcene_data_path <- "helicon/ARCENE"  # CHANGE this to your local path

train_data_file   <- file.path(arcene_data_path, "arcene_train.data")
train_label_file  <- file.path(arcene_data_path, "arcene_train.labels")
valid_data_file   <- file.path(arcene_data_path, "arcene_valid.data")
valid_label_file  <- file.path("helicon/arcene_valid.labels")

# Reading the data
arcene_train_data  <- as.matrix(read.table(train_data_file))
arcene_train_label <- as.numeric(read.table(train_label_file)[,1])  # numeric vector 1/-1 or 1/0
arcene_valid_data  <- as.matrix(read.table(valid_data_file))
arcene_valid_label <- as.numeric(read.table(valid_label_file)[,1])

# Convert labels to 0/1 if needed (Arcene sometimes has +1/-1)
arcene_train_label[arcene_train_label < 1] <- 0
arcene_valid_label[arcene_valid_label < 1] <- 0

# Typically, Arcene data has dimension: 100 x 10000 for training,
# and 100 x 10000 for validation
cat("Arcene Train dims:", dim(arcene_train_data), "\n")
cat("Arcene Valid dims:", dim(arcene_valid_data), "\n")

# Combine train + valid if desired, or keep separate
# For demonstration, let's create a single dataset (X, y), then do our own splits.
# Here, we'll do a simple approach: keep official train as 'train set' and
# official valid as 'test set'.

X_train_raw <- arcene_train_data
y_train     <- arcene_train_label
X_test_raw  <- arcene_valid_data
y_test      <- arcene_valid_label

# OPTIONAL: Scale features
# Because many features in Arcene can vary in scale or be constant, we typically
# standardize them. But watch out for constant features that cause NAs after scaling.
#
# We'll do a simple z-scale on the training set. We remove zero-variance features
# as well.

nzv <- apply(X_train_raw, 2, function(col) sd(col) == 0)
X_train_filtered <- X_train_raw[, !nzv]
X_test_filtered  <- X_test_raw[,  !nzv]

# scale columns
train_means <- apply(X_train_filtered, 2, mean)
train_sds   <- apply(X_train_filtered, 2, sd)

X_train <- sweep(X_train_filtered, 2, train_means, "-")
X_train <- sweep(X_train, 2, train_sds, "/")
X_test  <- sweep(X_test_filtered, 2, train_means, "-")
X_test  <- sweep(X_test, 2, train_sds, "/")

# Replace any NA with 0 if there's zero-variance after filtering (rare edge case)
X_train[is.na(X_train)] <- 0
X_test[is.na(X_test)]   <- 0

cat("Final train dims after filtering:", dim(X_train), "\n")
cat("Final test dims after filtering:",  dim(X_test), "\n")

n_train <- nrow(X_train)
p       <- ncol(X_train)
n_test  <- nrow(X_test)

###############################################################################
# 2) Define Stan Models
###############################################################################
# We'll do a logistic regression with three priors:
#   1) Cradle
#   2) Horseshoe
#   3) Lasso
#
# We'll produce a "generated quantities" block to get log_lik for each observation.
# Then we can compute ELPD or other metrics. For classification, we'll also
# get predicted probabilities and classify 0/1 using a 0.5 cutoff.
###############################################################################

cradle_code <- "
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n,p] X;
array[n] int<lower=0, upper=1> y;
}
parameters {
  vector[p] beta;
  real<lower=0> tau;
  vector<lower=0>[p] lambda;
  real<lower=0> alpha;
}
model {
  // logistic likelihood
  for(i in 1:n) {
    y[i] ~ bernoulli_logit(dot_product(row(X,i), beta));
  }
  // priors
  tau ~ cauchy(0,1);
  alpha ~ gamma(2,2);        // example hyperprior
  lambda ~ exponential(alpha);
  for(j in 1:p) {
    beta[j] ~ normal(0, tau*lambda[j]);
  }
}
generated quantities {
  vector[n] log_lik;
  vector[n] yhat_prob;
  for(i in 1:n) {
    real lp = dot_product(row(X,i), beta);
    log_lik[i] = bernoulli_logit_lpmf(y[i] | lp);
    yhat_prob[i] = inv_logit(lp);
  }
}
"

horseshoe_code <- "
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n,p] X;
array[n] int<lower=0, upper=1> y;
}
parameters {
  vector[p] beta;
  real<lower=0> tau;
  vector<lower=0>[p] lambda;
}
model {
  // logistic likelihood
  for(i in 1:n) {
    y[i] ~ bernoulli_logit(dot_product(row(X,i), beta));
  }
  // horseshoe priors
  tau ~ cauchy(0,1);
  lambda ~ cauchy(0,1);
  for(j in 1:p) {
    beta[j] ~ normal(0, tau*lambda[j]);
  }
}
generated quantities {
  vector[n] log_lik;
  vector[n] yhat_prob;
  for(i in 1:n) {
    real lp = dot_product(row(X,i), beta);
    log_lik[i] = bernoulli_logit_lpmf(y[i] | lp);
    yhat_prob[i] = inv_logit(lp);
  }
}
"

lasso_code <- "
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n,p] X;
array[n] int<lower=0, upper=1> y;
}
parameters {
  vector[p] beta;
  real<lower=0> lambda;
  vector<lower=0>[p] phi;
}
model {
  // logistic likelihood
  for(i in 1:n) {
    y[i] ~ bernoulli_logit(dot_product(row(X,i), beta));
  }
  // lasso priors
  lambda ~ cauchy(0,1);
  phi ~ exponential(lambda);
  for(j in 1:p) {
    beta[j] ~ normal(0, phi[j]);
  }
}
generated quantities {
  vector[n] log_lik;
  vector[n] yhat_prob;
  for(i in 1:n) {
    real lp = dot_product(row(X,i), beta);
    log_lik[i] = bernoulli_logit_lpmf(y[i] | lp);
    yhat_prob[i] = inv_logit(lp);
  }
}
"

###############################################################################
# 3) Compile Models
###############################################################################
cat("\nCompiling Stan models...\n")
model_cradle    <- stan_model(model_code = cradle_code)
model_horseshoe <- stan_model(model_code = horseshoe_code)
model_lasso     <- stan_model(model_code = lasso_code)
cat("Done compiling.\n")

###############################################################################
# 4) Fit Models on Arcene Training Data
###############################################################################
fit_stan_model <- function(stan_model, X, y,
                           chains=8, iter=5000, warmup=2500,
                           seed=123, control=list(adapt_delta=0.85, max_treedepth=12),
                           refresh=200,init=.5) {
  data_list <- list(
    n = nrow(X),
    p = ncol(X),
    X = X,
    y = y
  )
  fit <- sampling(
    object=stan_model,
    data=data_list,
    chains=chains,
    iter=iter,
    warmup=warmup,
    seed=seed,
    control=control,
    refresh=refresh
  )
  return(fit)
}

# # We'll do single-chain to keep it quicker, but you can set chains=4 if time permits.
# cat("\nFitting Cradle model...\n")
# fit_cradle <- fit_stan_model(model_cradle, X_train, y_train,
#                              chains=8, iter=5000, warmup=2500,
#                              seed=123, control=list(adapt_delta=0.85, max_treedepth=12))
# cat("\nFitting Horseshoe model...\n")
# fit_horse <- fit_stan_model(model_horseshoe, X_train, y_train,
#                             chains=4, iter=2000, warmup=1000, seed=2222)
# cat("\nFitting Lasso model...\n")
# fit_lasso <- fit_stan_model(model_lasso, X_train, y_train,
#                             chains=4, iter=2000, warmup=1000, seed=3333)

setwd("~")
fit_cradle_file <- "fit_cradle.RDS"
if (!file.exists(fit_cradle_file)) {
  cat("[4] Fitting Cradle...\n")
  fit_cradle <- fit_stan_model(model_cradle, X_train, y_train)
  #saveRDS(fit_cradle, fit_cradle_file)
  qsave(fit_cradle, fit_cradle_file)
} else {
  cat("[4] Skipping Cradle fit (already found fit_cradle.RDS)\n")
}

fit_cradle <- qread(fit_cradle_file)

# --- Fit Horseshoe ---
fit_horse_file <- "fit_horseshoe.RDS"
if (!file.exists(fit_horse_file)) {
  cat("[4] Fitting Horseshoe...\n")
  fit_horseshoe <- fit_stan_model(model_horseshoe, X_train, y_train)
  qsave(fit_horseshoe, fit_horse_file)
} else {
  cat("[4] Skipping Horseshoe fit (already found fit_horseshoe.RDS)\n")
}
fit_horseshoe <- qread(fit_horse_file)

# --- Fit Lasso ---
fit_lasso_file <- "fit_lasso.RDS"
if (!file.exists(fit_lasso_file)) {
  cat("[4] Fitting Lasso...\n")
  fit_lasso <- fit_stan_model(model_lasso, X_train, y_train)
  qsave(fit_lasso, fit_lasso_file)
} else {
  cat("[4] Skipping Lasso fit (already found fit_lasso.RDS)\n")
}
fit_lasso <- qread(fit_lasso_file)




###############################################################################
# 5) Extract Posterior Summaries & Evaluate on TRAIN set
###############################################################################
extract_stan_summaries <- function(fit) {
  # get posterior draws for log_lik, yhat_prob
  log_lik_mat  <- rstan::extract(fit, "log_lik")$log_lik  # (iterations x n)
  yhat_prob_mat<- rstan::extract(fit, "yhat_prob")$yhat_prob

  # average predicted prob across posterior
  mean_prob <- colMeans(yhat_prob_mat)  # length n
  return(list(
    log_lik_mat = log_lik_mat,
    mean_prob   = mean_prob
  ))
}

summ_cradle <- extract_stan_summaries(fit_cradle)
summ_horse  <- extract_stan_summaries(fit_horseshoe)
summ_lasso  <- extract_stan_summaries(fit_lasso)

train_prob_cradle <- summ_cradle$mean_prob
train_prob_horse  <- summ_horse$mean_prob
train_prob_lasso  <- summ_lasso$mean_prob

# Evaluate classification on training set
# use threshold 0.5
pred_cradle_train <- ifelse(train_prob_cradle > 0.5, 1, 0)
pred_horse_train  <- ifelse(train_prob_horse  > 0.5, 1, 0)
pred_lasso_train  <- ifelse(train_prob_lasso  > 0.5, 1, 0)

acc_cradle_train <- mean(pred_cradle_train == y_train)
acc_horse_train  <- mean(pred_horse_train  == y_train)
acc_lasso_train  <- mean(pred_lasso_train  == y_train)

cat("\n--- Training Accuracy ---\n")
cat("Cradle:", acc_cradle_train, "\n")
cat("Horse :", acc_horse_train,  "\n")
cat("Lasso :", acc_lasso_train,  "\n")

# AUC
auc_cradle_train <- pROC::auc(y_train, train_prob_cradle)
auc_horse_train  <- pROC::auc(y_train, train_prob_horse)
auc_lasso_train  <- pROC::auc(y_train, train_prob_lasso)

cat("\n--- Training AUC ---\n")
cat("Cradle:", auc_cradle_train, "\n")
cat("Horse :", auc_horse_train,  "\n")
cat("Lasso :", auc_lasso_train,  "\n")

###############################################################################
# 6) Prediction on TEST set
###############################################################################
# We'll do "posterior predictive" in a simplified sense:
# get posterior draws of beta from the fit and compute X_test %*% beta
# for each iteration. Then we average.

extract_posterior_betas <- function(fit) {
  # returns a matrix (iterations x p)
  rstan::extract(fit, "beta")$beta
}

posterior_betas_cradle <- extract_posterior_betas(fit_cradle)
posterior_betas_horse  <- extract_posterior_betas(fit_horseshoe)
posterior_betas_lasso  <- extract_posterior_betas(fit_lasso)

test_prob_cradle_mat <- 1 / (1 + exp(- X_test %*% t(posterior_betas_cradle)))
test_prob_horse_mat  <- 1 / (1 + exp(- X_test %*% t(posterior_betas_horse)))
test_prob_lasso_mat  <- 1 / (1 + exp(- X_test %*% t(posterior_betas_lasso)))

# average across posterior draws
test_prob_cradle <- rowMeans(test_prob_cradle_mat)
test_prob_horse  <- rowMeans(test_prob_horse_mat)
test_prob_lasso  <- rowMeans(test_prob_lasso_mat)

# classification
pred_cradle_test <- ifelse(test_prob_cradle > 0.5, 1, 0)
pred_horse_test  <- ifelse(test_prob_horse  > 0.5, 1, 0)
pred_lasso_test  <- ifelse(test_prob_lasso  > 0.5, 1, 0)

acc_cradle_test <- mean(pred_cradle_test == y_test)
acc_horse_test  <- mean(pred_horse_test  == y_test)
acc_lasso_test  <- mean(pred_lasso_test  == y_test)

cat("\n--- Test Accuracy ---\n")
cat("Cradle:", acc_cradle_test, "\n")
cat("Horse :", acc_horse_test,  "\n")
cat("Lasso :", acc_lasso_test,  "\n")

auc_cradle_test <- pROC::auc(y_test, test_prob_cradle)
auc_horse_test  <- pROC::auc(y_test, test_prob_horse)
auc_lasso_test  <- pROC::auc(y_test, test_prob_lasso)

cat("\n--- Test AUC ---\n")
cat("Cradle:", auc_cradle_test, "\n")
cat("Horse :", auc_horse_test,  "\n")
cat("Lasso :", auc_lasso_test,  "\n")

###############################################################################
# 7) Figures for Analysis
###############################################################################
# We'll produce some simple plots:
#   - Distribution of predicted probabilities on test set
#   - ROC curves for test set
#   - Possibly a bar chart of accuracy

# 7a) Distribution of predicted probabilities (test)
df_test_prob <- data.frame(
  Cradle = test_prob_cradle,
  Horse  = test_prob_horse,
  Lasso  = test_prob_lasso,
  Label  = factor(y_test, levels=c(0,1), labels=c("Negative","Positive"))
)

# We'll pivot this for ggplot2
library(reshape2)
df_melt <- melt(df_test_prob, id.vars="Label", variable.name="Model", value.name="Prob")

p_density <- ggplot(df_melt, aes(x=Prob, fill=Label)) +
  geom_density(alpha=0.4) +
  facet_wrap(~Model, ncol=1) +
  theme_bw() +
  labs(title="Test Probabilities by Model (Arcene)", x="Predicted Probability", y="Density")
print(p_density)

# 7b) ROC curves
plot_roc <- function(prob, label, model_name) {
  r <- pROC::roc(label, prob)
  df_roc <- data.frame(
    TPR = r$sensitivities,
    FPR = 1 - r$specificities,
    Model = model_name
  )
  return(df_roc)
}

roc_cradle <- plot_roc(test_prob_cradle, y_test, "Cradle")
roc_horse  <- plot_roc(test_prob_horse,  y_test, "Horseshoe")
roc_lasso  <- plot_roc(test_prob_lasso,  y_test, "Lasso")

df_roc_all <- rbind(roc_cradle, roc_horse, roc_lasso)

p_roc <- ggplot(df_roc_all, aes(x=FPR, y=TPR, color=Model)) +
  geom_line(size=1) +
  geom_abline(slope=1, intercept=0, linetype="dashed", color="gray") +
  theme_bw() +
  labs(title="ROC Curves on Arcene Test Set", x="False Positive Rate", y="True Positive Rate")
print(p_roc)

# 7c) Bar chart of test accuracy
df_acc <- data.frame(
  Model    = c("Cradle","Horseshoe","Lasso"),
  Accuracy = c(acc_cradle_test, acc_horse_test, acc_lasso_test),
  AUC      = c(as.numeric(auc_cradle_test), as.numeric(auc_horse_test), as.numeric(auc_lasso_test))
)

p_bar_acc <- ggplot(df_acc, aes(x=Model, y=Accuracy, fill=Model)) +
  geom_bar(stat="identity", width=0.6) +
  ylim(0,1) +
  theme_bw() +
  labs(title="Test Accuracy by Model", x="", y="Accuracy") +
  theme(legend.position="none")
print(p_bar_acc)

###############################################################################
# 8) Done
###############################################################################
cat("\nAll done. Figures have been plotted, and metrics printed.\n")
