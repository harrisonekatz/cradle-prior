
# -------------------------
install.packages("BiocManager")

BiocManager::install("multtest")
library(multtest)
library(rstan)
library(pROC)
library(VennDiagram)
library(caret)
library(dplyr)
library(ggplot2)

# For faster Stan compiles
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# -------------------------
# (1) Load & Subset Golub Data
# -------------------------
cat("\n[1] Loading Golub data...\n")
data(golub)
X.raw <- t(golub)   # 72 × 7129
y.raw <- golub.cl   # length 72, values c(ALL=1, AML=2)
print(table(golub.cl))
# Convert outcome to 0/1 for logistic regression: AML=1, ALL=0
y <- as.numeric(y.raw == 1)  # if 2=AML, then AML->1, ALL->0

cat("Check overall distribution of y:\n")
print(table(y))  # Should see something like 47 zeros, 25 ones (depending on the subset)

# Keep top 1000 genes by variance
gene.vars <- apply(X.raw, 2, var)
top.idx   <- order(gene.vars, decreasing=TRUE)[1:1000]
X.sub     <- X.raw[, top.idx]

# Scale features
X.centered <- scale(X.sub, center=TRUE, scale=TRUE)
n <- nrow(X.centered)
p <- ncol(X.centered)
cat("Data dimension after subsetting:", n, "samples ×", p, "genes\n")

# -------------------------
# (2) Define Stan Models (same as before)
# -------------------------
cradle_code <- "
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n,p] X;
  int<lower=0,upper=1> y[n];
}
parameters {
  vector[p] beta;
  real<lower=0> tau;
  vector<lower=0>[p] lambda;
  real<lower=0> alpha;
}
model {
  for (i in 1:n) {
    y[i] ~ bernoulli_logit(dot_product(row(X,i), beta));
  }
  tau ~ cauchy(0,1);
  lambda ~ exponential(alpha);
  alpha ~ gamma(2,2);
  for (j in 1:p) {
    beta[j] ~ normal(0, tau*lambda[j]);
  }
}
"
horseshoe_code <- "
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n,p] X;
  int<lower=0,upper=1> y[n];
}
parameters {
  vector[p] beta;
  real<lower=0> tau;
  vector<lower=0>[p] lambda;
}
model {
  for (i in 1:n) {
    y[i] ~ bernoulli_logit(dot_product(row(X,i), beta));
  }
  tau ~ cauchy(0,1);
  lambda ~ cauchy(0,1);
  for (j in 1:p) {
    beta[j] ~ normal(0, tau*lambda[j]);
  }
}
"
lasso_code <- "
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n,p] X;
  int<lower=0,upper=1> y[n];
}
parameters {
  vector[p] beta;
  real<lower=0> lambda;
  vector<lower=0>[p] phi;
}
model {
  for (i in 1:n) {
    y[i] ~ bernoulli_logit(dot_product(row(X,i), beta));
  }
  lambda ~ cauchy(0,1);
  phi ~ exponential(lambda);
  for (j in 1:p) {
    beta[j] ~ normal(0, phi[j]);
  }
}
"

# -------------------------
# (3) Compile Models
# -------------------------
cat("\n[3] Compiling Stan models...\n")
model_cradle    <- stan_model(model_code = cradle_code)
model_horseshoe <- stan_model(model_code = horseshoe_code)
model_lasso     <- stan_model(model_code = lasso_code)
cat("Done compiling.\n")

# -------------------------
# Helper: Fit a pre-compiled model
# -------------------------
fit_stan_model <- function(stan_model, X, y, 
                           chains=4,
                           iter=3000, 
                           warmup=1500, 
                           seed=1234,
                           control=list(adapt_delta=0.95, max_treedepth=12),
                           refresh=500) {
  # if training set has only one class, skip
  if (length(unique(y)) < 2) {
    warning("Training set has single class => skipping Stan fit.")
    return(NULL)
  }
  datlist <- list(n=nrow(X), p=ncol(X), X=X, y=y)
  sm_fit <- sampling(
    stan_model, 
    data=datlist,
    chains=chains, iter=iter, warmup=warmup, seed=seed,
    control=control, refresh=refresh
  )
  return(sm_fit)
}

extract_betas <- function(fit) {
  if (is.null(fit)) return(NULL)
  post <- rstan::extract(fit, "beta")$beta
  if (is.null(post)) return(NULL)
  apply(post, 2, mean)
}



K <- 10 
set.seed(999)
folds <- createFolds(factor(y), k=K, list=FALSE, returnTrain=FALSE)
cv_dir <- "cv_results"
if(!dir.exists(cv_dir)) dir.create(cv_dir)

run_cv_fold <- function(k, X, y,
                        model_cradle, model_horseshoe, model_lasso,
                        outdir=cv_dir,
                        chains=4, iter=3000, warmup=1500) {
  fold_file <- file.path(outdir, paste0("cv_fold", k, "_results.rds"))
  if(file.exists(fold_file)) {
    cat("Fold", k, "already done. Skipping.\n")
    return(NULL)
  }
  
  # Indices
  test_idx  <- which(folds == k)
  train_idx <- setdiff(seq_len(n), test_idx)
  
  # Subsets
  Xtrain <- X[train_idx, , drop=FALSE]
  ytrain <- y[train_idx]
  Xtest  <- X[test_idx, , drop=FALSE]
  ytest  <- y[test_idx]
  
  cat("\n--- Running CV fold", k, "---\n")
  cat(" Fold:", k, 
      " training size:", length(train_idx),
      ", test size:", length(test_idx), "\n")
  cat("  Classes in train:\n")
  print(table(ytrain))
  cat("  Classes in test:\n")
  print(table(ytest))
  
  # Fit each model
  fit_c <- fit_stan_model(model_cradle,    Xtrain, ytrain, chains=chains, iter=iter, warmup=warmup)
  fit_h <- fit_stan_model(model_horseshoe, Xtrain, ytrain, chains=chains, iter=iter, warmup=warmup)
  fit_l <- fit_stan_model(model_lasso,     Xtrain, ytrain, chains=chains, iter=iter, warmup=warmup)
  
  beta_c <- extract_betas(fit_c)
  beta_h <- extract_betas(fit_h)
  beta_l <- extract_betas(fit_l)
  
  # If all fits were NULL, skip
  if (is.null(beta_c) && is.null(beta_h) && is.null(beta_l)) {
    warning(paste("Fold", k, ": All Stan fits returned NULL => skipping."))
    fold_res <- data.frame(
      fold        = k,
      acc_cradle  = NA, auc_cradle  = NA,
      acc_horse   = NA, auc_horse   = NA,
      acc_lasso   = NA, auc_lasso   = NA
    )
    saveRDS(fold_res, fold_file)
    return(invisible(fold_res))
  }
  
  # Predictions
  pred_prob <- function(b, Xmat) {
    if (is.null(b)) return(rep(NA, nrow(Xmat)))
    1 / (1 + exp(- (Xmat %*% b)))
  }
  p_c <- pred_prob(beta_c, Xtest)
  p_h <- pred_prob(beta_h, Xtest)
  p_l <- pred_prob(beta_l, Xtest)
  
  pred_c <- ifelse(p_c>0.5, 1, 0)
  pred_h <- ifelse(p_h>0.5, 1, 0)
  pred_l <- ifelse(p_l>0.5, 1, 0)
  
  acc_c <- mean(pred_c == ytest, na.rm=TRUE)
  acc_h <- mean(pred_h == ytest, na.rm=TRUE)
  acc_l <- mean(pred_l == ytest, na.rm=TRUE)
  
  # AUC if test set has both classes
  if (length(unique(ytest))==2) {
    auc_c <- pROC::auc(ytest, p_c)
    auc_h <- pROC::auc(ytest, p_h)
    auc_l <- pROC::auc(ytest, p_l)
  } else {
    auc_c <- NA
    auc_h <- NA
    auc_l <- NA
    warning(sprintf("Fold %d: test set single-class => AUC=NA.", k))
  }
  
  fold_res <- data.frame(
    fold = k,
    acc_cradle = acc_c, auc_cradle = as.numeric(auc_c),
    acc_horse  = acc_h, auc_horse  = as.numeric(auc_h),
    acc_lasso  = acc_l, auc_lasso  = as.numeric(auc_l)
  )
  
  saveRDS(fold_res, fold_file)
  cat("Fold", k, "results saved to", fold_file, "\n")
  
  invisible(fold_res)
}




# Run folds
for(k in 1:K) {
  run_cv_fold(k, X.centered, y,
              model_cradle, model_horseshoe, model_lasso,
              outdir=cv_dir,
              chains=4, iter=3000, warmup=1500)
}

cat("\nAll CV folds completed.\n")
cv_dir <- "cv_results"

# Combine
cv_files <- list.files(cv_dir, pattern="^cv_fold", full.names=TRUE)
if (length(cv_files)==0) {
  stop("No folds completed successfully.")
}
cv_all <- do.call(rbind, lapply(cv_files, readRDS))
cat("Combined CV results:\n")
print(cv_all)

# Summaries
cv_summary <- data.frame(
  Method = c("Cradle","Horseshoe","Lasso"),
  Mean_Accuracy = c(mean(cv_all$acc_cradle, na.rm=TRUE),
                    mean(cv_all$acc_horse,  na.rm=TRUE),
                    mean(cv_all$acc_lasso,  na.rm=TRUE)),
  Mean_AUC = c(mean(cv_all$auc_cradle, na.rm=TRUE),
               mean(cv_all$auc_horse,  na.rm=TRUE),
               mean(cv_all$auc_lasso,  na.rm=TRUE))
)
cat("\n--- CV Summary ---\n")
print(cv_summary)

# combine results
cv_files <- list.files(cv_dir, pattern="^cv_fold", full.names=TRUE)
if(length(cv_files)==0) {
  stop("No folds completed successfully.")
}
cv_all <- do.call(rbind, lapply(cv_files, readRDS))
cat("\nCombined CV results:\n")
print(cv_all)

# summarize
cv_summary <- data.frame(
  Method = c("Cradle","Horseshoe","Lasso"),
  Mean_Accuracy = c(mean(cv_all$acc_cradle, na.rm=TRUE),
                    mean(cv_all$acc_horse,  na.rm=TRUE),
                    mean(cv_all$acc_lasso,  na.rm=TRUE)),
  Mean_AUC = c(mean(cv_all$auc_cradle, na.rm=TRUE),
               mean(cv_all$auc_horse,  na.rm=TRUE),
               mean(cv_all$auc_lasso,  na.rm=TRUE))
)
cat("\n--- CV Summary ---\n")
print(cv_summary)

###############################################################################
# (5) Visualization: Boxplots of Accuracy and AUC
###############################################################################
library(reshape2)

# Melt the fold-by-fold data for ggplot
df_long_acc <- melt(
  cv_all[, c("fold","acc_cradle","acc_horse","acc_lasso")],
  id.vars="fold",
  variable.name="Model",
  value.name="Accuracy"
)
df_long_auc <- melt(
  cv_all[, c("fold","auc_cradle","auc_horse","auc_lasso")],
  id.vars="fold",
  variable.name="Model",
  value.name="AUC"
)

# rename factor levels for clarity
df_long_acc$Model <- factor(df_long_acc$Model,
                            levels=c("acc_cradle","acc_horse","acc_lasso"),
                            labels=c("Cradle","Horseshoe","Lasso"))
df_long_auc$Model <- factor(df_long_auc$Model,
                            levels=c("auc_cradle","auc_horse","auc_lasso"),
                            labels=c("Cradle","Horseshoe","Lasso"))

# Boxplot of Accuracy by model
p_acc <- ggplot(df_long_acc, aes(x=Model, y=Accuracy, fill=Model)) +
  geom_boxplot(alpha=0.6) +
  theme_bw() +
  labs(title="K-Fold Accuracy by Model (Golub)", y="Accuracy", x="") +
  theme(legend.position="none")

# Boxplot of AUC by model
p_auc <- ggplot(df_long_auc, aes(x=Model, y=AUC, fill=Model)) +
  geom_boxplot(alpha=0.6) +
  theme_bw() +
  labs(title="K-Fold AUC by Model (Golub)", y="AUC", x="") +
  theme(legend.position="none")

# print or arrange
print(p_acc)
print(p_auc)

cat("\nVisualization complete. Script finished.\n")
