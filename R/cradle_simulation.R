# --------------------------------------------------------
# run_compare_priors_in_batches.R
# --------------------------------------------------------
# This script does a more extensive simulation study for Cradle, Horseshoe, and Lasso.
# It is designed to be sourced (e.g., source("run_compare_priors_in_batches.R")).
# We run in "batches" (chunks of scenarios) to avoid crashing or hogging memory,
# and save output after each chunk so we don't lose progress if R stops mid-run.

# ------------------------------
# 0. Load Libraries
# ------------------------------
library(rstan)
library(dplyr)
library(ggplot2)

# For faster Stan compilation
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# ------------------------------
# 1) Compile each Stan model
# ------------------------------
# Make sure your .stan files are in the "stan/" folder or adjust the paths below:
cat("Compiling Stan models...\n")
cradle_model    <- stan_model("stan/cradle_regression.stan")
horseshoe_model <- stan_model("stan/horseshoe_regression.stan")
lasso_model     <- stan_model("stan/lasso_regression.stan")

# ------------------------------
# 2) Create an extensive parameter grid
# ------------------------------
# We'll vary:
#   N in {100, 200, 400}
#   p in {200, 500, 1000}
#   s in {10, 20, 40} (number of nonzero signals)
#   beta_signal in {1.0, 1.5, 2.0, 3.0}
#   replicates = 10
# => 3 x 3 x 3 x 4 x 10 = 1080 total scenarios
# Adjust these as needed for your computational resources.
cat("Building parameter grid...\n")
big_grid <- expand.grid(
  N = c(100, 200, 400),
  p = c(200, 500, 1000),
  s = c(10, 20, 40),
  beta_signal = c(1.0, 1.5, 2.0, 3.0),
  repl = 1:10
)
cat("Total scenarios:", nrow(big_grid), "\n")

# ------------------------------
# 3) Simulation + Fit Functions
# ------------------------------
simulate_data <- function(N, p, s, beta_signal = 2, sigma_true = 1) {
  X <- matrix(rnorm(N * p), nrow = N, ncol = p)
  beta_true <- numeric(p)
  idx <- sample.int(p, s)
  beta_true[idx] <- rnorm(s, mean = 0, sd = beta_signal)
  y <- X %*% beta_true + rnorm(N, 0, sigma_true)
  list(X = X, y = y, beta_true = beta_true)
}

fit_model <- function(model, X, y, iter = 500, warmup = 250, chains = 2) {
  # Create a data list for Stan
  standat <- list(N = nrow(X), p = ncol(X), X = X, y = as.vector(y))
  sampling(model,
           data = standat,
           chains = chains,
           iter = iter,
           warmup = warmup,
           refresh = 0,          # reduce console messages
           save_warmup = FALSE,  # don't store warmup draws
           pars = c("beta"),     # only store 'beta'
           include = TRUE)
}

evaluate_performance <- function(beta_hat, beta_true, threshold = 0.1) {
  # MSE
  mse <- mean((beta_hat - beta_true)^2)
  # TPR and FDR
  true_idx <- which(abs(beta_true) > 1e-8)
  est_idx  <- which(abs(beta_hat) > threshold)
  
  tpr <- length(intersect(true_idx, est_idx)) / length(true_idx)
  zero_idx <- setdiff(seq_along(beta_true), true_idx)
  n_fp <- length(intersect(zero_idx, est_idx))
  fdr <- if (length(est_idx) == 0) 0 else n_fp / length(est_idx)
  
  data.frame(mse = mse, tpr = tpr, fdr = fdr)
}

# ------------------------------
# 4) Run a Single Chunk
# ------------------------------
# Each chunk is a subset of the big_grid (rows i..j)
run_chunk <- function(chunk_df,
                      chunk_id = 1,
                      outdir = "batch_results",
                      cradle_mod = cradle_model,
                      horseshoe_mod = horseshoe_model,
                      lasso_mod = lasso_model) {
  
  # Ensure the output directory exists
  if(!dir.exists(outdir)) dir.create(outdir)
  
  # File to save results for this chunk
  chunk_file <- file.path(outdir, paste0("sim_results_chunk_", chunk_id, ".rds"))
  
  # If the file already exists, skip to avoid overwriting (in case of partial re-runs)
  if (file.exists(chunk_file)) {
    cat("Chunk", chunk_id, "already completed. Skipping...\n")
    return(NULL)
  }
  
  # We'll store the scenario results here
  chunk_results <- data.frame()
  
  # Loop through each scenario in chunk_df
  for (i in seq_len(nrow(chunk_df))) {
    N_i    <- chunk_df$N[i]
    p_i    <- chunk_df$p[i]
    s_i    <- chunk_df$s[i]
    bs_i   <- chunk_df$beta_signal[i]
    repl_i <- chunk_df$repl[i]
    
    cat(sprintf("[Chunk %d] Scenario %d/%d: N=%d, p=%d, s=%d, beta=%.1f, repl=%d\n",
                chunk_id, i, nrow(chunk_df), N_i, p_i, s_i, bs_i, repl_i))
    
    # 1) Simulate data
    sim <- simulate_data(N = N_i, p = p_i, s = s_i,
                         beta_signal = bs_i, sigma_true = 1)
    
    # 2) Fit Cradle
    fit_cradle <- fit_model(cradle_mod, sim$X, sim$y)
    cradle_sum <- summary(fit_cradle, pars="beta")$summary
    beta_cradle <- cradle_sum[, "mean"]
    rm(fit_cradle, cradle_sum)
    gc()
    
    # 3) Fit Horseshoe
    fit_horseshoe <- fit_model(horseshoe_mod, sim$X, sim$y)
    horseshoe_sum <- summary(fit_horseshoe, pars="beta")$summary
    beta_horseshoe <- horseshoe_sum[, "mean"]
    rm(fit_horseshoe, horseshoe_sum)
    gc()
    
    # 4) Fit Lasso
    fit_lasso <- fit_model(lasso_mod, sim$X, sim$y)
    lasso_sum <- summary(fit_lasso, pars="beta")$summary
    beta_lasso <- lasso_sum[, "mean"]
    rm(fit_lasso, lasso_sum)
    gc()
    
    # 5) Evaluate
    cradle_res    <- evaluate_performance(beta_cradle, sim$beta_true)    %>% mutate(prior = "Cradle")
    horseshoe_res <- evaluate_performance(beta_horseshoe, sim$beta_true) %>% mutate(prior = "Horseshoe")
    lasso_res     <- evaluate_performance(beta_lasso, sim$beta_true)     %>% mutate(prior = "Lasso")
    
    # 6) Combine scenario results
    tmp <- rbind(cradle_res, horseshoe_res, lasso_res)
    tmp$N <- N_i
    tmp$p <- p_i
    tmp$s <- s_i
    tmp$beta_signal <- bs_i
    tmp$repl <- repl_i
    
    chunk_results <- rbind(chunk_results, tmp)
  }
  
  # Save the chunk's results
  saveRDS(chunk_results, chunk_file)
  cat("Chunk", chunk_id, "finished and saved to", chunk_file, "\n")
  invisible(chunk_results)
}

# ------------------------------
# 5) Run in Batches
# ------------------------------
chunk_size <- 50  # number of scenarios per chunk
n_rows <- nrow(big_grid)
n_chunks <- ceiling(n_rows / chunk_size)
cat("Total rows:", n_rows, ", chunk_size:", chunk_size,
    "=> # of chunks:", n_chunks, "\n")

# Loop over chunks
for (ch in seq_len(n_chunks)) {
  start_i <- (ch - 1) * chunk_size + 1
  end_i   <- min(ch * chunk_size, n_rows)
  chunk_df <- big_grid[start_i:end_i, ]
  
  cat("\n========== RUNNING CHUNK", ch, "of", n_chunks,
      "with rows", start_i, "to", end_i, "==========\n")
  
  run_chunk(chunk_df, chunk_id = ch)
}

cat("\nAll chunks completed!\n")

# ------------------------------
# 6) Combine Results and Summarize
# ------------------------------
# If you prefer to do this step in a separate script or after verifying all chunks, you can move it.
all_files <- list.files("batch_results", pattern = "^sim_results_chunk_.*\\.rds$", full.names = TRUE)
all_results <- do.call(rbind, lapply(all_files, readRDS))
cat("Combined results have", nrow(all_results), "rows in total.\n")

summary_df <- all_results %>%
  group_by(prior, N, p, s, beta_signal) %>%
  summarise(
    mse_mean = mean(mse),
    tpr_mean = mean(tpr),
    fdr_mean = mean(fdr),
    .groups = "drop"
  )

cat("\n--- Summary of Combined Results ---\n")
print(summary_df)

# Optional: Plot
ggplot(all_results, aes(x = prior, y = mse, fill = prior)) +
  geom_boxplot() +
  facet_grid(paste0("s=", s, ", beta=", beta_signal) ~ paste0("N=", N, ", p=", p)) +
  theme_minimal() +
  labs(title = "Extensive Comparison: Cradle, Horseshoe, Lasso",
       y = "Mean Squared Error", x = "Prior") +
  theme(legend.position = "none")

# Save final combined results
saveRDS(all_results, "all_results_combined.rds")
write.csv(all_results, "all_results_combined.csv", row.names = FALSE)
cat("Saved all_results_combined.rds and all_results_combined.csv\n")
