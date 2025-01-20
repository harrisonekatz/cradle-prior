data {
  int<lower=1> N;          // number of observations
  int<lower=1> p;          // number of predictors
  matrix[N, p] X;          // design matrix
  vector[N] y;             // response vector
}

parameters {
  real<lower=0> sigma;     // noise scale
  real<lower=0> lambda;    // global shrinkage parameter for Laplace
  vector<lower=0>[p] tau;  // per-coefficient scale
  vector[p] beta;          // regression coefficients
}

model {
  // 1) Likelihood
  y ~ normal(X * beta, sigma);

  // 2) Priors
  // Noise scale ~ half-Cauchy(0,2), for example
  sigma ~ cauchy(0, 2);

  // Global parameter lambda ~ Gamma-like or half-Cauchy, your choice
  // Often the standard Bayesian Lasso fixes lambda, or you can put a prior:
  lambda ~ cauchy(0, 1);   // or gamma, etc.

  // Local scales: tau[i] ~ Exponential(lambda^2/2) => Beta has Laplace(lambda)
  for(i in 1:p) {
    tau[i] ~ exponential(lambda^2 / 2);
    beta[i] ~ normal(0, sqrt(tau[i]));
  }
}
