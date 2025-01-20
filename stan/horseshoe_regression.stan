data {
  int<lower=1> N;            // number of observations
  int<lower=1> p;            // number of predictors
  matrix[N, p] X;            // design matrix
  vector[N] y;               // response vector
}

parameters {
  real<lower=0> sigma;       // noise scale
  real<lower=0> tau;         // global scale (half-Cauchy)
  vector<lower=0>[p] lambda; // local scales (each half-Cauchy)
  vector[p] beta;            // regression coefficients
}

model {
  // 1) Likelihood
  y ~ normal(X * beta, sigma);

  // 2) Priors
  // Noise scale ~ half-Cauchy
  sigma ~ cauchy(0, 2);  // or pick scale=2 etc.

  // Global scale ~ half-Cauchy(0,1)
  tau ~ cauchy(0, 1);

  // Local scales ~ half-Cauchy(0,1)
  lambda ~ cauchy(0, 1);

  // Coefficients: beta_i ~ Normal(0, (tau*lambda_i)^2)
  for(i in 1:p) {
    beta[i] ~ normal(0, tau * lambda[i]);
  }
}
