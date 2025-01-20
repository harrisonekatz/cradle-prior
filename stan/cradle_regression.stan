data {
  int<lower=1> N;              // number of observations
  int<lower=1> p;              // number of predictors
  matrix[N, p] X;              // design matrix
  vector[N] y;                 // response vector
}

parameters {
  real<lower=0> sigma;         // noise scale
  real<lower=0> tau;           // global shrinkage scale
  real<lower=0> alpha;         // rate for Half-Laplace (Exponential) on lambda_i
  vector<lower=0>[p] lambda;   // local shrinkage scales
  vector[p] beta;              // regression coefficients
}

model {
  // 1) Likelihood
  y ~ normal(X * beta, sigma);

  // 2) Priors
  // Noise scale ~ half-Cauchy(0,2)
  sigma ~ cauchy(0, 2);

  // Global scale ~ half-Cauchy(0,1)
  tau ~ cauchy(0, 1);

  // Local scale ~ half-Laplace(alpha) => Exponential(alpha)
  lambda ~ exponential(alpha);

  // Hyperprior on alpha (e.g. Gamma(2,2))
  alpha ~ gamma(2, 2);

  // Coefficients ~ Normal(0, (tau*lambda)^2)
  for (i in 1:p) {
    beta[i] ~ normal(0, tau * lambda[i]);
  }
}
