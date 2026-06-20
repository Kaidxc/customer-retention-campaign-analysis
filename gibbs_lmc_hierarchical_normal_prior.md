# Gibbs + LMC for Hierarchical Normal Priors

This note describes two hierarchical Bayesian prior settings that extend the
current matched-normal prior used in the experiment code.

Current code setting:

```text
theta = (alpha, beta)
theta | mu, sigma^2 ~ Normal(mu, sigma^2 I)
mu = ground-truth parameters
sigma = 0.5
sigma^2 = 0.25
```

Therefore, the current implementation is:

```text
known prior mean + known/fixed prior variance
```

To make the prior genuinely hierarchical, either the prior mean or the prior
variance must be treated as an unknown quantity and updated during learning.

Throughout this note:

```text
theta = (alpha_1, ..., alpha_I, beta_1, ..., beta_I)
d = dimension(theta) = 2I
H_t = observed history up to time t
ell(theta; H_t) = log likelihood from the observed history
```

For a target posterior density `pi(z)`, the unadjusted Langevin update is written
as:

```text
z^{k+1} = z^k + h * grad_z log pi(z^k) + sqrt(2h) * xi^k
xi^k ~ Normal(0, I)
```

Some implementations use the equivalent convention:

```text
z^{k+1} = z^k + (epsilon / 2) * grad_z log pi(z^k) + sqrt(epsilon) * xi^k
```

The two forms differ only by the step-size convention.

If the algorithm includes an additional penalty term, its gradient should be
added to the likelihood/prior gradient in the same way as in the current
`TSpLangevin` implementation.

---

## Case 1: Unknown Mean + Known Variance

### Model

Here the prior variance is fixed, but the prior mean is unknown:

```text
theta | mu, sigma^2 ~ Normal(mu, sigma^2 I)
mu ~ Normal(m0, s_mu^2 I)
sigma^2 is fixed
```

In the current matched-normal scale, if we keep the same fixed prior standard
deviation:

```text
sigma = 0.5
sigma^2 = 0.25
```

The posterior is:

```text
p(theta, mu | H_t)
  proportional to
  p(H_t | theta)
  * Normal(theta | mu, sigma^2 I)
  * Normal(mu | m0, s_mu^2 I)
```

The log posterior, up to constants, is:

```text
log p(theta, mu | H_t)
= ell(theta; H_t)
  - (1 / (2 sigma^2)) * ||theta - mu||^2
  - (1 / (2 s_mu^2)) * ||mu - m0||^2
```

### LMC Update for theta Given mu

Given the current prior mean `mu`, update `theta` by LMC targeting:

```text
p(theta | mu, H_t)
  proportional to
  p(H_t | theta) * Normal(theta | mu, sigma^2 I)
```

The log target is:

```text
log p(theta | mu, H_t)
= ell(theta; H_t)
  - (1 / (2 sigma^2)) * ||theta - mu||^2
```

The gradient is:

```text
grad_theta log p(theta | mu, H_t)
= grad_theta ell(theta; H_t)
  - (theta - mu) / sigma^2
```

The LMC step is:

```text
theta^{k+1}
= theta^k
  + h * [grad_theta ell(theta^k; H_t)
         - (theta^k - mu) / sigma^2]
  + sqrt(2h) * xi_theta^k

xi_theta^k ~ Normal(0, I_d)
```

If using penalty regularisation, replace the gradient term by:

```text
grad_theta ell(theta^k; H_t)
- (theta^k - mu) / sigma^2
+ grad_theta penalty(theta^k)
```

If the code enforces bounds on `alpha` and `beta`, apply the same projection or
clipping step after the LMC update.

### Gibbs Update for mu Given theta

Because the prior is Normal-Normal, the conditional posterior of `mu` is Normal.

For the component-wise version:

```text
theta_j | mu_j, sigma^2 ~ Normal(mu_j, sigma^2)
mu_j ~ Normal(m0_j, s_mu^2)
```

Then:

```text
mu_j | theta_j ~ Normal(m_post_j, v_post)
```

where:

```text
v_post = 1 / (1 / s_mu^2 + 1 / sigma^2)

m_post_j
= v_post * (m0_j / s_mu^2 + theta_j / sigma^2)
```

So the Gibbs step is:

```text
mu_j^{new} ~ Normal(m_post_j, v_post)
```

for each component `j = 1, ..., d`.

### Shared Mean Version

If all alpha parameters share one unknown mean `mu_alpha`, and all beta
parameters share one unknown mean `mu_beta`, then:

```text
alpha_i | mu_alpha, sigma_alpha^2 ~ Normal(mu_alpha, sigma_alpha^2)
mu_alpha ~ Normal(m0_alpha, s_alpha_mu^2)
```

The conditional posterior is:

```text
mu_alpha | alpha
~ Normal(m_alpha_post, v_alpha_post)
```

where:

```text
v_alpha_post
= 1 / (1 / s_alpha_mu^2 + I / sigma_alpha^2)

m_alpha_post
= v_alpha_post
   * (m0_alpha / s_alpha_mu^2
      + sum_i alpha_i / sigma_alpha^2)
```

Similarly:

```text
v_beta_post
= 1 / (1 / s_beta_mu^2 + I / sigma_beta^2)

m_beta_post
= v_beta_post
   * (m0_beta / s_beta_mu^2
      + sum_i beta_i / sigma_beta^2)
```

### Gibbs + LMC Algorithm

For each decision time `t`:

```text
Input:
  observed history H_t
  current theta
  current mu
  fixed sigma^2
  hyper-prior parameters m0, s_mu^2

Repeat for k = 1, ..., N_t:

  1. LMC update theta given current mu:

     theta <- theta
              + h * [grad_theta ell(theta; H_t)
                     - (theta - mu) / sigma^2]
              + sqrt(2h) * Normal(0, I_d)

  2. Gibbs update mu given current theta:

     For each component j:

       v_post = 1 / (1 / s_mu^2 + 1 / sigma^2)

       m_post_j
       = v_post * (m0_j / s_mu^2 + theta_j / sigma^2)

       mu_j <- sample Normal(m_post_j, v_post)

Output:
  updated theta
  updated mu
```

---

## Case 2: Known Mean + Unknown Variance

### Model

Here the prior mean is fixed, but the prior variance is unknown:

```text
theta | mu, sigma^2 ~ Normal(mu, sigma^2 I)
sigma^2 ~ Inverse-Gamma(a, b)
mu is fixed
```

The posterior is:

```text
p(theta, sigma^2 | H_t)
  proportional to
  p(H_t | theta)
  * Normal(theta | mu, sigma^2 I)
  * Inverse-Gamma(sigma^2 | a, b)
```

Here `Inverse-Gamma(a, b)` is written with density proportional to:

```text
(sigma^2)^(-(a + 1)) * exp(-b / sigma^2)
```

### LMC Update for theta Given sigma^2

Given the current prior variance `sigma^2`, update `theta` by LMC targeting:

```text
p(theta | sigma^2, H_t)
  proportional to
  p(H_t | theta) * Normal(theta | mu, sigma^2 I)
```

The log target is:

```text
log p(theta | sigma^2, H_t)
= ell(theta; H_t)
  - (1 / (2 sigma^2)) * ||theta - mu||^2
```

The gradient is:

```text
grad_theta log p(theta | sigma^2, H_t)
= grad_theta ell(theta; H_t)
  - (theta - mu) / sigma^2
```

The LMC step is:

```text
theta^{k+1}
= theta^k
  + h * [grad_theta ell(theta^k; H_t)
         - (theta^k - mu) / sigma^2]
  + sqrt(2h) * xi_theta^k

xi_theta^k ~ Normal(0, I_d)
```

If using penalty regularisation, replace the gradient term by:

```text
grad_theta ell(theta^k; H_t)
- (theta^k - mu) / sigma^2
+ grad_theta penalty(theta^k)
```

### Gibbs Update for sigma^2 Given theta

Because the prior is Normal-Inverse-Gamma in variance form, the conditional
posterior of `sigma^2` is Inverse-Gamma.

From:

```text
theta | mu, sigma^2 ~ Normal(mu, sigma^2 I)
sigma^2 ~ Inverse-Gamma(a, b)
```

we get:

```text
sigma^2 | theta
~ Inverse-Gamma(a_post, b_post)
```

where:

```text
a_post = a + d / 2

b_post = b + 0.5 * ||theta - mu||^2
```

So the Gibbs step is:

```text
sigma2_new ~ Inverse-Gamma(a + d / 2,
                           b + 0.5 * ||theta - mu||^2)

sigma_new = sqrt(sigma2_new)
```

### Separate Alpha and Beta Variances

In the MNL model, `alpha` and `beta` may naturally have different scales. A more
flexible version uses separate variance parameters:

```text
sigma_alpha^2 ~ Inverse-Gamma(a_alpha, b_alpha)
sigma_beta^2  ~ Inverse-Gamma(a_beta, b_beta)
```

The model is:

```text
alpha_i | mu_alpha_i, sigma_alpha^2
~ Normal(mu_alpha_i, sigma_alpha^2)

beta_i | mu_beta_i, sigma_beta^2
~ Normal(mu_beta_i, sigma_beta^2)
```

The Gibbs updates are:

```text
sigma_alpha^2 | alpha
~ Inverse-Gamma(a_alpha + I / 2,
                b_alpha + 0.5 * sum_i (alpha_i - mu_alpha_i)^2)
```

and:

```text
sigma_beta^2 | beta
~ Inverse-Gamma(a_beta + I / 2,
                b_beta + 0.5 * sum_i (beta_i - mu_beta_i)^2)
```

The LMC gradient for `alpha` becomes:

```text
grad_alpha log p(alpha | sigma_alpha^2, H_t)
= grad_alpha ell(theta; H_t)
  - (alpha - mu_alpha) / sigma_alpha^2
```

The LMC gradient for `beta` becomes:

```text
grad_beta log p(beta | sigma_beta^2, H_t)
= grad_beta ell(theta; H_t)
  - (beta - mu_beta) / sigma_beta^2
```

### Precision Notation as an Equivalent Form

Some derivations use the precision:

```text
tau = 1 / sigma^2
```

Then:

```text
theta | mu, tau ~ Normal(mu, tau^{-1} I)
tau ~ Gamma(a, b)
```

where `Gamma(a, b)` uses the rate parameterisation:

```text
p(tau) proportional to tau^(a-1) * exp(-b tau)
```

The Gibbs update is:

```text
tau | theta
~ Gamma(a + d / 2,
        b + 0.5 * ||theta - mu||^2)
```

This is mathematically equivalent to the Inverse-Gamma update for `sigma^2`.
In this note, `sigma^2` is the main notation because the modelling question is
about an unknown prior variance.

### Gibbs + LMC Algorithm

For each decision time `t`:

```text
Input:
  observed history H_t
  current theta
  fixed prior mean mu
  current sigma^2
  hyper-prior parameters a, b

Repeat for k = 1, ..., N_t:

  1. LMC update theta given current sigma^2:

     theta <- theta
              + h * [grad_theta ell(theta; H_t)
                     - (theta - mu) / sigma^2]
              + sqrt(2h) * Normal(0, I_d)

  2. Gibbs update sigma^2 given current theta:

     a_post = a + d / 2

     b_post = b + 0.5 * ||theta - mu||^2

     sigma^2 <- sample Inverse-Gamma(a_post, b_post)

  3. Optionally compute:

     sigma = sqrt(sigma^2)

Output:
  updated theta
  updated sigma^2
```

For separate alpha and beta variances, replace step 2 by:

```text
sigma_alpha^2 <- sample Inverse-Gamma(
                   a_alpha + I / 2,
                   b_alpha + 0.5 * ||alpha - mu_alpha||^2)

sigma_beta^2 <- sample Inverse-Gamma(
                  a_beta + I / 2,
                  b_beta + 0.5 * ||beta - mu_beta||^2)
```

---

## Comparison of the Two Gibbs + LMC Schemes

| Setting | Unknown quantity | LMC updates | Gibbs updates |
|---|---|---|---|
| unknown mean + known variance | `mu` | `theta` | `mu | theta` |
| known mean + unknown variance | `sigma^2` | `theta` | `sigma^2 | theta` |
| unknown mean + unknown variance | `mu`, `sigma^2` | `theta` | `mu | theta, sigma^2`; `sigma^2 | theta, mu` |

The important point is that LMC is still used for the main model parameters
`theta = (alpha, beta)`, because the likelihood term from the MNL model usually
does not give a simple conjugate posterior for `theta`.

The hierarchical parameters are then updated by Gibbs steps because their
conditional posteriors are available in closed form.

---

## Practical Notes for the Current Code

### Current matched-normal prior

The current code uses:

```text
mu = ground-truth parameters
sigma = 0.5
sigma^2 = 0.25
```

This is not an unknown-mean or unknown-variance model. It is a fixed informative
Gaussian prior.

### To implement unknown mean + known variance

Add state variables:

```text
mu_alpha
mu_beta
```

Keep:

```text
sigma_alpha = 0.5
sigma_beta = 0.5
```

At every LMC/Gibbs cycle:

```text
1. update alpha, beta by LMC given mu_alpha, mu_beta
2. update mu_alpha, mu_beta by Gibbs given alpha, beta
```

### To implement known mean + unknown variance

Keep:

```text
mu_alpha = fixed prior mean
mu_beta = fixed prior mean
```

Add state variables:

```text
sigma_alpha^2
sigma_beta^2
```

At every LMC/Gibbs cycle:

```text
1. update alpha, beta by LMC given sigma_alpha^2, sigma_beta^2
2. update sigma_alpha^2, sigma_beta^2 by Gibbs given alpha, beta
```

### What should be logged

For unknown mean:

```text
mu_alpha_i
mu_beta_i
```

For unknown variance:

```text
sigma_alpha^2
sigma_beta^2
sigma_alpha
sigma_beta
```

Optionally also log precision values:

```text
tau_alpha = 1 / sigma_alpha^2
tau_beta = 1 / sigma_beta^2
```

This makes it possible to check whether the hierarchical prior parameters are
actually being learned over time.
