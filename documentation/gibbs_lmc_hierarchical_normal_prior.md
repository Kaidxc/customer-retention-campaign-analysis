# Gibbs + LMC for Hierarchical Normal Priors

This note describes how to extend the current matched-normal prior into two
hierarchical Bayesian settings using a Gibbs + Langevin Monte Carlo (LMC)
scheme.

The formulas are written in GitHub-friendly Markdown math. To avoid GitHub
misinterpreting lines beginning with `+` or `-` as bullet lists, most display
equations are kept on a single line.

## Current Code Setting

Let the unknown MNL parameters be

$$\theta = (\alpha_1,\ldots,\alpha_I,\beta_1,\ldots,\beta_I), \qquad d=\dim(\theta)=2I.$$

The current matched-normal prior used in the code is

$$\theta \mid \mu,\sigma^2 \sim \mathcal{N}(\mu,\sigma^2 I_d).$$

In the current experiment,

$$\mu=\theta_{\mathrm{true}}, \qquad \sigma=0.5, \qquad \sigma^2=0.25.$$

Therefore, the current implementation is

$$\text{known prior mean} + \text{known/fixed prior variance}.$$

It is not yet an unknown-mean or unknown-variance hierarchical prior.

Let

$$H_t=\text{observed history up to time }t,$$

and define the log likelihood as

$$\ell(\theta;H_t)=\log p(H_t\mid \theta).$$

For a target posterior density $\pi(z)$, one common unadjusted Langevin update is

$$z^{(k+1)}=z^{(k)}+h\nabla_z\log\pi(z^{(k)})+\sqrt{2h}\xi^{(k)}, \qquad \xi^{(k)}\sim\mathcal{N}(0,I).$$

Some implementations use the equivalent convention

$$z^{(k+1)}=z^{(k)}+\frac{\varepsilon}{2}\nabla_z\log\pi(z^{(k)})+\sqrt{\varepsilon}\xi^{(k)}.$$

The two forms differ only by the step-size convention. If the algorithm also
uses a penalty term, its gradient should be added to the likelihood/prior
gradient in the same way as in the current `TSpLangevin` implementation.

---

## Case 1: Unknown Mean + Known Variance

### Model

Here the prior variance is fixed, but the prior mean is unknown:

$$\theta\mid\mu,\sigma^2\sim\mathcal{N}(\mu,\sigma^2 I_d), \qquad \mu\sim\mathcal{N}(m_0,s_\mu^2 I_d).$$

The prior variance $\sigma^2$ is fixed. If we keep the same scale as the current
matched-normal experiment, then

$$\sigma=0.5, \qquad \sigma^2=0.25.$$

The posterior is

$$p(\theta,\mu\mid H_t)\propto p(H_t\mid\theta)\mathcal{N}(\theta\mid\mu,\sigma^2 I_d)\mathcal{N}(\mu\mid m_0,s_\mu^2 I_d).$$

The log posterior, up to constants, is

$$\log p(\theta,\mu\mid H_t)=\ell(\theta;H_t)-\frac{1}{2\sigma^2}\|\theta-\mu\|^2-\frac{1}{2s_\mu^2}\|\mu-m_0\|^2.$$

### LMC Update for $\theta$ Given $\mu$

Given the current prior mean $\mu$, update $\theta$ by LMC targeting

$$p(\theta\mid\mu,H_t)\propto p(H_t\mid\theta)\mathcal{N}(\theta\mid\mu,\sigma^2 I_d).$$

The log target is

$$\log p(\theta\mid\mu,H_t)=\ell(\theta;H_t)-\frac{1}{2\sigma^2}\|\theta-\mu\|^2.$$

The gradient is

$$\nabla_\theta\log p(\theta\mid\mu,H_t)=\nabla_\theta\ell(\theta;H_t)-\frac{\theta-\mu}{\sigma^2}.$$

Therefore, the LMC update is

$$\theta^{(k+1)}=\theta^{(k)}+h\left[\nabla_\theta\ell(\theta^{(k)};H_t)-\frac{\theta^{(k)}-\mu}{\sigma^2}\right]+\sqrt{2h}\xi_\theta^{(k)}, \qquad \xi_\theta^{(k)}\sim\mathcal{N}(0,I_d).$$

If using penalty regularisation, replace the gradient term by

$$\nabla_\theta\ell(\theta^{(k)};H_t)-\frac{\theta^{(k)}-\mu}{\sigma^2}+\nabla_\theta\operatorname{Penalty}(\theta^{(k)}).$$

If the code enforces bounds on $\alpha$ and $\beta$, apply the same projection or
clipping step after the LMC update.

### Gibbs Update for $\mu$ Given $\theta$

Because this is a Normal-Normal conditional structure, $\mu\mid\theta$ is Normal.
For each component $j=1,\ldots,d$,

$$\theta_j\mid\mu_j,\sigma^2\sim\mathcal{N}(\mu_j,\sigma^2), \qquad \mu_j\sim\mathcal{N}(m_{0,j},s_\mu^2).$$

Then

$$\mu_j\mid\theta_j\sim\mathcal{N}(m_{\mathrm{post},j},v_{\mathrm{post}}),$$

where

$$v_{\mathrm{post}}=\left(\frac{1}{s_\mu^2}+\frac{1}{\sigma^2}\right)^{-1},$$

and

$$m_{\mathrm{post},j}=v_{\mathrm{post}}\left(\frac{m_{0,j}}{s_\mu^2}+\frac{\theta_j}{\sigma^2}\right).$$

So the Gibbs step is

$$\mu_j^{\mathrm{new}}\sim\mathcal{N}(m_{\mathrm{post},j},v_{\mathrm{post}}), \qquad j=1,\ldots,d.$$

### Shared Mean Version

If all $\alpha_i$ share one unknown mean $\mu_\alpha$, and all $\beta_i$ share one
unknown mean $\mu_\beta$, then for $\alpha$:

$$\alpha_i\mid\mu_\alpha,\sigma_\alpha^2\sim\mathcal{N}(\mu_\alpha,\sigma_\alpha^2), \qquad \mu_\alpha\sim\mathcal{N}(m_{0,\alpha},s_{\alpha,\mu}^2).$$

The conditional posterior is

$$\mu_\alpha\mid\alpha\sim\mathcal{N}(m_{\alpha,\mathrm{post}},v_{\alpha,\mathrm{post}}),$$

where

$$v_{\alpha,\mathrm{post}}=\left(\frac{1}{s_{\alpha,\mu}^2}+\frac{I}{\sigma_\alpha^2}\right)^{-1},$$

and

$$m_{\alpha,\mathrm{post}}=v_{\alpha,\mathrm{post}}\left(\frac{m_{0,\alpha}}{s_{\alpha,\mu}^2}+\frac{\sum_{i=1}^I\alpha_i}{\sigma_\alpha^2}\right).$$

Similarly,

$$v_{\beta,\mathrm{post}}=\left(\frac{1}{s_{\beta,\mu}^2}+\frac{I}{\sigma_\beta^2}\right)^{-1},$$

and

$$m_{\beta,\mathrm{post}}=v_{\beta,\mathrm{post}}\left(\frac{m_{0,\beta}}{s_{\beta,\mu}^2}+\frac{\sum_{i=1}^I\beta_i}{\sigma_\beta^2}\right).$$

### Gibbs + LMC Algorithm

For each decision time $t$:

1. Start with observed history $H_t$, current $\theta$, current $\mu$, fixed
   $\sigma^2$, and hyper-prior parameters $m_0,s_\mu^2$.

2. LMC update $\theta$ given current $\mu$:

$$\theta\leftarrow\theta+h\left[\nabla_\theta\ell(\theta;H_t)-\frac{\theta-\mu}{\sigma^2}\right]+\sqrt{2h}\xi_\theta, \qquad \xi_\theta\sim\mathcal{N}(0,I_d).$$

3. Gibbs update $\mu$ given current $\theta$:

$$v_{\mathrm{post}}=\left(\frac{1}{s_\mu^2}+\frac{1}{\sigma^2}\right)^{-1}, \qquad m_{\mathrm{post},j}=v_{\mathrm{post}}\left(\frac{m_{0,j}}{s_\mu^2}+\frac{\theta_j}{\sigma^2}\right).$$

4. Sample

$$\mu_j\leftarrow\mathcal{N}(m_{\mathrm{post},j},v_{\mathrm{post}}).$$

5. Repeat the LMC/Gibbs cycle for $k=1,\ldots,N_t$.

---

## Case 2: Known Mean + Unknown Variance

### Model

Here the prior mean is fixed, but the prior variance is unknown:

$$\theta\mid\mu,\sigma^2\sim\mathcal{N}(\mu,\sigma^2 I_d), \qquad \sigma^2\sim\operatorname{Inverse\text{-}Gamma}(a,b).$$

The prior mean $\mu$ is fixed.

The posterior is

$$p(\theta,\sigma^2\mid H_t)\propto p(H_t\mid\theta)\mathcal{N}(\theta\mid\mu,\sigma^2 I_d)\operatorname{Inverse\text{-}Gamma}(\sigma^2\mid a,b).$$

Here $\operatorname{Inverse\text{-}Gamma}(a,b)$ is written with density
proportional to

$$p(\sigma^2)\propto(\sigma^2)^{-(a+1)}\exp\left(-\frac{b}{\sigma^2}\right).$$

### LMC Update for $\theta$ Given $\sigma^2$

Given the current prior variance $\sigma^2$, update $\theta$ by LMC targeting

$$p(\theta\mid\sigma^2,H_t)\propto p(H_t\mid\theta)\mathcal{N}(\theta\mid\mu,\sigma^2 I_d).$$

The log target is

$$\log p(\theta\mid\sigma^2,H_t)=\ell(\theta;H_t)-\frac{1}{2\sigma^2}\|\theta-\mu\|^2.$$

The gradient is

$$\nabla_\theta\log p(\theta\mid\sigma^2,H_t)=\nabla_\theta\ell(\theta;H_t)-\frac{\theta-\mu}{\sigma^2}.$$

Therefore, the LMC update is

$$\theta^{(k+1)}=\theta^{(k)}+h\left[\nabla_\theta\ell(\theta^{(k)};H_t)-\frac{\theta^{(k)}-\mu}{\sigma^2}\right]+\sqrt{2h}\xi_\theta^{(k)}, \qquad \xi_\theta^{(k)}\sim\mathcal{N}(0,I_d).$$

If using penalty regularisation, replace the gradient term by

$$\nabla_\theta\ell(\theta^{(k)};H_t)-\frac{\theta^{(k)}-\mu}{\sigma^2}+\nabla_\theta\operatorname{Penalty}(\theta^{(k)}).$$

### Gibbs Update for $\sigma^2$ Given $\theta$

Because this is a Normal-Inverse-Gamma conditional structure, the conditional
posterior of $\sigma^2$ is Inverse-Gamma:

$$\sigma^2\mid\theta\sim\operatorname{Inverse\text{-}Gamma}(a_{\mathrm{post}},b_{\mathrm{post}}).$$

The posterior parameters are

$$a_{\mathrm{post}}=a+\frac{d}{2}, \qquad b_{\mathrm{post}}=b+\frac{1}{2}\|\theta-\mu\|^2.$$

So the Gibbs step is

$$\sigma_{\mathrm{new}}^2\sim\operatorname{Inverse\text{-}Gamma}\left(a+\frac{d}{2}, b+\frac{1}{2}\|\theta-\mu\|^2\right).$$

Then

$$\sigma_{\mathrm{new}}=\sqrt{\sigma_{\mathrm{new}}^2}.$$

### Separate Alpha and Beta Variances

In the MNL model, $\alpha$ and $\beta$ may naturally have different scales. A
more flexible version uses separate variance parameters:

$$\sigma_\alpha^2\sim\operatorname{Inverse\text{-}Gamma}(a_\alpha,b_\alpha), \qquad \sigma_\beta^2\sim\operatorname{Inverse\text{-}Gamma}(a_\beta,b_\beta).$$

The model is

$$\alpha_i\mid\mu_{\alpha,i},\sigma_\alpha^2\sim\mathcal{N}(\mu_{\alpha,i},\sigma_\alpha^2), \qquad \beta_i\mid\mu_{\beta,i},\sigma_\beta^2\sim\mathcal{N}(\mu_{\beta,i},\sigma_\beta^2).$$

The Gibbs updates are

$$\sigma_\alpha^2\mid\alpha\sim\operatorname{Inverse\text{-}Gamma}\left(a_\alpha+\frac{I}{2}, b_\alpha+\frac{1}{2}\sum_{i=1}^I(\alpha_i-\mu_{\alpha,i})^2\right),$$

and

$$\sigma_\beta^2\mid\beta\sim\operatorname{Inverse\text{-}Gamma}\left(a_\beta+\frac{I}{2}, b_\beta+\frac{1}{2}\sum_{i=1}^I(\beta_i-\mu_{\beta,i})^2\right).$$

The LMC gradients become

$$\nabla_\alpha\log p(\alpha\mid\sigma_\alpha^2,H_t)=\nabla_\alpha\ell(\theta;H_t)-\frac{\alpha-\mu_\alpha}{\sigma_\alpha^2},$$

and

$$\nabla_\beta\log p(\beta\mid\sigma_\beta^2,H_t)=\nabla_\beta\ell(\theta;H_t)-\frac{\beta-\mu_\beta}{\sigma_\beta^2}.$$

### Precision Notation as an Equivalent Form

Some derivations use the precision

$$\tau=\frac{1}{\sigma^2}.$$

Then

$$\theta\mid\mu,\tau\sim\mathcal{N}(\mu,\tau^{-1}I_d), \qquad \tau\sim\operatorname{Gamma}(a,b).$$

Here $\operatorname{Gamma}(a,b)$ uses the rate parameterisation

$$p(\tau)\propto\tau^{a-1}\exp(-b\tau).$$

The Gibbs update is

$$\tau\mid\theta\sim\operatorname{Gamma}\left(a+\frac{d}{2}, b+\frac{1}{2}\|\theta-\mu\|^2\right).$$

This is mathematically equivalent to the Inverse-Gamma update for $\sigma^2$.
In this note, $\sigma^2$ is the main notation because the modelling question is
about an unknown prior variance.

### Gibbs + LMC Algorithm

For each decision time $t$:

1. Start with observed history $H_t$, current $\theta$, fixed $\mu$, current
   $\sigma^2$, and hyper-prior parameters $a,b$.

2. LMC update $\theta$ given current $\sigma^2$:

$$\theta\leftarrow\theta+h\left[\nabla_\theta\ell(\theta;H_t)-\frac{\theta-\mu}{\sigma^2}\right]+\sqrt{2h}\xi_\theta, \qquad \xi_\theta\sim\mathcal{N}(0,I_d).$$

3. Gibbs update $\sigma^2$ given current $\theta$:

$$a_{\mathrm{post}}=a+\frac{d}{2}, \qquad b_{\mathrm{post}}=b+\frac{1}{2}\|\theta-\mu\|^2.$$

4. Sample

$$\sigma^2\leftarrow\operatorname{Inverse\text{-}Gamma}(a_{\mathrm{post}},b_{\mathrm{post}}).$$

5. Optionally compute

$$\sigma=\sqrt{\sigma^2}.$$

6. Repeat the LMC/Gibbs cycle for $k=1,\ldots,N_t$.

For separate alpha and beta variances, replace step 3 and step 4 by

$$\sigma_\alpha^2\leftarrow\operatorname{Inverse\text{-}Gamma}\left(a_\alpha+\frac{I}{2}, b_\alpha+\frac{1}{2}\|\alpha-\mu_\alpha\|^2\right),$$

and

$$\sigma_\beta^2\leftarrow\operatorname{Inverse\text{-}Gamma}\left(a_\beta+\frac{I}{2}, b_\beta+\frac{1}{2}\|\beta-\mu_\beta\|^2\right).$$

---

## Comparison of the Two Gibbs + LMC Schemes

| Setting | Unknown quantity | LMC updates | Gibbs updates |
|---|---|---|---|
| unknown mean + known variance | $\mu$ | $\theta$ | $\mu\mid\theta$ |
| known mean + unknown variance | $\sigma^2$ | $\theta$ | $\sigma^2\mid\theta$ |
| unknown mean + unknown variance | $\mu$, $\sigma^2$ | $\theta$ | $\mu\mid\theta,\sigma^2$ and $\sigma^2\mid\theta,\mu$ |

The important point is that LMC is still used for the main model parameters
$\theta=(\alpha,\beta)$, because the likelihood term from the MNL model usually
does not give a simple conjugate posterior for $\theta$.

The hierarchical parameters are then updated by Gibbs steps because their
conditional posteriors are available in closed form.

---

## Practical Notes for the Current Code

### Current matched-normal prior

The current code uses

$$\mu=\theta_{\mathrm{true}}, \qquad \sigma=0.5, \qquad \sigma^2=0.25.$$

This is not an unknown-mean or unknown-variance model. It is a fixed informative
Gaussian prior.

### To Implement Unknown Mean + Known Variance

Add state variables:

$$\mu_\alpha, \qquad \mu_\beta.$$

Keep:

$$\sigma_\alpha=0.5, \qquad \sigma_\beta=0.5.$$

At every LMC/Gibbs cycle:

1. Update $\alpha,\beta$ by LMC given $\mu_\alpha,\mu_\beta$.
2. Update $\mu_\alpha,\mu_\beta$ by Gibbs given $\alpha,\beta$.

### To Implement Known Mean + Unknown Variance

Keep:

$$\mu_\alpha=\text{fixed prior mean}, \qquad \mu_\beta=\text{fixed prior mean}.$$

Add state variables:

$$\sigma_\alpha^2, \qquad \sigma_\beta^2.$$

At every LMC/Gibbs cycle:

1. Update $\alpha,\beta$ by LMC given $\sigma_\alpha^2,\sigma_\beta^2$.
2. Update $\sigma_\alpha^2,\sigma_\beta^2$ by Gibbs given $\alpha,\beta$.

### What Should Be Logged

For unknown mean, log:

$$\mu_{\alpha,i}, \qquad \mu_{\beta,i}.$$

For unknown variance, log:

$$\sigma_\alpha^2, \qquad \sigma_\beta^2, \qquad \sigma_\alpha, \qquad \sigma_\beta.$$

Optionally also log precision values:

$$\tau_\alpha=\frac{1}{\sigma_\alpha^2}, \qquad \tau_\beta=\frac{1}{\sigma_\beta^2}.$$

This makes it possible to check whether the hierarchical prior parameters are
actually being learned over time.

