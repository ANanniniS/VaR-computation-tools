# Fundamental definitions
We're analyzing how the prices of different assets evolve. For each firm $i\in \{1,...,I\}$ we have a **price** at each time $t$, which we denote
$$
P_i(t)
$$
From that we define, its **return**,
$$
r_i(t) = \frac{P_i(t)}{P_i(t-1)}.
$$
and its **log-return** is
$$
	l_i(t) = \ln r_i(t).
$$
If we buy $x_i$ shares of each firm $i$, then the value of the portfolio as a function of time will be given by
$$
	V(t) = \sum_i P_i(t)x_i = \sum_i r_i(t)r_i(t-1)...r_i(1)w_i
$$
Where $w_i = x_i P_i(0)$ is the **weight of firm $i$**. As we are looking for percentage increments we always use the weights normalized to one. Then,
$$
	V(0) = 1
$$
Thus the profit or loss of our portfolio can be computed as
$$
	PnL(t,\mathbf{w}) = V(t)-1.
$$

The idea behind the Value at Risk (VaR) and Markowitz portfolio computations is to model the future quantities (as $P_i(t)$) as Random Variables (RV).

Saying that the VaR of a certain portfolio is $3\%$ at the $5\%$ significance level with a **horizon** of $3$ days means that we have $95\%$ chances of losing no more than $3\%$ of its value over those $3$ days.

## Historical VaR

The historical method estimates the VaR of a portfolio directly from the empirical distribution of past $PnL(t,\mathbf{w})$ observations, without assuming any particular distribution for the underlying returns.

Given a historical sample of $T$ observations of $PnL(t,\mathbf{w})$, we sort them from worst to best and estimate the $\alpha$-quantile of the empirical distribution:

$$q_\alpha = \inf\{x : \hat{F}(x) \geq \alpha\} \tag{H.1}$$

where $\hat{F}$ is the empirical cumulative distribution function of $PnL(t,\mathbf{w})$ built from the historical sample. In practice, this is computed directly as the $\alpha$-percentile of the sample.

The historical VaR at confidence level $1-\alpha$ is then defined as:

$$VaR_\alpha^{hist} = -q_\alpha \tag{H.2}$$

The minus sign follows the convention adopted throughout this document: VaR is reported as a **positive number representing a loss**, even though $q_\alpha$ itself is typically negative.

**Key property**: It is often claimed that the historical VaR makes no assumption because it doesn't assume any particular distribution. The truth is that it's actually making the assumption that the past measurements are a representative output of the future results.
## Parametric and Monte Carlo VaR

The following methods model the market dynamics by assuming that each
$$
	\mathbf{l}(t) = 
	\left[
		\begin{array}{c}
		l_1(t)\\
		l_2(t)\\
		\vdots\\
		l_I(t)
		\end{array}
	\right]
$$
is an independent RV drawn from a multivariate normal distribution,
$$
	\mathbf{l}(t)\sim \mathcal{N}(\mu,\Sigma)
$$
where $\mu$ and $\Sigma$ are the mean vector and covariance matrix.
If we use the independence of the $\mathbf{l}$s (as the model says), we can compute,
$$
	\mathbf{L} = 
		\mathbf{l}(1)+...+\mathbf{l}(t) \sim
		\mathcal{N}(t\mu,t\Sigma).
$$

Thus, we can compute the probability distribution of the portfolio value,
$$
	V(t) = \sum \exp(L_i) w_i = \mathbf{w}^T\exp(\mathbf{L})
$$
where $\exp(\mathbf{L})$ means the pointwise exponential. The RV $\exp(\mathbf{L})$ follows a *log-normal distribution* and can be handled analytically. Unfortunately, linear combinations of its components cannot. So, at this point there are two usual ways of approaching this problem.

The parameters $\mu$ and $\Sigma$ are obtained through point estimation using the data.

### Parametric approach

Even though $V(t)$ cannot be computed exactly as a linear combination of correlated log-normal variables, we can compute its exact mean and variance analytically, and then approximate its distribution as normal using only these two moments.

**Moments of the log-normal price factors.** For each asset $i$, since $L_i \sim \mathcal{N}(t\mu_i, t\Sigma_{ii})$, the log-normal identity gives:

$$E[\exp(L_i)] = \exp\left(t\mu_i + \frac{t\Sigma_{ii}}{2}\right) \tag{P.1}$$

$$E[\exp(L_i)\exp(L_j)] = \exp\left(t\mu_i + t\mu_j + \frac{t\Sigma_{ii}+t\Sigma_{jj}}{2} + t\Sigma_{ij}\right) \tag{P.2}$$

From these, the covariance between exponentiated components follows directly:

$$\text{Cov}(\exp(L_i), \exp(L_j)) = E[\exp(L_i)]E[\exp(L_j)]\left(\exp(t\Sigma_{ij}) - 1\right) \tag{P.3}$$

**Portfolio moments.** Since $V(t) = \mathbf{w}^T\exp(\mathbf{L})$ is a linear combination of these exponentiated components, its mean and variance follow directly from (P.1) and (P.3):

$$E[V(t)] = \mathbf{w}^T E[\exp(\mathbf{L})] \tag{P.4}$$

$$\text{Var}(V(t)) = \mathbf{w}^T \, \text{Cov}(\exp(\mathbf{L})) \, \mathbf{w} \tag{P.5}$$

**Normal approximation and VaR.** Approximating $PnL(t,\mathbf{w}) = V(t) - 1$ as normally distributed with mean $\mu_p = E[V(t)] - 1$ and standard deviation $\sigma_p = \sqrt{\text{Var}(V(t))}$, the parametric VaR at confidence level $1-\alpha$ is obtained analytically from the normal quantile:

$$VaR_\alpha^{param} = -\left(\mu_p + z_\alpha \, \sigma_p\right) \tag{P.6}$$

where $z_\alpha = \Phi^{-1}(\alpha)$ is the $\alpha$-quantile of the standard normal distribution.

### Monte Carlo approach

Instead of approximating the distribution of $V(t)$ analytically, the Monte Carlo approach draws directly from the known distribution of $\mathbf{L}$ and evaluates $V(t)$ exactly on each draw, letting the true (non-normal) shape of $V(t)$ emerge empirically.

**Simulation.** We draw $N$ independent samples from the exact distribution of the aggregated log-returns,

$$\mathbf{L}^{(n)} \sim \mathcal{N}(t\mu, t\Sigma), \qquad n = 1,\dots,N \tag{M.1}$$

and compute the corresponding simulated portfolio value for each draw,

$$V^{(n)}(t) = \mathbf{w}^T\exp\left(\mathbf{L}^{(n)}\right) \tag{M.2}$$

so that $PnL^{(n)}(t,\mathbf{w}) = V^{(n)}(t) - 1$ is one simulated outcome of the portfolio's profit or loss.

**Empirical VaR.** With $N$ simulated outcomes in hand, the Monte Carlo VaR is estimated exactly as in the historical method (Eq. H.1–H.2), but applied to the *simulated* sample instead of the *historical* one:

$$VaR_\alpha^{mc} = -q_\alpha\left(\{PnL^{(n)}(t,\mathbf{w})\}_{n=1}^N\right) \tag{M.3}$$

where $q_\alpha(\cdot)$ denotes the empirical $\alpha$-quantile of the simulated sample.

## Bayesian VaR

The previous methods do not depend only on the model, but also on point estimates of its parameters ($\mu$, $\Sigma$). This means these methods are unable to account for the uncertainty that comes from the amount of data available — they produce a single estimate, regardless of how much data supports it.

To address this, we use Bayesian inference tools to make the simulation aware of the amount of data available, by treating $\mu$ and $\Sigma$ themselves as random variables with their own distribution, rather than fixed point estimates.

**Model.** We place a Normal-Inverse-Wishart (NIW) prior over $(\mu,\Sigma)$, the standard conjugate prior for a multivariate normal with unknown mean and covariance:

$$\Sigma \sim \mathcal{IW}(\nu_0, \Psi_0), \qquad \mu \mid \Sigma \sim \mathcal{N}\left(\mu_0, \frac{\Sigma}{\kappa_0}\right) \tag{B.1}$$

Given a historical sample of $T$ observations $\mathbf{l}(1),\dots,\mathbf{l}(T)$, and starting from an uninformative prior ($\kappa_0 \to 0$), the NIW posterior parameters reduce to:

$$\kappa = T, \qquad \nu = T - 1, \qquad \Psi = \sum_{t=1}^{T}\left(\mathbf{l}(t)-\hat\mu\right)\left(\mathbf{l}(t)-\hat\mu\right)^T \tag{B.2}$$

where $\hat\mu = \frac{1}{T}\sum_t \mathbf{l}(t)$ is the sample mean. Note that $\kappa$ grows with the sample size $T$ — this is precisely the mechanism that lets the model "know" how much data supports the estimate: more data means a tighter posterior on $\mu$ (Eq. B.1), and therefore less parameter uncertainty in the final VaR.

**Posterior predictive simulation.** Rather than computing $VaR$ conditional on fixed $\hat\mu$ and $\hat\Sigma$, we simulate the _posterior predictive distribution_ of $\mathbf{L}$, which integrates over the uncertainty in $(\mu,\Sigma)$ itself. For each of $N$ simulations $n=1,\dots,N$:

$$\Sigma^{(n)} \sim \mathcal{IW}(\nu,\Psi) \tag{B.3}$$

$$\mu^{(n)} \mid \Sigma^{(n)} \sim \mathcal{N}\left(\hat\mu, \frac{\Sigma^{(n)}}{\kappa}\right) \tag{B.4}$$

$$\mathbf{L}^{(n)} \mid \mu^{(n)}, \Sigma^{(n)} \sim \mathcal{N}\left(t\mu^{(n)}, t\Sigma^{(n)}\right) \tag{B.5}$$

Each draw first samples a plausible covariance matrix (B.3), then a plausible mean conditional on that covariance (B.4) — capturing parameter uncertainty — and finally a plausible horizon-$t$ outcome conditional on that particular $(\mu^{(n)},\Sigma^{(n)})$ (B.5) — capturing the process' own randomness. As in the Monte Carlo approach, each simulated $\mathbf{L}^{(n)}$ is exponentiated and combined with the portfolio weights:

$$V^{(n)}(t) = \mathbf{w}^T\exp\left(\mathbf{L}^{(n)}\right) \tag{B.6}$$

and the Bayesian VaR is the empirical quantile of the resulting simulated $PnL$, exactly as in Eq. M.3:

$$VaR_\alpha^{bayes} = -q_\alpha\left(\{V^{(n)}(t)-1\}_{n=1}^N\right) \tag{B.7}$$

**Key property**: because $\kappa$ scales with the amount of historical data $T$, this method automatically widens the simulated distribution of outcomes — and therefore increases the estimated VaR — when little data is available, and converges to the plain Monte Carlo estimate (Eq. M.1–M.3) as $T\to\infty$, since the posterior on $(\mu,\Sigma)$ becomes arbitrarily tight. This makes the Bayesian VaR strictly more conservative than the Monte Carlo VaR under the same model, with the gap between them serving as a direct, interpretable measure of parameter uncertainty — something none of the previous methods can quantify.


### My own development
Note: while historical, parametric, and Monte Carlo VaR are standard industry methods, the Bayesian approach developed in this section is an **original** extension of this project, aimed at explicitly quantifying parameter uncertainty — something the other three methods cannot do.

## Finding the best portfolio — The Markowitz problem

Section "Parametric approach" showed that, under the normal approximation, the portfolio's expected value and variance at horizon $t$ can be written explicitly as functions of the weights $\mathbf{w}$ (Eq. P.4–P.5):

$$\mu_p(\mathbf{w}) = \mathbf{w}^T E[\exp(\mathbf{L})] - 1, \qquad \sigma_p^2(\mathbf{w}) = \mathbf{w}^T  \text{Cov}(\exp(\mathbf{L}))  \mathbf{w} \tag{K.1}$$

To lighten notation, we write $\mu_{ln} = E[\exp(\mathbf{L})]$ and $\Sigma_{ln} = \text{Cov}(\exp(\mathbf{L}))$ for the mean vector and covariance matrix already derived in Eq. P.1–P.3, so that:

$$\mu_p(\mathbf{w}) = \mathbf{w}^T\mu_{ln} - 1, \qquad \sigma_p^2(\mathbf{w}) = \mathbf{w}^T\Sigma_{ln}\mathbf{w} \tag{K.2}$$

Given this, a natural question follows: among all portfolios achieving a target expected return $R$, which one has the lowest possible risk (variance)? This is the classical Markowitz problem. Note that $R$ is the target expected portfolio value, so the target expected PnL is $R-1$.

The problem can be solved in closed form via Lagrange multipliers. Let $\lambda$ be the multiplier of the normalization constraint $\mathbf{1}^T\mathbf{w}=1$ and $\gamma$ the multiplier of the return constraint $\mu_{ln}^T\mathbf{w}=R$. The Lagrangian reads

$$\mathcal{L}(\mathbf{w},\lambda,\gamma) = \frac{1}{2}\mathbf{w}^T\Sigma_{ln}\mathbf{w} + \lambda\left(1 - \mathbf{1}^T\mathbf{w}\right) + \gamma\left(R - \mu_{ln}^T\mathbf{w}\right) \tag{K.3}$$

Setting its gradient with respect to $\mathbf{w}$ to zero gives

$$\Sigma_{ln}\mathbf{w} = \lambda\mathbf{1} + \gamma\mu_{ln} \qquad\Longrightarrow\qquad \mathbf{w} = \lambda b + \gamma a \tag{K.4}$$

with $a = \Sigma_{ln}^{-1}\mu_{ln}$ and $b = \Sigma_{ln}^{-1}\mathbf{1}$. Imposing the two constraints turns into two linear equations in $(\gamma,\lambda)$:

$$\begin{pmatrix}\mathbf{1}^T a & \mathbf{1}^T b\\ \mu_{ln}^T a & \mu_{ln}^T b\end{pmatrix}\begin{pmatrix}\gamma\\ \lambda\end{pmatrix} = \begin{pmatrix}1\\ R\end{pmatrix} \tag{K.5}$$

where the off-diagonal entries coincide, $\mathbf{1}^T a = \mu_{ln}^T b$, because $\Sigma_{ln}$ is symmetric. Solving (K.5) and substituting back into (K.4) yields the optimal weights. Since no non-negativity constraint is imposed, the optimal weights may be negative, i.e. the solution can include short positions.