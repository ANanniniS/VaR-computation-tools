# Portfolio VaR & Markowitz Toolkit

A small library for estimating portfolio risk (Value at Risk) and finding optimal portfolio weights, computed four different ways — including an original Bayesian extension that accounts for how much you actually know about your own estimates.

## The problem

If you hold a portfolio of assets, one of the most basic questions you can ask is: *how much could I lose?* Value at Risk (VaR) answers this at a given confidence level — say, "there's a 95% chance I won't lose more than X% tomorrow." It's the standard risk metric used by banks, funds, and regulators to size positions and set capital requirements.

But *how* you estimate VaR matters. Read it straight from historical data, and you're implicitly assuming the past is representative of the future. Fit a distribution to it, and you're assuming that distribution is the right one. Simulate it, and your result is only as good as the model behind the simulation. Each method makes a different trade-off between simplicity, realism, and the assumptions it's willing to make — and this repository implements four of them side by side, on the same portfolio, so those trade-offs are visible rather than hidden behind a single number.

## What this repository offers

- **Four different VaR methods** — historical, parametric, Monte Carlo, and Bayesian — cross-validated against each other on the same portfolio.
- **A closed-form solution to the classical Markowitz problem**, using standard Lagrange multipliers rather than numerical optimization.
- **[`math.md`](math.md)** — every formula used, derived from first assumptions to final implementation, with numbered equations cited directly from the code.
- **[`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb)** — the same toolkit run end-to-end on real market data, with the key empirical findings interpreted in context.

## The five functions, one philosophy

Everything in this library is exposed through five methods on a single `financial_context` class: `historical_VaR`, `parametric_VaR`, `montecarlo_VaR`, `bayesian_VaR`, and `markowitz_portfolio`. They're designed to be interchangeable — switching between methods should never require relearning an interface:

- **Same argument names across all methods**: `portfolio`, `horizon`, `alpha`, `verbose`, and `plot`. `N` (number of simulations) appears on the two methods that actually simulate — `montecarlo_VaR` and `bayesian_VaR` — since `historical_VaR` and `parametric_VaR` have no simulation step to control. If you know one function's signature, you know them all.
- **Same sign convention**: VaR is always returned as a positive number representing a loss, never a raw (and easy to misread) negative quantile.
- **Same validation**: every argument is checked up front, with the same rules applied consistently, so a mistake fails loudly at the call site instead of silently three lines of matrix algebra later.

```python
import numpy as np
from var_methods import financial_context

fc = financial_context("data/precios.csv")
portfolio = np.array([25_000] * 6)  # dollar amount invested in each of 6 tickers

fc.historical_VaR(portfolio)
fc.parametric_VaR(portfolio)
fc.montecarlo_VaR(portfolio)
fc.bayesian_VaR(portfolio)
```

## The Bayesian method — an original extension

Historical, parametric, and Monte Carlo VaR are the standard toolkit — but all three share a blind spot: they treat estimated parameters (expected return, covariance) as if they were known exactly, no matter how much or how little data supports them.

The Bayesian method implemented here is **this project's own extension**, built on a Normal-Inverse-Wishart model that treats those parameters as uncertain and integrates over that uncertainty when simulating outcomes. It makes one concrete promise: it should be **more conservative when less data is available**, and converge to the other methods as data accumulates.

This isn't just a theoretical claim — [`walkthrough.ipynb`](notebooks/walkthrough.ipynb) tests it directly, recomputing all four VaR estimates using between 1 and 12 months of history. The result confirms the promise: with a single month of data, the Bayesian estimate is unambiguously the most conservative of the four; as the sample grows, the gap narrows into noise, exactly as the underlying theory predicts (see [`math.md`](math.md) for the full derivation).

## Technical highlights

- **Fully vectorized, no Python-level loops.** Every simulation — including sampling thousands of covariance matrices from an Inverse-Wishart posterior — is done as batched NumPy/SciPy operations, not `for` loops over individual draws.
- **Horizon scaling is (almost) free.** Thanks to deriving the exact horizon-scaling of every relevant distribution analytically (see `math.md`), computing VaR at `horizon=1` costs essentially the same as `horizon=100` — the simulation cost depends on the number of samples `N`, not on the horizon.
- **Shared Cholesky decompositions.** Where a method needs two different covariance scalings from the same underlying matrix (e.g. the Bayesian method's parameter uncertainty vs. process noise), the expensive Cholesky factorization is computed once and reused, instead of repeating the most costly step of the simulation twice.
- **Parameter caching.** Quantities that are expensive to compute but constant across calls (e.g. the fitted mean and covariance of returns) are computed once, cached on the instance, and reused across every method that needs them — so calling `historical_VaR` and then `parametric_VaR` doesn't redo the same underlying estimation twice.
- **No caching across simulations — on purpose.** Unlike the parameter cache above, results are deliberately **not** cached or reused *between separate Monte Carlo or Bayesian calls*: each call draws an entirely fresh set of random samples. This was a conscious trade-off — reusing simulations across calls would be faster, but would introduce statistical dependence between measurements that are meant to be independent, which matters if you're comparing or averaging VaR estimates across calls.

## Documentation: `math.md` and the walkthrough notebook

Two documents complement the code, each answering a different question:

- **[`math.md`](math.md)** answers *why*: it lays out the price model, derives the mean and covariance behind each method from first assumptions, and numbers every equation so the code can cite it directly (e.g. `# Eq. B.3, math.md`).
- **[`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb)** answers *what happens in practice*: it runs the full toolkit on a real portfolio, compares the four VaR methods, runs the data-sensitivity experiment described above, and solves for the Markowitz-optimal portfolio — with every result interpreted, not just printed.

## Installation & quickstart

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone <repo-url>
cd portfolio-var
uv sync
```

To reproduce the analysis end-to-end, open the walkthrough notebook:

```bash
uv run jupyter lab notebooks/walkthrough.ipynb
```

### Project structure

```
portfolio-var/
├── README.md
├── math.md
├── pyproject.toml
├── data/
├── src/
│   ├── data_loader.py
│   └── var_methods.py
└── notebooks/
    └── walkthrough.ipynb
```
