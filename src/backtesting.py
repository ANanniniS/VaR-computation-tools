"""
Rolling-window backtesting of the VaR methods in `var_methods`.

Re-estimates the Value at Risk of a fixed portfolio on a rolling window of
historical prices, steps forward one day at a time, and checks how often
the realized loss exceeded the predicted VaR. Two exact hypothesis tests
then judge whether each method is well calibrated.

The maths is derived in the "Backtesting" section of `MATH.md`; the
`# Eq. T.x` comments in the code point back to its numbered equations.
"""

import pandas as pd
import numpy as np
from var_methods import FinancialContext
from scipy.stats import binomtest
from scipy.stats import fisher_exact
from tqdm import tqdm

def rolling_back_tests(datos,portfolio,horizon=1,alpha=0.05,N=20000,seed = None,window=100):
    """
    Backtest the four VaR methods on a rolling historical window.

    For each day `i` from `window` onwards, a `FinancialContext` is built
    from the previous `window` prices and used to estimate the portfolio
    VaR with all four methods (historical, parametric, Monte Carlo and
    Bayesian). The estimate is then compared against the realized P&L over
    the next `horizon` days to flag VaR breaches, and the sequence of
    breaches is assessed with two exact tests:

    - an exact binomial test on the number of breaches (in the spirit of
      Kupiec's proportion-of-failures test), checking that the breach rate
      matches `alpha`;
    - Fisher's exact test on the 2x2 table of consecutive breach/no-breach
      transitions (in the spirit of Christoffersen's independence test),
      checking that breaches do not cluster in time. Only computed when
      `horizon == 1`.

    Parameters
    ----------
    datos : pd.DataFrame
        Price data with a 'Date' first column followed by one adjusted
        closing price column per ticker.
    portfolio : array-like
        Dollar amount invested in each ticker (same order as the price
        columns of `datos`). Normalized internally to sum to 1.
    horizon : int, default=1
        Number of days ahead over which each VaR is computed and the
        realized loss is measured.
    alpha : float, default=0.05
        Significance level of the VaR (e.g. 0.05 for 95% confidence).
    N : int, default=20000
        Number of simulations used by the Monte Carlo and Bayesian methods.
    seed : int, optional
        Seed for the random number generator, for reproducibility. A
        distinct child seed is derived from it for each rolling window.
    window : int, default=100
        Number of past days used to estimate the model at each step.

    Returns
    -------
    pnl : np.ndarray, shape (num_days - window - horizon,)
        Realized portfolio P&L over `horizon` days, as a fraction of
        capital, for each rolling step.
    VaR_estimations : np.ndarray, shape (4, num_days - window - horizon)
        Estimated VaR at each step. Rows are ordered
        [historical, parametric, Monte Carlo, Bayesian].
    binom_test_pvalues : np.ndarray, shape (4,)
        Exact binomial test p-value for each method, in the same row order.
    fisher_test_pvalues : np.ndarray or None, shape (4,)
        Fisher's exact test p-value for each method, in the same row order,
        or None when `horizon != 1`.
    """
    num_data = datos.shape[0]
    VaR_estimations = np.zeros((4,num_data-window-horizon))
    pnl = np.zeros(num_data-window-horizon)
    data = datos.values[:,1:]
    portfolio = portfolio/np.sum(portfolio)

    seed_seq = np.random.SeedSequence(seed)
    window_seeds = seed_seq.generate_state(num_data - horizon - window)
    for i in tqdm(range(window, num_data - horizon), desc="Rolling backtest"):
        partial_context = FinancialContext(data = datos[i+1-window:i+1])
        window_seed = int(window_seeds[i-window])
        VaR_estimations[0,i-window] = partial_context.historical_VaR(portfolio,horizon,alpha,verbose=False,plot=False)
        VaR_estimations[1,i-window] = partial_context.parametric_VaR(portfolio,horizon,alpha,verbose=False,plot=False)
        VaR_estimations[2,i-window] = partial_context.montecarlo_VaR(portfolio,horizon,alpha,N,window_seed,verbose=False,plot=False)
        VaR_estimations[3,i-window] = partial_context.bayesian_VaR(portfolio,horizon,alpha,N,window_seed,verbose=False,plot=False)

        pnl[i-window] = np.sum(data[i+horizon]*portfolio/data[i]) - 1


    # Kupiec-style test: exact binomial test on the number of VaR breaches
    exceptions = np.zeros(VaR_estimations.shape,dtype=bool)
    binom_test_pvalues = np.zeros(4)
    n = len(pnl)
    for m in range(4):
        exceptions[m,:] = -pnl > VaR_estimations[m,:] # Eq. T.1
        k = exceptions[m,:].sum()
        binom_test_pvalues[m] = binomtest(k,n,p=alpha).pvalue # Eq. T.1 null: X ~ Binomial(n, alpha)


    # Christoffersen-style test: Fisher's exact test on the 2x2 table of
    # consecutive breach/no-breach transitions

    if horizon == 1:
        fisher_test_pvalues = np.zeros(4)
        for m in range(4):
            # transition counts n_ij of the two-state Markov chain (Eq. T.2)
            n00 = n01 = n10 = n11 = 0
            for i in range(n-1):
                if exceptions[m,i]:
                    if exceptions[m,i+1]:
                        n11+=1
                    else:
                        n10+=1
                else:
                    if exceptions[m,i+1]:
                        n01+=1
                    else:
                        n00+=1
            table = np.array([[n00,n01],[n10,n11]]) # 2x2 contingency table for Eq. T.3
            fisher_test_pvalues[m] = fisher_exact(table)[1] # Eq. T.3 null: pi_01 = pi_11
    else:
        fisher_test_pvalues = None

    return pnl,VaR_estimations,binom_test_pvalues,fisher_test_pvalues