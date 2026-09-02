"""
Portfolio Value at Risk and Markowitz optimization from first principles.

Everything is exposed through the `FinancialContext` class, which loads a
price history once and then answers VaR and portfolio questions about it
with four methods: historical, parametric, Monte Carlo and Bayesian. The
full mathematical derivation of every formula lives in `MATH.md`, and the
`# Eq. X.Y` comments in the code point back to it.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, invwishart

class FinancialContext:
    """
    Financial context shared by all VaR and portfolio methods.

    Holds a price history and the model parameters estimated from it
    (log returns, Gaussian moments, lognormal price moments and the
    Normal-Inverse-Wishart posterior). Parameters are computed lazily on
    first use and cached, so building one context and calling several
    methods on it does not repeat work.
    """

    def __init__(
        self,
        data_location: str | None = None,
        data: pd.DataFrame | None = None
        ):
        """
        Load historical price data and prepare the financial context
        used by all VaR and portfolio methods.

        Exactly one of `data_location` or `data` must be provided.

        Parameters
        ----------
        data_location : str, optional
            Path to a CSV file with a 'Date' column and one column
            of adjusted closing prices per ticker.
        data : pd.DataFrame, optional
            Price data already in memory. Dates may be either a 'Date'
            column or the index. Useful for rolling-window backtesting,
            where writing one CSV per window would be wasteful.
        """
        self._validate_init(data_location, data)

        if not (data_location is None):
            data = pd.read_csv(data_location) # we import the data


        self.dates = data["Date"] #we save the dates

        #we save the tickers and the data
        data = data.drop(columns = ["Date"]) #for the historical
        self.data = data
        self.tickers = data.columns
        self.N_tickers = len(self.tickers)

        #In order to save computations we save some parameters when they are computed
        self._log_returns_computed = False
        self._gaussian_parameters_computed = False
        self._lognorm_moments_computed = False
        self._wishart_params_computed = False


    def historical_VaR(
            self,
            portfolio: np.ndarray,
            horizon: int = 1,
            alpha: float = 0.05,
            verbose: bool = True,
            plot: bool = True
            ) -> float:
        """
        Compute the historical Value at Risk (VaR) of a portfolio.

        Uses the empirical distribution of past portfolio profit and
        loss (P&L), without assuming any parametric distribution.

        Parameters
        ----------
        portfolio : array-like
            Dollar amount invested in each ticker (same order as self.tickers).
        horizon : int, default=1
            Number of days ahead over which the VaR is computed.
        alpha : float, default=0.05
            Significance level (e.g. 0.05 for 95% confidence).
        verbose : bool, default=True
            If True, print the resulting VaR as a percentage.
        plot : bool, default=True
            If True, display a histogram of the P&L distribution,
            with the loss tail beyond the VaR highlighted in red.

        Returns
        -------
        float
            Historical VaR, expressed as a positive fraction of capital
            (e.g. 0.023 means a potential loss of 2.3%).
        """
        self._validate_historical_VaR(portfolio,horizon,alpha,verbose,plot)
        
        portfolio = portfolio/np.sum(portfolio)

        portfolio_pnl = (self.data*portfolio/self.data.shift(horizon)).sum(axis=1)[horizon:] - 1
        VaR_hist = -portfolio_pnl.quantile(alpha) # Eq. H.1, H.2

        if verbose:
            print(f"historical-VaR: {round(VaR_hist*100,2)}%")
        if plot:
            counts, bins, patches = plt.hist(portfolio_pnl, bins=30)
            threshold = -VaR_hist 
            for patch, left_edge in zip(patches, bins):
                if left_edge < threshold:
                    patch.set_facecolor("red")
            plt.show()
        return VaR_hist

    def parametric_VaR(
            self,
            portfolio: np.ndarray,
            horizon: int = 1,
            alpha: float = 0.05,
            verbose: bool = True,
            plot: bool = True
            ) -> float:
        """
        Compute the parametric (variance-covariance) Value at Risk of a portfolio.

        Assumes portfolio returns follow a distribution implied by a
        multivariate lognormal model of prices, and computes VaR
        analytically from its mean and standard deviation.

        Parameters
        ----------
        portfolio : array-like
            Dollar amount invested in each ticker (same order as self.tickers).
        horizon : int, default=1
            Number of days ahead over which the VaR is computed.
        alpha : float, default=0.05
            Significance level (e.g. 0.05 for 95% confidence).
        verbose : bool, default=True
            If True, print the resulting VaR as a percentage.
        plot : bool, default=True
            If True, display the fitted normal density, with the loss
            tail beyond the VaR highlighted in red.

        Returns
        -------
        float
            Parametric VaR, expressed as a positive fraction of capital.
        """
        self._validate_parametric_VaR(portfolio,horizon,alpha,verbose,plot)


        w = portfolio/sum(portfolio)

        horizon_mu,horizon_cov = self._lognorm_moments_horizon(horizon) # Eq. P.1, P.2, P.3

        mean_R = np.dot(w,horizon_mu) - 1 # Eq. P.4
        std_R = np.sqrt( np.dot(w,np.dot(horizon_cov,w)))  # Eq. P.5

        z = norm.ppf(alpha) # Eq. P.6

        VaR_para = - (mean_R + z*std_R) # Eq. P.6

        if verbose:
            print(f"parametric-VaR: {round(VaR_para*100,2)}%")

        if plot:
            x = np.linspace(mean_R - 4 * std_R, mean_R + 4 * std_R, 300)
            y = norm.pdf(x, loc=mean_R, scale=std_R)

            plt.plot(x, y, color="black")
            plt.fill_between(x, y, where=(x < -VaR_para), color="red")
            plt.fill_between(x, y, where=(x >= -VaR_para), color="steelblue")
            plt.axvline(-VaR_para, color="red", linestyle="--", label=f"VaR ({alpha*100:.0f}%)")
            plt.legend()
            plt.show()

        return VaR_para

    def montecarlo_VaR(
            self,
            portfolio: np.ndarray,
            horizon: int = 1,
            alpha: float = 0.05,
            N: int = 100000,
            seed: int | None = None,
            verbose : bool = True,
            plot : bool = True
            ):
        """
        Compute the Value at Risk of a portfolio via Monte Carlo simulation.

        Simulates N possible portfolio outcomes under the same lognormal
        model used by parametric_VaR, and estimates VaR as the empirical
        quantile of the simulated results. Serves as a validation of the
        parametric method: both should converge to the same value.

        Parameters
        ----------
        portfolio : array-like
            Dollar amount invested in each ticker (same order as self.tickers).
        horizon : int, default=1
            Number of days ahead over which the VaR is computed.
        alpha : float, default=0.05
            Significance level (e.g. 0.05 for 95% confidence).
        N : int, default=100000
            Number of Monte Carlo simulations to run.
        seed : int, optional
            Seed for the random number generator, for reproducibility.
        verbose : bool, default=True
            If True, print the resulting VaR as a percentage.
        plot : bool, default=True
            If True, display a histogram of simulated outcomes, with the
            loss tail beyond the VaR highlighted in red.

        Returns
        -------
        float
            Simulated VaR, expressed as a positive fraction of capital.
        """
        self._validate_montecarlo_VaR(portfolio,horizon,alpha,N,seed,verbose,plot)

        w= portfolio/sum(portfolio)

        if not self._gaussian_parameters_computed:
            self._gaussian_parameters()

        rng = np.random.default_rng(seed)

        simulation_result = np.exp(rng.multivariate_normal(horizon*self.mu,horizon*self.volatility,size=N))@w-1 # Eq. M.1, M.2
        VaR_mont = -np.quantile(simulation_result,q=alpha) # Eq. M.3

        if verbose:
            print(f"montecarlo-VaR: {round(VaR_mont * 100, 2)}%")

        if plot:
            counts, bins, patches = plt.hist(simulation_result, bins=30)
            threshold = -VaR_mont  # VaR is positive, but the actual percentile is negative
            for patch, left_edge in zip(patches, bins):
                if left_edge < threshold:
                    patch.set_facecolor("red")
            plt.show()
        return VaR_mont

    def bayesian_VaR(
            self,
            portfolio: np.ndarray,
            horizon: int = 1,
            alpha: float = 0.05,
            N: int = 100000,
            seed: int | None = None,
            verbose : bool = True,
            plot : bool = True
        ):
        """
        Compute the Value at Risk of a portfolio under a Bayesian model.

        This is a more advanced but well-documented technique: the
        Normal-Inverse-Wishart posterior predictive method, long standard
        in Bayesian portfolio analysis (e.g. Klein & Bawa 1976, Jorion
        1986), implemented here from scratch behind the same interface as
        the other VaR methods. Unlike montecarlo_VaR, it also accounts for
        parameter uncertainty, producing a wider (more conservative) VaR
        estimate when historical data is scarce.

        Parameters
        ----------
        portfolio : array-like
            Dollar amount invested in each ticker (same order as self.tickers).
        horizon : int, default=1
            Number of days ahead over which the VaR is computed.
        alpha : float, default=0.05
            Significance level (e.g. 0.05 for 95% confidence).
        N : int, default=100000
            Number of posterior samples to draw.
        seed : int, optional
            Seed for the random number generator, for reproducibility.
        verbose : bool, default=True
            If True, print the resulting VaR as a percentage.
        plot : bool, default=True
            If True, display a histogram of simulated outcomes, with the
            loss tail beyond the VaR highlighted in red.

        Returns
        -------
        float
            Bayesian VaR, expressed as a positive fraction of capital.
        """
        self._validate_bayesian_VaR(portfolio,horizon,alpha,N,seed,verbose,plot)

        w = portfolio/sum(portfolio)

        if not self._wishart_params_computed:
            self._wishart_params()
        if not self._gaussian_parameters_computed:
            self._gaussian_parameters()

        rng = np.random.default_rng(seed)

        sigmas = invwishart.rvs(df=self.nu, scale=self.psi, size=N, random_state=rng) # Eq. B.3
        L_base = np.linalg.cholesky(sigmas)


        zs = rng.standard_normal((N,self.N_tickers,2))

        mus = self.mu + np.einsum('nij,nj->ni',L_base/np.sqrt(self.kappa),zs[:,:,0]) # Eq. B.4

        simulation_result = np.exp(horizon*mus + np.einsum("nij,nj->ni",np.sqrt(horizon)*L_base,zs[:,:,1]))@w - 1 # Eq. B.5, B.6


        VaR_bayes = -np.quantile(simulation_result, q=alpha) # Eq. B.7

        if verbose:
            print(f"bayesian-VaR: {round(VaR_bayes * 100, 2)}%")
        if plot:
            counts, bins, patches = plt.hist(simulation_result, bins=30)
            threshold = -VaR_bayes
            for patch, left_edge in zip(patches, bins):
                if left_edge < threshold:
                    patch.set_facecolor("red")
            plt.show()

        return VaR_bayes

    def markowitz_portfolio(
            self,
            expected_return: float,
            horizon: int = 1,
            verbose: bool = True
            ) -> np.ndarray:
        """
        Compute the classical Markowitz minimum-variance portfolio for a
        given expected return, solved in closed form via Lagrange multipliers.

        Uses the exact covariance of simple returns implied by the lognormal
        price model (not the log-return covariance), so it is consistent
        with parametric_VaR.

        Parameters
        ----------
        expected_return : float
            Target expected return of the portfolio over the given horizon.
        horizon : int, default=1
            Number of days ahead over which the optimization is performed.
        verbose : bool, default=True
            If True, print the resulting markowitz portfolio.


        Returns
        -------
        np.ndarray
            Optimal portfolio weights (summing to 1). May include negative
            weights (short positions), since no non-negativity constraint
            is imposed.
        """
        self._validate_markowitz_portfolio(expected_return,horizon,verbose)

        horizon_mu, horizon_cov = self._lognorm_moments_horizon(horizon) # Eq. K.1, K.2

        rhs = np.column_stack([horizon_mu, np.ones_like(horizon_mu)])
        sol = np.linalg.solve(horizon_cov, rhs) # Eq. K.4

        mu_hat = sol[:, 0]
        one_hat = sol[:, 1]

        xi = np.linalg.solve(
            np.array([[np.sum(mu_hat),np.sum(one_hat)],
                      [np.dot(mu_hat,horizon_mu),np.dot(one_hat,horizon_mu)]]),
            np.array([1,expected_return])
        ) # Eq. K.5

        markowitz_weights = xi[0]*mu_hat + xi[1]*one_hat # Eq. K.4

        if verbose:
            print(f"Markowitz' portfolio: {markowitz_weights}")

        return markowitz_weights

    ######################### Finantial Context Computations ##############################

    def _log_returns(self):
        """
        Compute the daily log returns of every ticker from the price data.

        Stores the result (a matrix with one row per day and one column
        per ticker) in self.l and marks it as computed.
        """
        self.l = np.log(self.data/self.data.shift(1))[1:].values
        self._log_returns_computed = True
        
    def _gaussian_parameters(self):
        """
        Estimate the Gaussian model of the log returns.

        Computes the vector of means self.mu and the covariance matrix
        self.volatility of the daily log returns, and marks them as
        computed.
        """
        if not self._log_returns_computed:
            self._log_returns()
        self.mu = np.mean(self.l,axis=0)
        self.volatility = np.cov(self.l,rowvar=False,ddof=1)
        self._gaussian_parameters_computed = True
        return None

    def _lognorm_moments(self):
        """
        Compute the moments of the implied multivariate lognormal price model.

        Stores the expected future price of each ticker in self.mu_ln, the
        outer product of these expectations in self.cov_exp, and the
        expectation of pairwise price products in self.A_ln.
        """
        if not self._gaussian_parameters_computed:
            self._gaussian_parameters()
        self.mu_ln = np.exp(self.mu + np.diag(self.volatility)/2) # Eq. P.1
        self.cov_exp = np.outer(self.mu_ln,self.mu_ln) # Eq. P.3
        self.A_ln = np.exp(self.volatility)*self.cov_exp # Eq. P.2
        self._lognorm_moments_computed = True

    def _lognorm_moments_horizon(self,horizon):
        """
        Scale the lognormal moments from one day to a given horizon.

        Parameters
        ----------
        horizon : int
            Number of days ahead over which the moments are scaled.

        Returns
        -------
        tuple of np.ndarray
            Expected price vector and covariance matrix of prices over
            the given horizon.
        """

        if not self._lognorm_moments_computed:
            self._lognorm_moments()

        return self.mu_ln**horizon, self.A_ln**horizon - self.cov_exp**horizon # Eq. P.1, P.2, P.3

    def _wishart_params(self):
        """
        Compute the Normal-Inverse-Wishart posterior parameters.

        Stores the sum of squares matrix in self.psi, the effective sample
        size in self.kappa, and the degrees of freedom in self.nu. These
        are used by bayesian_VaR to draw posterior covariance matrices.
        """
        if not self._log_returns_computed:
            self._log_returns()
        if not self._gaussian_parameters_computed:
            self._gaussian_parameters()
        self.psi = (self.l - self.mu).T@(self.l-self.mu) # Eq. B.2
        self.kappa = self.l.shape[0] # Eq. B.2
        self.nu = self.kappa -1  # Eq. B.2
        self._wishart_params_computed = True

    ######################### Validation Functions ##############################

    def _validate_init(self, data_location, data):
        """Validate the arguments of __init__."""
        if (data_location is None) == (data is None):
            raise ValueError(
                "Exactly one of `data_location` or `data` must be provided."
            )
        if data_location is not None and not isinstance(data_location, str):
            raise ValueError(
                f"data_location must be a str, got {type(data_location)}"
            )
        if data is not None:
            if not isinstance(data, pd.DataFrame):
                raise ValueError(
                    f"data must be a pandas DataFrame, got {type(data)}"
                )
            if "Date" not in data.columns:
                raise ValueError("data must have a 'Date' column")
            if data.shape[1] < 2:
                raise ValueError("data must have at least one price column besides 'Date'")

    def _validate_portfolio(self,portfolio: np.ndarray) -> None:
        """Validate the portfolio argument."""
        if portfolio.shape[0] != self.N_tickers:
            raise ValueError(f"The portfolio must have {self.N_tickers} elements, got {portfolio.shape[0]}")
        if np.any(portfolio < 0):
            raise ValueError("portfolio cannot have negative values")

    def _validate_alpha(self,alpha):
        """Validate the alpha argument."""
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1. Got {alpha}")

    def _validate_horizon(self,horizon):
        """Validate the horizon argument."""
        if not isinstance(horizon,int) or horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

    def _validate_N(self, N):
        """Validate the N argument (number of Monte Carlo simulations)."""
        if not isinstance(N, int) or N < 1:
            raise ValueError(f"N must be a positive integer, got {N}")

    def _validate_bool_flag(self, value, name):
        """Validate a generic boolean argument (verbose, plot, etc.), given its name for the error message."""
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be bool, got {type(value)}")

    def _validate_expected_return(self,expected_return):
        """Validate the expected_return argument."""
        if not (isinstance(expected_return,int) or isinstance(expected_return,float)):
            raise ValueError(f"Expected return must be a float (or int), it is {type(expected_return)}")
        if expected_return < 0:
            raise ValueError(f"Expected return must not be a negative number. It is {expected_return}")

    def _validate_seed(self, seed):
        """Validate the seed argument (RNG seed, for reproducibility)."""
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be an int (or None), got {type(seed)}")

    def _validate_historical_VaR(self, portfolio, horizon, alpha, verbose, plot):
        """Validate the arguments of historical_VaR."""
        self._validate_portfolio(portfolio)
        self._validate_horizon(horizon)
        self._validate_alpha(alpha)
        self._validate_bool_flag(verbose, "verbose")
        self._validate_bool_flag(plot, "plot")

    def _validate_parametric_VaR(self, portfolio, horizon, alpha, verbose, plot):
        """Validate the arguments of parametric_VaR."""
        self._validate_portfolio(portfolio)
        self._validate_horizon(horizon)
        self._validate_alpha(alpha)
        self._validate_bool_flag(verbose, "verbose")
        self._validate_bool_flag(plot, "plot")

    def _validate_montecarlo_VaR(self, portfolio, horizon, alpha, N, seed, verbose, plot):
        """Validate the arguments of montecarlo_VaR."""
        self._validate_portfolio(portfolio)
        self._validate_horizon(horizon)
        self._validate_alpha(alpha)
        self._validate_N(N)
        self._validate_seed(seed)
        self._validate_bool_flag(verbose, "verbose")
        self._validate_bool_flag(plot, "plot")

    def _validate_bayesian_VaR(self, portfolio, horizon, alpha, N, seed, verbose, plot):
        """Validate the arguments of bayesian_VaR."""
        self._validate_portfolio(portfolio)
        self._validate_horizon(horizon)
        self._validate_alpha(alpha)
        self._validate_N(N)
        self._validate_seed(seed)
        self._validate_bool_flag(verbose, "verbose")
        self._validate_bool_flag(plot, "plot")

    def _validate_markowitz_portfolio(self,expected_return,horizon,verbose):
        """Validate the arguments of markowitz_portfolio."""
        self._validate_expected_return(expected_return)
        self._validate_horizon(horizon)
        self._validate_bool_flag(verbose, "verbose")


if __name__ == "__main__":
    fc = FinancialContext("data/precios.csv")