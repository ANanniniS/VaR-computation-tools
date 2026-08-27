import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, invwishart, multivariate_normal
from scipy.optimize import minimize
import time

def identity(x):
    return x

class financial_context:
    def __init__(self,data_location: str):
        """
        Load historical price data and prepare the financial context
        used by all VaR and portfolio methods.

        Parameters
        ----------
        data_location : str
            Path to a CSV file with a 'Date' column and one column
            of adjusted closing prices per ticker.
        """

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
        
        x = portfolio/((self.data.iloc[0].values)*sum(portfolio))

        portfolio_pnl = (x*(self.data - self.data.shift(horizon)))[horizon:].sum(axis=1)

        VaR_hist = -portfolio_pnl.quantile(alpha)

        if verbose:
            print(f"historical-VaR: {round(VaR_hist*100,2)}%")
        if plot:
            counts, bins, patches = plt.hist(portfolio_pnl, bins=30)
            umbral = -VaR_hist 
            for patch, left_edge in zip(patches, bins):
                if left_edge < umbral:
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

        horizon_mu,horizon_cov = self._lognorm_moments_horizon(horizon)

        mean_R = np.dot(w,horizon_mu) - 1
        std_R = np.sqrt( np.dot(w,np.dot(horizon_cov,w))) 

        z = norm.ppf(alpha)

        VaR_para = - (mean_R + z*std_R)

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
        self._validate_montecarlo_VaR(portfolio,horizon,alpha,N,verbose,plot)

        w= portfolio/sum(portfolio)

        if not self._gaussian_parameters_computed:
            self._gaussian_parameters()
        simulation_result = np.exp(np.random.multivariate_normal(horizon*self.mu,horizon*self.volatility,size=N))@w-1
        VaR_mont = -np.quantile(simulation_result,q=alpha)

        if verbose:
            print(f"montecarlo-VaR: {round(VaR_mont * 100, 2)}%")

        if plot:
            counts, bins, patches = plt.hist(simulation_result, bins=30)
            umbral = -VaR_mont  # var_hist es positivo, pero el percentil real es negativo
            for patch, left_edge in zip(patches, bins):
                if left_edge < umbral:
                    patch.set_facecolor("red")
            plt.show()
        return VaR_mont

    def bayesian_VaR(
            self,
            portfolio: np.ndarray,
            horizon: int = 1,
            alpha: float = 0.05,
            N: int = 100000,
            verbose : bool = True,
            plot : bool = True
        ):
        """
        Compute the Value at Risk of a portfolio under a Bayesian model.

        Unlike montecarlo_VaR, this method also accounts for parameter
        uncertainty. This produces a wider (more
        conservative) VaR estimate when historical data is scarce.

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
        self._validate_bayesian_VaR(portfolio,horizon,alpha,N,verbose,plot)

        w = portfolio/sum(portfolio)

        if not self._wishart_params_computed:
            self._wishart_params()
        if not self._gaussian_parameters_computed:
            self._gaussian_parameters()



        sigmas = invwishart.rvs(df=self.nu, scale=self.psi, size=N)
        L_base = np.linalg.cholesky(sigmas)

        zs = np.random.standard_normal((N,self.N_tickers,2))

        mus = horizon*self.mu + np.einsum('nij,nj->ni',L_base/np.sqrt(self.kappa),zs[:,:,0]) #valors medios

        simulation_result = np.exp(mus + np.einsum("nij,nj->ni",np.sqrt(horizon)*L_base,zs[:,:,1]))@w - 1


        VaR_bayes = -np.quantile(simulation_result, q=alpha)

        if verbose:
            print(f"bayesian-VaR: {round(VaR_bayes * 100, 2)}%")
        if plot:
            counts, bins, patches = plt.hist(simulation_result, bins=30)
            umbral = -VaR_bayes
            for patch, left_edge in zip(patches, bins):
                if left_edge < umbral:
                    patch.set_facecolor("red")
            plt.show()

        return VaR_bayes

    def markowitz_portfolio(
            self,
            expected_return: float,
            horizon: int = 1
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

        Returns
        -------
        np.ndarray
            Optimal portfolio weights (summing to 1). May include negative
            weights (short positions), since no non-negativity constraint
            is imposed.
        """
        self._validate_markowitz_portfolio(expected_return,horizon)

        horizon_mu, horizon_cov = self._lognorm_moments_horizon(horizon)

        rhs = np.column_stack([horizon_mu, np.ones_like(horizon_mu)])
        sol = np.linalg.solve(horizon_cov, rhs)

        mu_hat = sol[:, 0]
        one_hat = sol[:, 1]

        xi = np.linalg.solve(
            np.array([[np.sum(mu_hat),np.sum(one_hat)],
                      [np.dot(mu_hat,horizon_mu),np.dot(one_hat,horizon_mu)]]),
            np.array([1,expected_return])
        )

        markowitz_weights = xi[0]*mu_hat + xi[1]*one_hat

        print(np.sum(markowitz_weights),np.dot(horizon_mu,markowitz_weights))

        return markowitz_weights

    ######################### Finantial Context Computations ##############################

    def _log_returns(self):
        self.l = np.log(self.data/self.data.shift(1))[1:].values
        self._log_returns_computed = True
        
    def _gaussian_parameters(self):
        if not self._log_returns_computed:
            self._log_returns()
        self.mu = np.mean(self.l,axis=0)
        self.volatility = np.cov(self.l,rowvar=False,ddof=1)
        self._gaussian_parameters_computed = True
        return None

    def _lognorm_moments(self):
        if not self._gaussian_parameters_computed:
            self._gaussian_parameters()
        self.mu_ln = np.exp(self.mu + np.diag(self.volatility)/2)
        self.cov_exp = np.outer(self.mu_ln,self.mu_ln)
        self.A_ln = np.exp(self.volatility)*self.cov_exp
        self._lognorm_moments_computed = True

    def _lognorm_moments_horizon(self,horizon):

        if not self._lognorm_moments_computed:
            self._lognorm_moments()

        return self.mu_ln**horizon, self.A_ln**horizon - self.cov_exp**horizon

    def _wishart_params(self):
        if not self._log_returns_computed:
            self._log_returns()
        if not self._gaussian_parameters_computed:
            self._gaussian_parameters()
        self.psi = (self.l - self.mu).T@(self.l-self.mu)
        self.kappa = self.l.shape[0]
        self.nu = self.kappa -1 
        self._wishart_params_computed = True

    ######################### Validation Functions ##############################

    def _validate_portfolio(self,portfolio: np.ndarray) -> None:
        """Valida el argumento portfolio."""
        if portfolio.shape[0] != self.N_tickers:
            raise ValueError(f"El portfolio debe tener {self.N_tickers} elementos, recibió {portfolio.shape[0]}")
        if np.any(portfolio < 0):
            raise ValueError("portfolio no puede tener valores negativos")

    def _validate_alpha(self,alpha):
        """Valida el argumento alpha."""
        if not (0 < alpha < 1):
            raise ValueError(f"alpha debe estar entre 0 y 1. Recibió {alpha}")

    def _validate_horizon(self,horizon):
        """Valida el argumento horizon."""
        if not isinstance(horizon,int) or horizon < 1:
            raise ValueError(f"horizon debe ser un entero positivo, recibió {horizon}")

    def _validate_N(self, N):
        """Valida el argumento N (cantidad de simulaciones Monte Carlo)."""
        if not isinstance(N, int) or N < 1:
            raise ValueError(f"N debe ser un entero positivo, recibió {N}")

    def _validate_bool_flag(self, value, name):
        """Valida un argumento booleano genérico (verbose, plot, etc.), dado su nombre para el mensaje de error."""
        if not isinstance(value, bool):
            raise ValueError(f"{name} debe ser bool, recibió {type(value)}")

    def _validate_expected_return(self,expected_return):
        if not (isinstance(expected_return,int) or isinstance(expected_return,float)):
            raise ValueError(f"Expected return must be a float (or int), it is {type(expected_return)}")
        if expected_return < 0:
            raise ValueError(f"Expected return must not be a negative number. It is {expected_return}")

    def _validate_historical_VaR(self, portfolio, horizon, alpha, verbose, plot):
        """Valida los argumentos de historical_VaR."""
        self._validate_portfolio(portfolio)
        self._validate_horizon(horizon)
        self._validate_alpha(alpha)
        self._validate_bool_flag(verbose, "verbose")
        self._validate_bool_flag(plot, "plot")

    def _validate_parametric_VaR(self, portfolio, horizon, alpha, verbose, plot):
        """Valida los argumentos de parametric_VaR."""
        self._validate_portfolio(portfolio)
        self._validate_horizon(horizon)
        self._validate_alpha(alpha)
        self._validate_bool_flag(verbose, "verbose")
        self._validate_bool_flag(plot, "plot")

    def _validate_montecarlo_VaR(self, portfolio, horizon, alpha, N, verbose, plot):
        """Valida los argumentos de montecarlo_VaR."""
        self._validate_portfolio(portfolio)
        self._validate_horizon(horizon)
        self._validate_alpha(alpha)
        self._validate_N(N)
        self._validate_bool_flag(verbose, "verbose")
        self._validate_bool_flag(plot, "plot")

    def _validate_bayesian_VaR(self, portfolio, horizon, alpha, N, verbose, plot):
        """Valida los argumentos de bayesian_VaR_alt."""
        self._validate_portfolio(portfolio)
        self._validate_horizon(horizon)
        self._validate_alpha(alpha)
        self._validate_N(N)
        self._validate_bool_flag(verbose, "verbose")
        self._validate_bool_flag(plot, "plot")

    def _validate_markowitz_portfolio(self,expected_return,horizon):
        self._validate_expected_return(expected_return)
        self._validate_horizon(horizon)


if __name__ == "__main__":
    fc = financial_context("data/precios.csv")
    # --- Test 1: utilidad identidad, sanity check contra Markowitz ---
    identity = lambda r: r**3
    w_identity = fc.portfolio_maximization(perceived_value=identity, N=1000000)

    print("Pesos (utilidad identidad):", np.round(w_identity, 3))
    print("Suma de pesos:", round(w_identity.sum(), 4))  # debería dar ~1.0

    # --- Test 2: utilidad logarítmica (cóncava, penaliza riesgo) ---
    log_utility = lambda r: np.log(1 + r)
    w_log = fc.portfolio_maximization(perceived_value=log_utility, N=20000)

    print("\nPesos (utilidad log):", np.round(w_log, 3))
    print("Suma de pesos:", round(w_log.sum(), 4))

    # --- Test 3: comparar el rating de ambos portafolios bajo la misma utilidad ---
    rating_identity = fc.portfolio_rating(w_identity, perceived_value=log_utility, N=50000)
    rating_log = fc.portfolio_rating(w_log, perceived_value=log_utility, N=50000)

    print(f"\nUtilidad log del portafolio 'identity': {rating_identity:.5f}")
    print(f"Utilidad log del portafolio 'log':      {rating_log:.5f}")
    # el portafolio optimizado para utilidad log debería ganarle (o empatar) al otro