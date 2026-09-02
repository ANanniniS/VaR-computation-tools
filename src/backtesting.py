import pandas as pd
import numpy as np
from var_methods import FinancialContext
from scipy.stats import binomtest
from scipy.stats import fisher_exact

def rolling_back_tests(datos,portfolio,horizon=1,alpha=0.05,N=20000,window=100):
    num_data = datos.shape[0]
    VaR_estimations = np.zeros((4,num_data-window-horizon))
    pnl = np.zeros(num_data-window-horizon)
    data = datos.values[:,1:]
    portfolio = portfolio/np.sum(portfolio)

    for i in range(window,num_data-horizon):
        partial_context = FinancialContext(data = datos[i+1-window:i+1])
        VaR_estimations[0,i-window] = partial_context.historical_VaR(portfolio,horizon,alpha,verbose=False,plot=False)
        VaR_estimations[1,i-window] = partial_context.parametric_VaR(portfolio,horizon,alpha,verbose=False,plot=False)
        VaR_estimations[2,i-window] = partial_context.montecarlo_VaR(portfolio,horizon,alpha,N,verbose=False,plot=False)
        VaR_estimations[3,i-window] = partial_context.bayesian_VaR(portfolio,horizon,alpha,N,verbose=False,plot=False)

        pnl[i-window] = np.sum(data[i+horizon]*portfolio/data[i]) - 1


    #test de kupiec
    exceptions = np.zeros(VaR_estimations.shape,dtype=bool)
    binom_test_pvalues = np.zeros(4)
    n = len(pnl)
    for m in range(4):
        exceptions[m,:] = -pnl > VaR_estimations[m,:]
        k = exceptions[m,:].sum()
        binom_test_pvalues[m] = binomtest(k,n,p=alpha).pvalue


    #test de christoffer

    if horizon == 1:
        fisher_test_pvalues = np.zeros(4)
        for m in range(4):
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
            table = np.array([[n00,n01],[n10,n11]])
            fisher_test_pvalues[m] = fisher_exact(table)[1]
    else:
        fisher_test_pvalues = None

    return pnl,VaR_estimations,binom_test_pvalues,fisher_test_pvalues
    

def kupiec_test(VaR_estimation,pnl,sign_level):
    x = -pnl > VaR_estimation
