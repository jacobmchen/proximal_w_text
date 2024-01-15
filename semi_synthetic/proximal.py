import pandas as pd
import numpy as np
from scipy.special import expit
import statsmodels.api as sm

def proximal_find_ace(A, Y, W, Z, covariates, data):
    # fit a model W~A+Z
    formula = W+"~"+A+"+"+Z
    if len(covariates) > 0:
        formula += '+' + '+'.join(covariates)
    model1 = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Binomial()).fit()

    # make predictions for What
    What = model1.predict(data)
    data["What"] = What

    # fit a model Y~A+What
    formula = Y+"~"+A+"+What"
    if len(covariates) > 0:
        formula += '+' + '+'.join(covariates)
    model2 = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian()).fit()

    # the ACE is the coefficient for A in model2
    return model2.params[A]

def compute_confidence_intervals(A, Y, W, Z, covariates, data, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals for backdoor adjustment via bootstrap
    
    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """
    
    Ql = alpha/2
    Qu = 1 - alpha/2
    # two lists for the two indexes of output
    estimates = []
    
    for i in range(num_bootstraps):
        
        # resample the data with replacement
        data_sampled = data.sample(len(data), replace=True)
        data_sampled.reset_index(drop=True, inplace=True)
        
        # add estimate from resampled data
        output = proximal_find_ace(A, Y, W, Z, covariates, data_sampled)
        estimates.append(output)

    # calculate the quantiles
    quantiles = np.quantile(estimates, q=[Ql, Qu])
    q_low = quantiles[0]
    q_up = quantiles[1]
    
    return (q_low, q_up)