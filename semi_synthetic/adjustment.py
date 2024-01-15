"""
This code is copied from Prof. Rohit Bhattacharya. It contains functions for calculating the causal
effect of a treatment variable and confidence intervals via bootstrap sampling.
"""

import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.special import expit

def backdoor_adjustment(Y, A, Z, data):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via backdoor adjustment
    
    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    formula: list of variable names included the backdoor adjustment set
    data: pandas dataframe
    
    Return
    ------
    ACE: float corresponding to the causal effect
    """
    
    formula = Y + "~" + A
    if len(Z) > 0:
        formula += " + " + "+".join(Z)
    
    model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian()).fit()
    #print(model.summary())
    data_A0 = data.copy()
    data_A1 = data.copy()
    data_A0[A] = 0
    data_A1[A] = 1
    return(np.mean(model.predict(data_A1)) - np.mean(model.predict(data_A0)))

def backdoor_adjustment_binary(Y, A, Z, data):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via backdoor adjustment when the outcome is binary
    
    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    formula: list of variable names included the backdoor adjustment set
    data: pandas dataframe
    
    Return
    ------
    ACE: float corresponding to the causal effect
    """
    
    formula = Y + "~" + A
    if len(Z) > 0:
        formula += " + " + "+".join(Z)
    
    model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Binomial()).fit()
    data_A0 = data.copy()
    data_A1 = data.copy()
    data_A0[A] = 0
    data_A1[A] = 1
    return(np.mean(model.predict(data_A1)) - np.mean(model.predict(data_A0)))

def compute_confidence_intervals(Y, A, Z, data, method_name, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals for backdoor adjustment via bootstrap
    
    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """
    
    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []
    
    for i in range(num_bootstraps):
        
        # resample the data with replacement
        data_sampled = data.sample(len(data), replace=True)
        data_sampled.reset_index(drop=True, inplace=True)
        
        # add estimate from resampled data
        if method_name == "backdoor":
            estimates.append(backdoor_adjustment(Y, A, Z, data_sampled))
        elif method_name == "backdoor_binary":
            estimates.append(backdoor_adjustment_binary(Y, A, Z, data_sampled))
        else:
            print("Invalid method")
            estimates.append(1)

    # calculate the quantiles
    quantiles = np.quantile(estimates, q=[Ql, Qu])
    q_low = quantiles[0]
    q_up = quantiles[1]
    
    return q_low, q_up