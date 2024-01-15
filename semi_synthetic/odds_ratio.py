import statsmodels.api as sm
import pandas as pd
import numpy as np

def odds_ratio(X, Y, Z, data):
    # calculate the odds ratio assuming linearity

    formula = X + '~1+' + Y
    if len(Z) > 0:
        formula += '+' + '+'.join(Z)

    model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Binomial()).fit()

    return np.exp(model.params[1])