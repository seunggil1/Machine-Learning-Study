import pandas as pd
from sklearn.datasets import load_boston

import statsmodels.formula.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

boston_ds = load_boston()

boston = pd.DataFrame(boston_ds.data, columns=boston_ds.feature_names)
boston['target'] = boston_ds.target
boston.head()

result = sm.ols(formula = 'target ~ CRIM + ZN + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT', data = boston).fit()
result.summary()
