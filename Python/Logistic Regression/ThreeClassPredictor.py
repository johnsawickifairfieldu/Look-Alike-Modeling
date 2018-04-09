import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('data/berkeleySCF/sub-data.txt', low_memory=False)
X = df[['INCOME', 'KIDS']]
Y = df[['WSAVED']]

log_reg = linear_model.LogisticRegression(C=1)

log_reg.fit(X, Y.values.ravel())



