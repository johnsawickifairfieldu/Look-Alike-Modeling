import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import Imputer

CONST_Y1 = 'Annual Dropout Rate \'08'
CONST_Y2 = 'Graduation Rate \'08'

df = pd.read_csv('data/CT_school_counties_data.csv', low_memory=False, delimiter='|')

data_vars = df.columns.values.tolist()
X_cols = [i for i in data_vars if i not in [CONST_Y1, CONST_Y2, 'District']]

for y in [CONST_Y1, CONST_Y2]:

    df1 = df[~df[y].isnull()]

    Y1 = df1[y]
    X1 = df1[X_cols]

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X1 = imp.fit_transform(X1)

    lin_reg1 = linear_model.LinearRegression()
    lin_reg1.fit(X1, Y1)

    np.set_printoptions(suppress=True)

    # The correlation coefficients
    print("Correlation coefficients for " + y + ":\n", lin_reg1.coef_ * np.std(X1, axis=0) / np.std(Y1))
    # Returns the coefficient of determination R^2 of the prediction.
    print("The coefficient of determination R^2 of the prediction of " + y + ": %.2f" % lin_reg1.score(X1, Y1))
