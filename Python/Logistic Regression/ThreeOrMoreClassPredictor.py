import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn import linear_model

CONST_X1 = 'INCOME'
CONST_X2 = 'NETWORTH'
CONST_Y = 'AGECL'

df = pd.read_csv('data/berkeleySCF/sub-data.txt', low_memory=False)

df = df[df[CONST_Y].apply(lambda x: isinstance(x, int) or x.isdigit())]
df = df.loc[(df[CONST_X1] > 0) & (df[CONST_X2] > 0)][[CONST_X1, CONST_X2, CONST_Y]]

X = df[[CONST_X1, CONST_X2]].values.astype(np.float64)
Y = np.ceil(df[[CONST_Y]].values.astype(np.int64) / 2).astype(np.int64)

X = np.log10(X)

log_reg = linear_model.LogisticRegression(C=1e5)

log_reg.fit(X, Y.ravel())

h_X = .05
h_Y = .05

x_min, x_max = X[:, 0].min(), X[:, 0].max() + 1
y_min, y_max = X[:, 1].min(), X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h_X), np.arange(y_min, y_max, h_Y))
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])

c_map = plt.cm.get_cmap('Paired')

Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=c_map)

Y_cats = np.unique(Y)

norm = colors.Normalize(vmin=Y_cats.min(), vmax=Y_cats.max())

for i in Y_cats:
    plt.scatter(x_min - 1, y_min - 1, c=c_map(norm(i)), edgecolors='k', label=str(i))

plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], edgecolors='k', cmap=c_map)

plt.xlabel(CONST_X1)
plt.ylabel(CONST_X2)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.legend(loc=3)
plt.show()
