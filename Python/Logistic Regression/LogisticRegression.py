import time

import pandas as pd
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn import model_selection
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    print("Running...")

    cat_vars = ['AGECL',
                'HHSEX',
                'EDCL',
                'HOUSECL',
                'KIDS',
                'MARRIED',
                'OCCAT1',
                'OCCAT2',
                'RACE',
                'LF',
                'INDCAT',
                'EXPENSHILO',
                'TURNDOWN',
                'BNKRUPLAST5',
                'FORECLLAST5',
                'ASSETCAT',
                'HBUS',
                'LEASE',
                'NVEHIC',
                'WSAVED',
                'INCCAT',
                'NWCAT']

    y_name = 'NOCHK'#'HLIQ','NOCCBAL','HSTOCKS','DCPLANCJ','HBROK','HTRAD','LATE'
    #really good Ys: 'BSHOPGRDL','ISHOPGRDL','BINTERNET','IINTERNET','INTERNET','YESFINRISK'

    all_vars = list(cat_vars)
    all_vars.append(y_name)

    df = pd.read_csv('data/berkeleySCF/sub-data.txt', low_memory=False)

    df = df.loc[df['YEAR'] == 2016][all_vars]

    for var in cat_vars:
        cat_list = pd.get_dummies(df[var], prefix=var)
        data1 = df.join(cat_list)
        df = data1

    data_vars = df.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in cat_vars]

    df_final = df[to_keep]

    Y = [y_name]
    X = [i for i in df_final.columns.values.tolist() if i not in Y]

    log_reg = linear_model.LogisticRegression()

    rfe = RFE(log_reg,4)
    rfe = rfe.fit(df_final[X], df_final[Y].values.ravel())

    print(rfe.support_)
    print(rfe.ranking_)

    print(X)

    X = {i for i in X if X.index(i) in rfe.get_support(indices=True)}

    print(X)

    data_vars = df.columns.values.tolist()
    to_keep = [i for i in data_vars if i in X]

    X_data = df_final[to_keep]
    Y_data = df_final[Y]

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_data, Y_data, test_size=.75,
                                                                        random_state=int(time.time()))

    log_reg.fit(X_train, Y_train.values.ravel())

    Y_pred = log_reg.predict(X_test)

    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_reg.score(X_test, Y_test)))

    confusion_matrix = confusion_matrix(Y_test, Y_pred)
    print(confusion_matrix)

    prob = log_reg.predict_proba(X_test)

    print(log_reg.coef_)

    print(X_train.std())

    print(X_train.std().values * log_reg.coef_)
