import pandas as pd

from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

# RFE method model = LinearRegression()
def rfeFeatureSelect(model, X, y):

    rfe = RFE(
        estimator=model
        # n_features_to_select = 19
    )
    feature_names = X.drop(columns='kfold').columns
    rfe = rfe.fit(X.drop(columns='kfold'), y)
    support = rfe.get_support()
    selected_feature_rfe = [x for x, y in zip(feature_names, support) if y == True] + ['kfold']
    X = X[selected_feature_rfe]
    return X, selected_feature_rfe


## Select from model method
# model = RandomForestRegressor()
# # select from the model
# sfm = SelectFromModel(estimator=model)
# X_transformed = sfm.fit_transform(X, Y)
# # see which features were selected
# support = sfm.get_support()
# # get feature names
# selected_feature = [
#  x for x, y in zip(col_names, support) if y == True
# ]
# print(selected_feature, len(selected_feature))
# print()
# print(sfm.threshold_)

#wrapper method
def seqFeatureSelect(model, X, y, direction, scoring):
    sfs = SequentialFeatureSelector(model,
                                    # n_features_to_select=19,
                                    direction=direction,
                                    scoring=scoring)
    # fit the object to the training data
    feature_names = X.drop(columns='kfold').columns
    sfs = sfs.fit(X.drop(columns='kfold'), y)
    support = sfs.get_support()
    selected_feature = [x for x, y in zip(feature_names, support) if y == True] + ['kfold']
    X = X[selected_feature]
    return X, selected_feature


def create_folds(data, n_splits, seed=None):
    data["kfold"] = -1
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for f, (t_, v_) in enumerate(kf.split(X=data, y=data['defects'])):
        data.loc[v_, 'kfold'] = f

    return data

# df = pd.DataFrame({'team': ['A', 'A', 'A', 'B', 'B', 'B'],
#                    'points': [11, 7, 8, 10, 13, 13],
#                    'assists': [5, 7, 7, 9, 12, 9],
#                    'rebounds': [11, 8, 10, 6, 6, 5],
#                    'defects': [1, 1, 1, 1, 1, 1]})
# print(create_folds(df, 3, 4))

if __name__ == '__main__' :
    raw_data = pd.read_csv('./train.csv')
    # X = raw_data.iloc[:,:-1]
    # y = raw_data['defects'].astype(int)
    direction = 'backward'
    scoring = 'roc_auc'
    model = Lasso()

    data = create_folds(raw_data, n_splits=5, seed=1)
    X = data[data.columns[~data.columns.isin(['defects'])]]
    y = data['defects'].astype(int)
    print(data.groupby(['kfold','defects']).size())

    X, selected_features = seqFeatureSelect(model, X, y, direction, scoring)
    print(selected_features)
