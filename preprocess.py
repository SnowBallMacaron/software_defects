import pandas as pd
import numpy as np
import seaborn as sns
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt

# RFE method model = LinearRegression()
def rfeFeatureSelect(model, data, n_features_to_select):

    rfe = RFE(
        estimator=model,
        n_features_to_select=n_features_to_select
    )
    feature_names = data.drop(columns=['defects']).columns
    rfe = rfe.fit(data.drop(columns=['defects']), data['defects'])
    support = rfe.get_support()
    selected_feature_rfe = [x for x, y in zip(feature_names, support) if y == True] + ['defects']
    data = data[selected_feature_rfe]
    return data, selected_feature_rfe


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
def seqFeatureSelect(model, data, direction, scoring, n_features_to_select):
    sfs = SequentialFeatureSelector(model,
                                    n_features_to_select=n_features_to_select,
                                    direction=direction,
                                    scoring=scoring)
    # fit the object to the training data
    feature_names = data.drop(columns=['defects']).columns
    sfs = sfs.fit(data.drop(columns=['defects']), data['defects'])
    support = sfs.get_support()
    selected_feature = [x for x, y in zip(feature_names, support) if y == True] + ['defects']
    data = data[selected_feature]
    return data, selected_feature


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

def outlier(data, factor):
    no_outliers = data.copy()
    metrics = data.describe()
    for i in data.columns:
        if i != "defects":
            Q1 = metrics[i]["25%"]
            Q3 = metrics[i]["75%"]
            IQR = Q3 - Q1
            upper = Q3 + IQR * factor
            lower = Q1 - IQR * factor
            no_outliers.loc[no_outliers[i] > upper, i] = np.NaN
            no_outliers.loc[no_outliers[i] < lower, i] = np.NaN

    no_outliers.dropna(inplace=True)
    no_outliers.reset_index(drop=True, inplace=True)
    return no_outliers


if __name__ == '__main__' :
    raw_data = pd.read_csv('./train.csv')
    # X = raw_data.iloc[:,:-1]
    # y = raw_data['defects'].astype(int)
    direction = 'backward'
    scoring = 'roc_auc'
    model = Lasso()  # LinearRegression Lasso

    data, selected_features = seqFeatureSelect(model, raw_data, direction, scoring)
    print(selected_features)

    data = outlier(data)

    data = create_folds(data, n_splits=5, seed=1)
    data['defects'] = data['defects'].astype(int)
    print(data.groupby(['kfold', 'defects']).size())


    # plt.plot(figsize=(10, 3), dpi=200)
    # sns.countplot(data, y="defects", hue="defects")
    # plt.show()
def sampling(data, strategy, method): ## under, over
    if method == 'under':
        sample = RandomUnderSampler(sampling_strategy=strategy)
    else:
        sample = RandomOverSampler(sampling_strategy=strategy)
    X, y = sample.fit_resample(data.drop(columns=['defects']), data['defects'])
    data = pd.concat([X, y], axis=1).reset_index(drop=True)
    return data

