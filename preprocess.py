import pandas as pd

from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

raw_data = pd.read_csv('./train.csv')
X = raw_data.iloc[:,:-1]
y = raw_data['defects'].astype(int)
col_names = raw_data.columns

## RFE method
# rfe = RFE(
#     estimator=model,
#     n_features_to_select=19
# )
# rfe.fit(X, Y)
#
# X_transformed = rfe.transform(X)
# pass

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
model = Lasso()
sfs_f = SequentialFeatureSelector(model,
#                                n_features_to_select=19,
                                direction='forward',
                                scoring='roc_auc')
# fit the object to the training data
sfs_f = sfs_f.fit(X, y)
support_f = sfs_f.get_support()
selected_feature_f = [x for x, y in zip(col_names, support_f) if y == True]
print(selected_feature_f)

sfs_b = SequentialFeatureSelector(model,
#                                n_features_to_select=19,
                                direction='backward',
                                scoring='roc_auc')
# fit the object to the training data
sfs_b = sfs_b.fit(X, y)
support_b = sfs_b.get_support()
selected_feature_b = [x for x, y in zip(col_names, support_b) if y == True]
print(selected_feature_b)