import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

raw_data = pd.read_csv('./train.csv')
X = raw_data.iloc[:,:-1]
Y = raw_data['defects'].astype(int)
model = LinearRegression()
rfe = RFE(
    estimator=model,
    n_features_to_select=19
)
rfe.fit(X, Y)

X_transformed = rfe.transform(X)
pass