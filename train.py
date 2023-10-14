from preprocess import rfeFeatureSelect, create_folds, seqFeatureSelect, outlier, sampling
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn import metrics
import xgboost as xgb

def modelSelect(data, model, k=5):
    for fold in range(k):
        train_data = data[data.kfold != fold].reset_index(drop=True).drop(columns='kfold')
        val_data = data[data.kfold == fold].reset_index(drop=True).drop(columns='kfold')

        x_train = train_data.drop(columns='defects').values
        x_val = val_data.drop(columns='defects').values

        model.fit(x_train, train_data.defects.values)
        val_preds = model.predict_proba(x_val)[:, 1]

        auc = metrics.roc_auc_score(val_data.defects.values, val_preds)

        print(f"Fold = {fold}, AUC = {auc}")

def train_model(data, ):
    direction = 'backward'
    scoring = 'roc_auc'
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        n_jobs=-1,
        max_depth=10
    )
    train_model = xgb_model








if __name__ == "__main__":
    raw_data = pd.read_csv('./train.csv')
    # X = raw_data.iloc[:,:-1]
    # y = raw_data['defects'].astype(int)
    direction = 'backward'
    scoring = 'roc_auc'

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        n_jobs=-1,
        max_depth=10
    )
    lasso_model = Lasso()
    linear_model = LinearRegression()

    train_model = xgb_model
    feature_model = lasso_model
    n_features_to_select = 13
    factor = 10

    print('selecting features')
    data, selected_features = seqFeatureSelect(feature_model, raw_data, direction, scoring, n_features_to_select)
    #data, selected_features = rfeFeatureSelect(feature_model, raw_data, 19)
    print(selected_features)

    # print('removing outliers')
    # data = outlier(data, factor=factor)

    print('sampling')
    sample_under1 = sampling(data, 'majority', 'under')
    sample_under2 = sampling(data, 1/3, 'under')
    sample_under3 = sampling(data, 0.5, 'under')
    sample_over1 = sampling(data, 'minority', 'over')
    sample_over2 = sampling(data, 1/3, 'over')
    sample_over3 = sampling(data, 0.5, 'over')


    samples = [data, sample_under1, sample_under2, sample_under3, sample_over1, sample_over2, sample_over3]
    for i, sample in enumerate(samples):
        print(f'----------sample{i+1}------------')
        data = create_folds(sample, n_splits=5, seed=1)
        data['defects'] = data['defects'].astype(int)
        # print(data.groupby(['kfold', 'defects']).size())
        modelSelect(data, train_model)
