from preprocess import rfeFeatureSelect, create_folds, seqFeatureSelect, outlier, sampling, saveSamples, process_data
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn import metrics
import xgboost as xgb
import cupy as cp
import time
import lightgbm as lgb

def modelCV(data, model, k=5):
    for fold in range(k):
        train_data = data[data.kfold != fold].reset_index(drop=True).drop(columns='kfold')
        val_data = data[data.kfold == fold].reset_index(drop=True).drop(columns='kfold')

        x_train = train_data.drop(columns='defects').values
        y_train = train_data.defects.values
        x_val = val_data.drop(columns='defects').values
        y_val = val_data.defects.values

        model.fit(x_train, y_train)
        val_preds = model.predict_proba(x_val,verbose=-1)[:, 1]

        auc = metrics.roc_auc_score(y_val, val_preds)

        print(f"Fold = {fold}, AUC = {auc}")

# def modelCV(data, model, k=5):
#     data = cp.asarray(data.values)
#     for fold in range(k):
#         train_data = data[data[:, -1] != fold][:, :-1]
#         val_data = data[data[:, -1] == fold][:, :-1]
#
#         x_train = train_data[:, :-1]
#         y_train = train_data[:, -1]
#         x_val = val_data[:, :-1]
#         y_val = val_data[:, -1]
#
#         model.fit(x_train, y_train)
#         val_preds = model.predict_proba(x_val,verbose=-1)[:, 1]
#
#         auc = metrics.roc_auc_score(y_val, val_preds)
#
#         print(f"Fold = {fold}, AUC = {auc}")

# def train_model(data, ):
#     direction = 'backward'
#     scoring = 'roc_auc'
#     xgb_model = xgb.XGBClassifier(
#         n_estimators=300,
#         n_jobs=-1,
#         max_depth=3
#     )
#     train_model = xgb_model

if __name__ == "__main__":
    # raw_data = pd.read_csv('./train.csv')
    data_path = './train.csv'
    # X = raw_data.iloc[:,:-1]
    # y = raw_data['defects'].astype(int)
    direction = 'backward'
    scoring = 'roc_auc'
    device = 'cpu'
    # device = 'cuda'

    lgb_model = lgb.LGBMClassifier(
        boosting_type='dart', #gbdt, dart
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.01,
        n_estimators=200,
        objective='binary',
        n_jobs=-1,
        verbose=-1,
        # device=device
    )
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        n_jobs=-1,
        max_depth=7,
        device=device
    )
    lasso_model = Lasso()
    linear_model = LinearRegression()

    train_model = lgb_model
    feature_model = lasso_model
    n_features_to_select = 16

    sample_params = {'sample1': ('majority', 'under'),
                     # 'sample2': (1/3, 'under'),
                     'sample3': (0.5, 'under'),
                     'sample4': ('minority', 'over'),
                     # 'sample5': (1/3, 'over'),
                     'sample6': (0.5, 'over'),
                     }


    # factor = 10

    # print('selecting features')
    # data, selected_features = seqFeatureSelect(feature_model, raw_data, direction, scoring, n_features_to_select)
    # #data, selected_features = rfeFeatureSelect(feature_model, raw_data, 19)
    # print(selected_features)
    #
    # # print('removing outliers')
    # # data = outlier(data, factor=factor)
    #
    # print('sampling')
    # # print(data.groupby(['defects']).size())
    # samples = sampling(data, sample_params)
    # saveSamples(samples)

    # process_data(data_path,sample_params, feature_model, direction, scoring, n_features_to_select)
    begin = time.time()
    for i in range(7):
        sample = pd.read_csv(f'sample_{i}.csv')
        print(f'----------sample{i}------------')
        data = create_folds(sample, n_splits=5, seed=7)
        data.loc[:, 'defects'] = data['defects'].astype(int)
        print(data.groupby(['kfold', 'defects']).size())
        modelCV(data, train_model)
    print(time.time() - begin)
