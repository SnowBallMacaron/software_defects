from preprocess import rfeFeatureSelect, create_folds, seqFeatureSelect, outlier, sampling, saveSamples, process_data
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn import metrics
import xgboost as xgb
# import cupy as cp
import time
import lightgbm as lgb
import optuna
import logging
import sys

def modelCV(data, model, k=5):
    for fold in range(k):
        train_data = data[data.kfold != fold].reset_index(drop=True).drop(columns='kfold')
        val_data = data[data.kfold == fold].reset_index(drop=True).drop(columns='kfold')

        x_train = train_data.drop(columns='defects').values
        y_train = train_data.defects.values
        x_val = val_data.drop(columns='defects').values
        y_val = val_data.defects.values

        model.fit(x_train, y_train)
        val_preds = model.predict_proba(x_val)[:, 1]
        val_preds_bin = val_preds>0.5

        auc = metrics.roc_auc_score(y_val, val_preds)
        cf = metrics.confusion_matrix(y_val, val_preds_bin)
        print(f"Fold = {fold}, AUC = {auc}")
        print(cf)


def trainModel(params, model, file_path):
    data = pd.read_csv(file_path)


def objectiveXGB(trial):
    params = {
        # "eval_metric": "auc",
        'n_estimators': trial.suggest_int('n_estimators', 300, 900, 10),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0),
        "max_depth" : trial.suggest_int('max_depth', 3, 8),
        "scale_pos_weight" : trial.suggest_float('scale_pos_weight', 1, 2)
    }
    # dtrain = xgb.DMatrix(data.drop(columns=['defects']).values, label=data.defects.values)
    # # pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
    # cv_results = xgb.cv(
    #     params,
    #     dtrain,
    #     num_boost_round=1000,
    #     nfold=5,
    #     stratified=True,
    #     early_stopping_rounds=10,
    #     # callbacks=[pruning_callback],
    #     seed=7
    # )
    # mean_auc = cv_results['test-auc-mean']
    mean_auc = 0
    model = xgb.XGBClassifier(**params)
    for fold in range(5):
        train_data = data[data.kfold != fold].reset_index(drop=True).drop(columns='kfold')
        val_data = data[data.kfold == fold].reset_index(drop=True).drop(columns='kfold')

        x_train = train_data.drop(columns='defects').values
        y_train = train_data.defects.values
        x_val = val_data.drop(columns='defects').values
        y_val = val_data.defects.values
        model = model.fit(x_train, y_train)
        val_preds = model.predict_proba(x_val)[:, 1]

        auc = metrics.roc_auc_score(y_val, val_preds)
        mean_auc += auc
    return mean_auc / 5

def objectiveLGBM(trial):
    params = {
        "boosting_type": "gbdt",
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 512),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100,),
        'max_depth':-1,
        'learning_rate':0.01,
        'n_estimators':trial.suggest_int('n_estimators', 100, 500, 10),
        'objective':'binary',
        'n_jobs':-1,
        'verbose':-1,
    }
    # dtrain = xgb.DMatrix(data.drop(columns=['defects']).values, label=data.defects.values)
    # # pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
    # cv_results = xgb.cv(
    #     params,
    #     dtrain,
    #     num_boost_round=1000,
    #     nfold=5,
    #     stratified=True,
    #     early_stopping_rounds=10,
    #     # callbacks=[pruning_callback],
    #     seed=7
    # )
    # mean_auc = cv_results['test-auc-mean']
    mean_auc = 0
    model = lgb.LGBMClassifier(**params)
    for fold in range(5):
        train_data = data[data.kfold != fold].reset_index(drop=True).drop(columns='kfold')
        val_data = data[data.kfold == fold].reset_index(drop=True).drop(columns='kfold')

        x_train = train_data.drop(columns='defects').values
        y_train = train_data.defects.values
        x_val = val_data.drop(columns='defects').values
        y_val = val_data.defects.values
        model = model.fit(x_train, y_train)
        val_preds = model.predict_proba(x_val)[:, 1]

        auc = metrics.roc_auc_score(y_val, val_preds)
        mean_auc += auc
    return mean_auc / 5

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
        # scale_pos_weight=2
        # device=device
    )
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        n_jobs=-1,
        max_depth=9,
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
    # begin = time.time()
    # for i in range(4):
    #     sample = pd.read_csv(f'sample_{i}.csv')
    #     print(f'----------sample{i}------------')
    #     data = create_folds(sample, n_splits=5, seed=7)
    #     data.loc[:, 'defects'] = data['defects'].astype(int)
    #     # print(data.groupby(['kfold', 'defects']).size())
    #     modelCV(data, train_model)
    # print(time.time() - begin)

    for i in range(4):
        sample = pd.read_csv(f'sample_{i}.csv')
        print(f'----------sample{i}------------')
        data = create_folds(sample, n_splits=5, seed=7)
        data.loc[:, 'defects'] = data['defects'].astype(int)
        # print(data.groupby(['kfold', 'defects']).size())
        sampler = optuna.samplers.TPESampler()

        study = optuna.create_study(sampler=sampler, direction='maximize')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objectiveXGB, n_trials=100)
        # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        # study.optimize(objective, n_trials=100, timeout=600)
        print(f"Best value: {study.best_value}, Best params: {study.best_params}")

    for i in range(4):
        sample = pd.read_csv(f'sample_{i}.csv')
        print(f'----------sample{i}------------')
        data = create_folds(sample, n_splits=5, seed=7)
        data.loc[:, 'defects'] = data['defects'].astype(int)
        # print(data.groupby(['kfold', 'defects']).size())
        sampler = optuna.samplers.TPESampler()

        study = optuna.create_study(sampler=sampler, direction='maximize')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objectiveLGBM, n_trials=100)
        # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        # study.optimize(objective, n_trials=100, timeout=600)
        print(f"Best value: {study.best_value}, Best params: {study.best_params}")