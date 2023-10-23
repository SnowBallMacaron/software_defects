import numpy as np
from functools import partial
from scipy.optimize import fmin
from sklearn import metrics
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from preprocess import create_folds

class OptimizeAUC:
    """
    Class for optimizing AUC.
    This class is all you need to find best weights for
    any model and for any metric and for any types of predictions.
    With very small changes, this class can be used for optimization of
    weights in ensemble models of _any_ type of predictions
    """

    def __init__(self):
        self.coef = 0

    def _auc(self, coef, X, y):
        """
        This functions calulates and returns AUC.
        :param coef: coef list, of the same length as number of models
        :param X: predictions, in this case a 2d array
        :param y: targets, in our case binary 1d array
        """

        # multiply coefficients with every column of the array
        # with predictions.
        # this means: element 1 of coef is multiplied by column 1
        # of the prediction array, element 2 of coef is multiplied
        # by column 2 of the prediction array and so on!
        x_coef = X * coef
        # create predictions by taking row wise sum
        predictions = np.sum(x_coef, axis=1)

        # calculate auc score
        auc_score = metrics.roc_auc_score(y, predictions)
        # return negative auc
        return -1.0 * auc_score

    def fit(self, X, y):

        # remember partial from hyperparameter optimization chapter?
        loss_partial = partial(self._auc, X=X, y=y)

        # dirichlet distribution. you can use any distribution you want
        # to initialize the coefficients
        # we want the coefficients to sum to 1
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)
        # use scipy fmin to minimize the loss function, in our case auc
        self.coef = fmin(loss_partial, initial_coef, disp=False)

    def predict(self, X):
        # this is similar to _auc function
        x_coef = X * self.coef
        predictions = np.sum(x_coef, axis=1)
        return predictions


def ensembleCV(data, models, k=5):
    for fold in range(k):
        train_data = data[data.kfold != fold].reset_index(drop=True).drop(columns='kfold')
        val_data = data[data.kfold == fold].reset_index(drop=True).drop(columns='kfold')

        x_train = train_data.drop(columns='defects').values
        y_train = train_data.defects.values
        x_val = val_data.drop(columns='defects').values
        y_val = val_data.defects.values
        val_preds = []
        train_preds = []
        for modelCls, params in models:

            model = modelCls(**params)
            model.fit(x_train, y_train)
            val_preds.append(model.predict_proba(x_val)[:, 1])

            model = modelCls(**params)
            model.fit(x_val, y_val)
            train_preds.append(model.predict_proba(x_train)[:, 1])

        val_preds = np.stack(val_preds, axis=1)
        train_preds = np.stack(train_preds, axis=1)

        opt = OptimizeAUC()
        opt.fit(train_preds, y_train)
        opt_preds = opt.predict(val_preds)
        auc = metrics.roc_auc_score(y_val, opt_preds)
        print(f"Optimized AUC, Fold val = {auc}")
        print(f"Coefficients = {opt.coef}")

        # opt = OptimizeAUC()
        # opt.fit(val_preds, y_val)
        # opt_preds = opt.predict(train_preds)
        # auc = metrics.roc_auc_score(y_train, opt_preds)
        # print(f"Optimized AUC, Fold train = {auc}")
        # print(f"Coefficients = {opt.coef}")

def votePredict(train_data_path, test_path, models, coef, selected_feature):
    train_data_raw = [pd.read_csv(path) for path in train_data_path]
    train_data = [(raw.drop(columns='defects'), raw.defects) for raw in train_data_raw]
    X_test = pd.read_csv(test_path)[selected_feature]
    X_test_id = X_test.pop('id')


    preds = []
    for (X_train, y_train), (modelCls, param) in zip(train_data, models):
        model = modelCls(**param)
        model.fit(X_train.values, y_train.values)
        pred = model.predict_proba(X_test.values)[:, 1]
        preds.append(pred)

    preds = np.stack(preds, axis=1)
    preds_coef = preds * coef
    predictions = np.sum(preds_coef, axis=1)
    submission = pd.DataFrame({'id': X_test_id, 'defects': predictions})
    return submission

if __name__ == '__main__':
    param1 = {"boosting_type":'dart', #gbdt, dart
                "num_leaves":46,
                "max_depth":5,
                "learning_rate":0.0526,
                "n_estimators":750,
                "objective":'binary',
                "n_jobs":-1,
                "verbose":-1,}

    param2 = {
        "n_estimators" : 350,
        "n_jobs" : -1,
        "max_depth" : 10
    }

    param3 = {
        "n_estimators": 500,
        "n_jobs": -1,
        "max_depth": 9
    }


    models = [
        [lgb.LGBMClassifier, param1],
        [xgb.XGBClassifier, param2],
        [xgb.XGBClassifier, param3]
    ]

    samples = [
        'sample_0.csv',
        'sample_1.csv',
        'sample_2.csv'
    ]

    # for i in range(4):
    #     sample = pd.read_csv(f'sample_{i}.csv')
    #     print(f'----------sample{i}------------')
    #     data = create_folds(sample, n_splits=5, seed=7)
    #     data.loc[:, 'defects'] = data['defects'].astype(int)
    #     # print(data.groupby(['kfold', 'defects']).size())
    #     ensembleCV(data, models, k=5)


    votePredict(samples, 'test.csv', models, [1,1,1], ['id', 'loc', 'ev(g)', 'iv(g)', 'l', 'd', 'i', 'b', 'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount'])
