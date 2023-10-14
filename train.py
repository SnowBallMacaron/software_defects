from preprocess import rfeFeatureSelect, create_folds, seqFeatureSelect
import pandas as pd
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


if __name__ == "__main__":
    raw_data = pd.read_csv('./train.csv')
    # X = raw_data.iloc[:,:-1]
    # y = raw_data['defects'].astype(int)
    direction = 'backward'
    scoring = 'roc_auc'
    feature_model = Lasso()
    train_model = xgb.XGBClassifier(
        n_estimators=200,
        n_jobs=-1,
        max_depth=3
    )

    data = create_folds(raw_data, n_splits=5, seed=1)
    data['defects'] = data['defects'].astype(int)
    print(data.groupby(['kfold','defects']).size())

    data, selected_features = seqFeatureSelect(feature_model, data, direction, scoring)
    print(selected_features)

    modelSelect(data, train_model)