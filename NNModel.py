import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from preprocess import create_folds
from sklearn import metrics

class NNModel(nn.Module):
    def __init__(self, feature_lists, dropout):
        super().__init__()
        self.net = nn.Sequential()
        for in_feature, out_feature in zip(feature_lists[:-1], feature_lists[1:]):
            self.net.append(nn.Linear(in_feature, out_feature))
            # self.net.append(nn.BatchNorm1d(out_feature))
            self.net.append(nn.ReLU())
            # self.net.append(nn.Sigmoid())
            # self.net.append(nn.Softmax(1))
            self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(feature_lists[-1], 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        x = self.sigmoid(x)
        return x

class TrainDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x.astype('float32'), y.astype('float32')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class ModelCrossValidator:
    def __init__(self, model_features, n_samples, k, seed=7, file_name='nn_sample'):
        self.file_name = file_name
        self.model_features = model_features
        self.n_samples = n_samples
        self.k = k
        self.all_model_performance = []
        total_epochs = 0
        for features_list in self.model_features:
            total_epochs += features_list['n_epochs']
        self.pbar = tqdm(total=self.n_samples*k*total_epochs)
        self.seed = seed
        self.samples_data = [create_folds(pd.read_csv(f'{self.file_name}_{i}.csv'), n_splits=self.k, seed=self.seed)
                             for i in range(self.n_samples)]

    def validateModel(self):
        for model_num, feature_list in enumerate(self.model_features):
            performance = []
            for i in range(self.n_samples):
                performance.append([])
                data = self.samples_data[i]
                # print(f'----------sample{i}------------')
                # data = create_folds(sample, n_splits=self.k, seed=self.seed)
                # data.loc[:, 'defects'] = data['defects'].astype(int)
                # print(data.groupby(['kfold', 'defects']).size())

                for fold in range(self.k):
                    model = NNModel(feature_list['layer_params'], feature_list['dropout']).to(device)
                    train_data = data[data.kfold != fold].reset_index(drop=True).drop(columns='kfold')
                    val_data = data[data.kfold == fold].reset_index(drop=True).drop(columns='kfold')

                    x_train = train_data.drop(columns='defects').values
                    y_train = train_data.defects.values
                    x_val = val_data.drop(columns='defects').values
                    y_val = val_data.defects.values

                    train_dl = DataLoader(dataset=TrainDataset(x_train, y_train),
                                          batch_size=256,
                                          shuffle=True,
                                          pin_memory=True,
                                          drop_last=False,
                                          num_workers=0)
                    val_dl = DataLoader(dataset=TrainDataset(x_val, y_val),
                                        batch_size=256,
                                        shuffle=False,
                                        pin_memory=True,
                                        drop_last=False)

                    performance[-1].append(trainNNModel(model, train_dl, val_dl, feature_list['n_epochs'], self.pbar))
            self.all_model_performance.append(performance)

    def reportCV(self):
        for i, (sample_performance_list, feature_list) in enumerate(zip(self.all_model_performance, self.model_features)):
            print(f'\n==============model{i + 1}==================')
            mean_auc = 0
            num_val = 0
            print(f'''n_features = {feature_list['n_features']}
layer_params = {feature_list['layer_params']}
n_epochs = {feature_list['n_epochs']}
dropout = {feature_list['dropout']}''')

            for j, sample in enumerate(sample_performance_list):
                print(f'----------sample{j}------------')
                for fold, auc in enumerate(sample):
                    mean_auc += auc
                    num_val += 1
                    print(f"Fold = {fold}, AUC = {auc}")
            print(f'\nMean AUC = {mean_auc/num_val}')


def trainNNModel(model, train_dl, val_dl, n_epochs, pbar):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=1 / 3,
                                                           patience=10,
                                                           verbose=0,
                                                           cooldown=2,
                                                           min_lr=1e-5)
  # number of epochs to run

    # Hold the best model
    # best_acc = - np.inf  # init to negative infinity
    # best_weights = None

    for epoch in range(n_epochs):
        model.train()
        loss_all = 0
        for i, (X, y) in enumerate(train_dl):

            # forward pass
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(1))
            loss_all += loss.item()

            optimizer.zero_grad()
            # backward pass

            loss.backward()
            # update weights
            optimizer.step()
            # print progress
        # if epoch % 5 == 0:
        #     print(f"epoch {epoch}, Loss: {loss_all/len(train_dl)}")
        scheduler.step(loss_all/len(train_dl))
        pbar.update(1)

    model.eval()
    predictions = []
    with torch.no_grad():
        for i, (X, _) in enumerate(val_dl):
            X = X.to(device)
            prediction = model.forward(X)
            predictions.append(prediction.detach().cpu().numpy())
    predictions = np.concatenate(predictions)
    predictions = np.squeeze(predictions, axis=1)
    auc = metrics.roc_auc_score(val_dl.dataset.y, predictions)

    return auc

if __name__ == '__main__':
    device = torch.device('cuda')
    n_epochs = 300
    n_features = 15
    modelCVer = ModelCrossValidator(
        n_samples = 1,
        k=5,
        model_features=[
            # {'n_features': n_features,
            #  'layer_params': [n_features, 64, 64],
            #  'n_epochs': n_epochs,
            #  'dropout': 0.01},
            # {'n_features': n_features,
            #  'layer_params': [n_features, 64, 64],
            #  'n_epochs': n_epochs,
            #  'dropout': 0.3},
            # {'n_features': n_features,
            #  'layer_params': [n_features, 16, 16, 16, 16, 16, 16, 16],
            #  'n_epochs': n_epochs,
            #  'dropout': 0.2},
            # {'n_features': n_features,
            #  'layer_params': [n_features, 16, 16, 16, 16, 16, 16, 16],
            #  'n_epochs': n_epochs,
            #  'dropout': 0.2},
            {'n_features': n_features,
             'layer_params': [n_features, 32, 32, 32],
             'n_epochs': n_epochs,
             'dropout': 0.2},
        ]
    )

    modelCVer.validateModel()
    modelCVer.reportCV()
