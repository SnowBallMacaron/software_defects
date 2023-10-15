import copy
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
            self.net.append(nn.BatchNorm1d(out_feature))
            self.net.append(nn.ReLU())
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

def trainNNModel(model, train_dl, val_dl, fold):
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

    n_epochs = 50  # number of epochs to run

    # Hold the best model
    # best_acc = - np.inf  # init to negative infinity
    # best_weights = None

    for epoch in tqdm(range(n_epochs)):
        model.train()
        loss_all = 0
        for i, (X, y) in enumerate(train_dl):
            optimizer.zero_grad()
            # forward pass
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(1))
            loss_all += loss.item()
            # backward pass

            loss.backward()
            # update weights
            optimizer.step()
            # print progress
        if epoch % 5 == 0:
            print(f"epoch {epoch}, Loss: {loss_all/len(train_dl)}")
        scheduler.step(loss_all/len(train_dl))

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

    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == '__main__':
    device = torch.device('cuda')
    samples = 5
    k=5


    for i in range(samples):
        sample = pd.read_csv(f'sample_{i}.csv')
        print(f'----------sample{i}------------')
        data = create_folds(sample, n_splits=5, seed=7)
        data.loc[:, 'defects'] = data['defects'].astype(int)
        print(data.groupby(['kfold', 'defects']).size())

        for fold in range(k):
            model = NNModel([19, 64, 128, 256, 512], 0.1).to(device)
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
                                  drop_last=False)
            val_dl = DataLoader(dataset=TrainDataset(x_val, y_val),
                                  batch_size=256,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False)

            trainNNModel(model, train_dl, val_dl, fold)
