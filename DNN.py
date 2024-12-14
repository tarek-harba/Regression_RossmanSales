import numpy as np
import sklearn
import sklearn.model_selection
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import os
import json


def preprocess(
    df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocesses the input DataFrame by:
    - Splitting the data into training and validation sets (95% for training and 5% for validation).
    - Normalizing numeric columns using the statistics (mean, standard deviation) from the training set.
    - One-hot encoding categorical columns.

    The function prepares the data for use in a Deep Neural Network (DNN) by returning feature tensors and target tensors.

    Parameters:
    -----------
    df : pd.DataFrame
        A pandas DataFrame containing the input data, with both numeric and categorical features,
        and a target column ('Sales').

    Returns:
    --------
    tuple
        A tuple containing four torch.Tensor objects:
        - x_train (torch.Tensor): The features for the training set.
        - y_train (torch.Tensor): The target values (Sales) for the training set.
        - x_valid (torch.Tensor): The features for the validation set.
        - y_valid (torch.Tensor): The target values (Sales) for the validation set.
    """
    numeric_col_names = ["CompetitionDistance"]
    df_train, df_valid = sklearn.model_selection.train_test_split(
        df, train_size=0.95, random_state=1
    )
    train_stats = df_train.describe().transpose()
    df_train_norm, df_valid_norm = df_train.copy(), df_valid.copy()
    for col_name in numeric_col_names:
        mean = train_stats.loc[col_name, "mean"]
        std = train_stats.loc[col_name, "std"]
        df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean) / std
        df_valid_norm.loc[:, col_name] = (df_valid_norm.loc[:, col_name] - mean) / std

    x_train = torch.tensor(df_train_norm[numeric_col_names].values)
    x_valid = torch.tensor(df_valid_norm[numeric_col_names].values)
    """we one-hot encode all categorical features, we remind that 'Store' column had many labels (1115) and so we 
    keep this in mind for later."""
    nominal_col_names = [
        "Assortment",
        "StoreType",
        "Promo",
        "MonthOfYear",
        "DayOfWeek",
        "Store_compressed",
    ]
    for col_name in nominal_col_names:
        unique, inverse = np.unique(df_train_norm[col_name].values, return_inverse=True)
        encoded_col = torch.from_numpy(np.eye(unique.shape[0])[inverse])
        x_train = torch.cat([x_train, encoded_col], 1).float()

        unique, inverse = np.unique(df_valid_norm[col_name].values, return_inverse=True)
        encoded_col = torch.from_numpy(np.eye(unique.shape[0])[inverse])
        x_valid = torch.cat([x_valid, encoded_col], 1).float()

    y_train = torch.tensor(df_train_norm["Sales"].values).float()
    y_valid = torch.tensor(df_valid_norm["Sales"].values).float()

    return x_train, y_train, x_valid, y_valid


class DNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        l1 = nn.Linear(49, 30)
        a1 = nn.ReLU()
        l2 = nn.Linear(30, 10)
        a2 = nn.ReLU()
        l3 = nn.Linear(10, 1)
        a3 = nn.ReLU()
        l = [l1, a1, l2, a2, l3, a3]
        self.module_list = nn.ModuleList(l)

        # Initialization:
        # kaiming normal best suited given ReLU activation function
        # Source: https://doi.org/10.48550/arXiv.1502.01852
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
                if m.bias is not None:
                    m.bias.detach().zero_()

    def forward(self, x):
        for f in self.module_list:
            x = f(x)
        return x


def rmspe_loss_fn(prediction, target):
    mask = target != 0
    prediction = prediction[mask]
    target = target[mask]
    loss = torch.sqrt(torch.mean(((target - prediction) / target) ** 2))
    return loss


def dnn_init(x_train, y_train, x_valid=None, y_valid=None):
    # DataLoader
    train_ds = TensorDataset(x_train, y_train)
    batch_size = 128
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    # Model, architecture, loss, optimizer
    model = DNN()
    MSE_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-3)
    num_epochs = 30
    MSE_loss_hist_train = [0] * num_epochs

    if x_valid is not None:  # if we want to train DNN on all data
        torch.manual_seed(1)
        valid_ds = TensorDataset(x_valid, y_valid)
        MSE_loss_hist_valid = [0] * num_epochs
        RMSPE_loss_hist_valid = [0] * num_epochs
        valid_dl = DataLoader(valid_ds, batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()  # Set to training mode
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:, 0]
            MSE_loss = MSE_loss_fn(pred, y_batch)
            MSE_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            MSE_loss_hist_train[epoch] += MSE_loss.item()
        MSE_loss_hist_train[epoch] /= len(train_dl)

        if x_valid is not None:  # the code below is used when tuning model parameters
            model.eval()  # Set to evaluation mode
            with torch.no_grad():
                for x_batch, y_batch in valid_dl:
                    pred = model(x_batch)[:, 0]
                    MSE_loss_hist_valid[epoch] += MSE_loss_fn(pred, y_batch).item()
                    RMSPE_loss_hist_valid[epoch] += rmspe_loss_fn(pred, y_batch).item()

                # Average validation losses over batches
            MSE_loss_hist_valid[epoch] /= len(valid_dl)
            RMSPE_loss_hist_valid[epoch] /= len(valid_dl)

            print(
                f"Epoch {epoch} - MSE Train Loss:{MSE_loss_hist_train[epoch]:.4f}"
                f"      MSE Valid Loss:{MSE_loss_hist_valid[epoch]:.4f}"
                f"      RMSPE Valid Loss:{RMSPE_loss_hist_valid[epoch]:.4f}"
            )

    if x_valid is None:
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # model_path = f'{dir_path}/DNN.pth'
        torch.save(model, "./DNN.pth")
        print("Model Saved!")

    # plt.plot([i for i in range(num_epochs)],MSE_loss_hist_train, lw=2)
    # plt.plot([i for i in range(num_epochs)],MSE_loss_hist_valid, lw=2)
    # plt.legend(['train_loss', 'valid_loss'], fontsize = 15)
    # plt.grid()
    # plt.show()
