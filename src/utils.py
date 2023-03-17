import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def visual(true, preds, name):
    plt.figure()
    plt.plot(true, label='test ground truth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='test prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


class EarlyStopping:

    def __init__(self, patience: int, verbose: bool = True, delta: int = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model, path='./runs'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.path = os.path.join(path)
            self.save_checkpoint(val_loss, model, self.path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'validation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f})',
                'saving model',
                sep='\n')
        os.makedirs(path, exist_ok=True)
        torch.save(model, os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


def create_sequences(df, lookback, target, inference, univariate):
    x_train, y_train = [], []
    if univariate:
        for i in range(lookback, len(df)):
            x_train.append(df.iloc[i - lookback:i][target])
            y_train.append(df.iloc[i][target])
        x_train = np.expand_dims(x_train, axis=-1)
    else:
        for i in range(lookback, len(df)):
            x_train.append(df.iloc[i - lookback:i])
            y_train.append(df.iloc[i][target])
        x_train = np.stack(x_train)
    y_train = np.expand_dims(y_train, axis=-1)
    if inference:
        return x_train
    return x_train, y_train
