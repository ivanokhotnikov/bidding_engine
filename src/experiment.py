import os
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import BiddingData
from model import LSTM
from utils import EarlyStopping, visual


class Experiment:

    def __init__(self, args):
        self.args = args
        torch.manual_seed(self.args['seed'])

    def _select_criterion(self):
        return nn.MSELoss()

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(),
                          lr=self.args['learning_rate'])

    def _get_data(self, flag):
        if flag in ['test', 'pred']:
            shuffle_flag = False
        else:
            shuffle_flag = True
        data = BiddingData(seq_len=self.args['seq_len'],
                           pred_len=self.args['pred_len'],
                           features=self.args['features'],
                           val_split=self.args['val_split'],
                           test_split=self.args['test_split'],
                           target=self.args['target'],
                           cluster=self.args['cluster'],
                           scale=True,
                           flag=flag)
        loader = DataLoader(dataset=data,
                            batch_size=self.args['batch_size'],
                            shuffle=shuffle_flag,
                            drop_last=True)
        return data, loader

    def _build_model(self):
        model = LSTM
        return model(input_size=self.train_data.x.shape[-1],
                     output_size=self.train_data.y.shape[-1],
                     hidden_size=self.args['hidden_size'],
                     num_layers=self.args['num_layers'],
                     batch_size=self.args['batch_size'])

    def train(self):
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.val_data, self.val_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')
        self.model = self._build_model()
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args['patience'],
                                       verbose=True)
        self.path = os.path.join(os.environ['RUNS_PATH'],
                                 self.args['timestamp'],
                                 f'cluster_{self.args["cluster"]}')
        os.makedirs(self.path, exist_ok=True)
        for epoch in range(self.args['epochs']):
            print('epoch: {}'.format(epoch + 1))
            train_losses = []
            self.model.train()
            epoch_time = datetime.now()
            self.writer = SummaryWriter(log_dir=os.path.join(
                self.path, 'tensorboards', f'epoc_{epoch+1}'))
            for i, (batch_x, batch_y) in enumerate(self.train_loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                model_optim.zero_grad()
                pred = self.model(batch_x)
                pred = pred[:, -self.args['pred_len']:, :]
                loss = torch.sqrt(criterion(pred, batch_y))
                self.writer.add_scalar('loss train/iter', loss, i)
                train_losses.append(loss.item())
                if i % self.args['log_interval'] == 0:
                    print('\titer: {0:>5d}/{1:>5d} | loss: {2:.3f}'.format(
                        i, train_steps, loss.item()))
                loss.backward()
                model_optim.step()
            train_loss = np.mean(train_losses)
            val_loss = self.evaluate(self.val_loader, criterion)
            test_loss = self.evaluate(self.test_loader, criterion)
            print('epoch {0} time: {1} s'.format(epoch + 1,
                                                 datetime.now() - epoch_time))
            print(
                'train loss: {0:.3f} | val loss: {1:.3f} | test loss: {2:.3f}'.
                format(train_loss, val_loss, test_loss))
            early_stopping(val_loss, self.model, self.path)
            self.writer.flush()
            if early_stopping.early_stop:
                print('early stopping!')
                break
            self.writer.close()
        self.best_model_path = os.path.join(self.path, 'checkpoint.pth')
        self.model = torch.load(self.best_model_path)
        return self.model

    def evaluate(self, loader, criterion):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                pred = self.model(batch_x)
                pred = pred[:, -self.args['pred_len']:, :]
                loss = torch.sqrt(criterion(pred, batch_y))
                self.writer.add_scalar(f'loss {loader.dataset.flag}/iter',
                                       loss, i)
                losses.append(loss.item())
        loss = np.mean(losses)
        self.model.train()
        return loss

    def test(self):
        self.test_data, self.test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        test_folder = os.path.join(self.path, 'test_results')
        os.makedirs(test_folder, exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.test_loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                pred = self.model(batch_x)
                pred = pred[:, -self.args['pred_len']:, -1]
                true = batch_y[:, -self.args['pred_len']:]
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                if i % self.args['log_interval'] == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :]), axis=0)
                    visual(gt, pd, os.path.join(test_folder, f'{i}.pdf'))

    def predict(self):
        pred_data, pred_loader = self._get_data(flag='test')
        self.model.load_state_dict(torch.load(self.best_model_path))
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                pred = self.model(batch_x)
                pred = pred[:, -self.args['pred_len']:, -1]
                preds.append(pred)
        preds = np.array(preds).flatten()
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
            # result save
        np.save(self.path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]),
                               preds[0],
                               axis=1),
                     columns=pred_data.cols).to_csv(self.path +
                                                    'prediction.csv',
                                                    index=False)
