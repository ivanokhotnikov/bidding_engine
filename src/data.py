import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class BiddingData(Dataset):

    def __init__(self,
                 seq_len: int,
                 pred_len: int,
                 scale: bool,
                 flag: str,
                 val_split: float,
                 test_split: float,
                 cluster: int,
                 features: list,
                 target: str | None = None) -> None:
        assert flag in ['train', 'test', 'val', 'pred']
        assert target
        assert self.features
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[self.flag]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.val_split = val_split
        self.test_split = test_split
        self.target = target
        self.features = self.features
        self.scale = scale
        self.cluster = cluster
        self.__read_data__()

    def __len__(self) -> int:
        return len(self.x) - self.seq_len - self.pred_len + 1

    def __read_data__(self) -> pd.DataFrame:
        cluster_df = pd.read_csv(
            os.path.join(os.environ['PROCESSED_DATA_PATH'],
                         f'processed_{self.cluster}.csv'))
        size = len(cluster_df)
        border1s = [
            0,
            int(size * (1 - self.val_split - self.test_split)),
            int(size * (1 - self.test_split))
        ]
        border2s = [
            int(size * (1 - self.val_split - self.test_split)),
            int(size * (1 - self.test_split)), size
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.forecast_features = cluster_df.columns
        self.dataset = cluster_df.loc[border1:border2,
                                      self.features + [self.target]]
        self.data = self.dataset.values
        self.y = cluster_df.loc[border1:border2,
                                self.target].values.reshape(-1, 1)
        self.x = cluster_df.loc[border1:border2, self.self.features].values

        if self.scale:
            self.scaler = StandardScaler()
            train_data = cluster_df.loc[border1s[0]:border2s[0],
                                        self.features + [self.target]]
            self.scaler.fit(train_data)
            self.data = self.scaler.transform(self.dataset.values)
            self.x = pd.DataFrame(data=self.data,
                                  columns=self.features +
                                  [self.target])[self.features].values
            self.y = pd.DataFrame(data=self.data,
                                  columns=self.features +
                                  [self.target])[self.target].values

    def __getitem__(self, index: int):
        x_begin = index
        x_end = x_begin + self.seq_len
        y_begin = x_end
        y_end = y_begin + self.pred_len
        seq_x = self.x[x_begin:x_end]
        seq_y = self.y[y_begin:y_end]
        return seq_x, seq_y

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
