import os
import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset


class Scaler(object):
    """
    Desc: Normalization utilities
    """
    def __init__(self):
        self.mean = np.array([0.])
        self.std = np.array([1.])

    def fit(self, data):
        """
        Desc:
            Fit the data
        Args:
            data:
        Returns:
            None
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data):
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        mean = paddle.to_tensor(self.mean) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std) if paddle.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """
        mean = paddle.to_tensor(self.mean) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std) if paddle.is_tensor(data) else self.std
        return (data * std) + mean


class WPFDataset(Dataset):
    """
    把134个turbine的patv拿出来拼成新df,len*134
    """

    first_initialized = False

    def __init__(
            self,
            data_path,
            filename='wtbdata_245days.csv',
            flag='train',
            size=None,
            scale=True,
            capacity=134,
            day_len=24 * 6,
            train_days=214,
            val_days=31,
            total_days=245,
            is_test=False):

        super().__init__()
        self.unit_size = day_len
        if size is None:
            self.input_len = self.unit_size
            self.label_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.label_len = size[1]
            self.output_len = size[2]
        self.capacity = capacity
        self.scale = scale

        assert flag in ['train', 'val']
        type_map = {'train': 0, 'val': 1}
        self.set_type = type_map[flag]
        self.flag = flag
        self.data_path = data_path
        self.filename = filename

        self.total_size = self.unit_size * total_days
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        self.is_test = is_test

        if self.is_test:
            if not WPFDataset.first_initialized:
                self.__read_data__()
                WPFDataset.first_initialized = True
        else:
            self.__read_data__()
        if not self.is_test:
            self.data_x, self.data_y = self.__get_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))
        # df_data, raw_df_data都是后十个feature组成的df,df_data做了缺失值填充
        df_data, raw_df_data = self.data_preprocess(df_raw)
        self.df_data = df_data
        self.raw_df_data = raw_df_data

    def __get_patv__(self):
        pd.set_option('mode.chained_assignment', None)
        border1s = [0, self.train_size - self.input_len]
        border2s = [self.train_size, self.train_size + self.val_size]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        patv_series = self.df_data['Patv']
        patv_arr = patv_series.values
        col_ls = []
        for i in range(134):
            str_i = str(i)
            col_ls.append(str_i)
        df_patv = pd.DataFrame(patv_arr.reshape((35280, self.capacity), order='F'), columns=col_ls,
                               index=None)
        print('the shape of df_patv:{}'.format(df_patv.shape))
        # 算归一化方差和均值只能用训练集, train_size * feature_len，train_data是用来算归一化参数的
        train_data = df_patv[border1s[0]:border2s[0]]

        # 依照不同flag选出训练集和验证集,data_x需要做归一化
        data_x = df_patv[border1:border2]

        # 计算metrics时用来对照的raw_df_data,valDataset才调用
        raw_df_data = self.raw_df_data
        cols_data = raw_df_data.columns
        raw_data = raw_df_data.values
        raw_data = np.reshape(
            raw_data, [self.capacity, self.total_size, len(cols_data)])
        self.raw_df = []
        for turb_id in range(self.capacity):
            self.raw_df.append(
                pd.DataFrame(
                    # 计算metrics，用到每个turbine最原始的df，val/test时调用，拿到的是对应val_dataset的
                    data=raw_data[turb_id, border1 + self.input_len:border2, :],
                    columns=cols_data))
        # 算loss时的input_y来源
        input_df_data = self.df_data
        cols_data = input_df_data.columns
        input_data = input_df_data.values
        input_data = np.reshape(
            input_data, [self.capacity, self.total_size, len(cols_data)])
        self.input_y = input_data[:, border1:border2, :]
        # 留出data_y，不作预处理和归一化，用来计算filter_loss
        data_y = df_patv[border1:border2]
        # 拟合归一化
        scaler = Scaler()
        scaler.fit(train_data.values)
        self.scaler = scaler
        if self.scale:
            data_x = self.scaler.transform(data_x.values)
            data_y = self.scaler.transform(data_y.values)
        else:
            data_x = data_x.values
            data_y = data_y.values
        return data_x, data_y

    def __get_data__(self):
        data_x, data_y = self.__get_patv__()
        return data_x, data_y

    def get_raw_df(self):
        assert self.flag == 'val'
        return self.raw_df

    def get_scaler(self):
        return self.scaler

    def get_col_names(self):
        feature_name = [
            n for n in self.df_data.columns
            if 'Day' not in n and 'Tmstamp' not in n and
               'TurbID' not in n
        ]
        col_names = dict(
            [(v, k) for k, v in enumerate(feature_name)])
        return col_names

    def __getitem__(self, index):
        # Sliding window with the size of input_len + output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.output_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # input_y是134,L, C(10features)
        input_y = self.input_y[:, r_begin:r_end, :]
        # batch_x,batch_y皆为L,C(134turb)
        return seq_x, seq_y, input_y

    def __len__(self):
        return len(self.data_x) - self.input_len - self.output_len + 1

    def data_preprocess(self, df_data):
        feature_name = [
            n for n in df_data.columns
            if 'Day' not in n and 'Tmstamp' not in n and
               'TurbID' not in n
        ]
        # ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv','Patv']
        # new_df_data就是后十个属性组成的新df
        new_df_data = df_data[feature_name]
        pd.set_option('mode.chained_assignment', None)
        raw_df_data = new_df_data
        new_df_data = new_df_data.replace(
            to_replace=np.nan, value=0.0, inplace=False)
        # 负值取0.0
        new_df_data.loc[(0.0 > new_df_data['Patv']), 'Patv'] = 0.0
        # new_df_data是12个列并且处理缺失值的新df,raw_df_data是不处理缺失值的新df
        return new_df_data, raw_df_data


class TestDataset(Dataset):
    def __init__(
            self,
            filename,
            size=None,
            capacity=134,
            day_len=24 * 6,
            is_test=False):
        super().__init__()
        self.unit_size = day_len
        if size is None:
            self.input_len = self.unit_size
            self.label_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.label_len = size[1]
            self.output_len = size[2]
        self.capacity = capacity
        self.filename = filename
        self.is_test = is_test
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.filename)
        # df_data, raw_df_data都是后十个feature组成的df,df_data做了缺失值填充
        df_data, raw_df_data = self.data_preprocess(df_raw)
        self.df_data = df_data
        self.raw_df_data = raw_df_data
        data_x = self.__get_patv__()
        self.data_x = data_x

    def __get_patv__(self):
        pd.set_option('mode.chained_assignment', None)
        data_len = len(self.df_data)
        patv_len = int(data_len/self.capacity)
        patv_series = self.df_data['Patv']
        patv_arr = patv_series.values
        col_ls = []
        for i in range(134):
            str_i = str(i)
            col_ls.append(str_i)

        df_patv = pd.DataFrame(patv_arr.reshape((patv_len, self.capacity), order='F'), columns=col_ls,
                               index=None)
        print('the shape of df_patv:{}'.format(df_patv.shape))
        # 依照不同flag选出训练集和验证集,data_x需要做归一化
        data_x = df_patv.values
        # L,134-->1,L,134
        data_x = np.expand_dims(data_x, 0)
        return data_x

    def data_preprocess(self, df_data):
        feature_name = [
            n for n in df_data.columns
            if 'Day' not in n and 'Tmstamp' not in n and
               'TurbID' not in n
        ]
        # ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv','Patv']
        # new_df_data就是后十个属性组成的新df
        new_df_data = df_data[feature_name]
        pd.set_option('mode.chained_assignment', None)
        raw_df_data = new_df_data
        new_df_data = new_df_data.replace(
            to_replace=np.nan, value=0.0, inplace=False)
        new_df_data.loc[(0.0 > new_df_data['Patv']), 'Patv'] = 0.0
        # new_df_data是12个列并且处理缺失值的新df,raw_df_data是不处理缺失值的新df
        return new_df_data, raw_df_data

    def get_data(self):
        return self.data_x
