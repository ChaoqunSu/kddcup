import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SeriesDecomp(nn.Layer):
    """
    Ideas comes from AutoFormer
    Decompose a time series into trends and seasonal
    Refs:  https://arxiv.org/abs/2106.13008
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        t_x = paddle.transpose(x, [0, 2, 1])
        mean_x = F.avg_pool1d(
            t_x, self.kernel_size, stride=1, padding="SAME", exclusive=False)
        mean_x = paddle.transpose(mean_x, [0, 2, 1])
        return x - mean_x, mean_x


class WPFModel(nn.Layer):
    """Models for Wind Power Prediction
    """

    def __init__(self, settings):
        super(WPFModel, self).__init__()
        self.input_len = settings["input_len"]
        self.output_len = settings["output_len"]
        self.var_len = settings["var_len"]
        self.hidden_dims = settings["hidden_dims"]

        DECOMP = 18
        self.decomp = SeriesDecomp(DECOMP)

        a = float(1 / self.input_len)
        # season
        self.Linear_Seasonal = nn.Linear(self.input_len, self.output_len)
        x1 = a * paddle.ones([self.input_len, self.output_len], dtype='float32')
        self.Linear_Seasonal.weight = paddle.create_parameter(x1.shape, dtype='float32',
                                                              default_initializer=nn.initializer.Assign(x1))
        # trend
        self.Linear_Trend = nn.Linear(self.input_len, self.output_len)
        y1 = a * paddle.ones([self.input_len, self.output_len], dtype='float32')
        self.Linear_Trend.weight = paddle.create_parameter(y1.shape, dtype='float32',
                                                           default_initializer=nn.initializer.Assign(y1))

        self.Linear_Decoder = nn.Linear(self.hidden_dims, self.var_len)

    def forward(self, batch_x):
        """
        :param batch_x: [N, L, C]  C=134(turbines)
        :return: prediction output
        """
        seasonal_init, trend_init = self.decomp(batch_x)
        seasonal_init = paddle.transpose(seasonal_init, [0, 2, 1])
        trend_init = paddle.transpose(trend_init, [0, 2, 1])

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        # N, hidden_dims, output_len
        pred_y = seasonal_output + trend_output
        return pred_y[:, :, -self.output_len:]
