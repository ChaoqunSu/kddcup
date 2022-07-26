import os
import numpy as np
import paddle
import paddle.nn.functional as F
from DLinear import WPFModel
from dataset import WPFDataset, TestDataset
from prepare import prep_env


def forecast(settings):
    size = [settings["input_len"], settings["output_len"], settings["label_len"]]
    train_dataset1 = WPFDataset(
        data_path=settings["data_path"],
        filename=settings["train_filename"],
        flag='train',
        size=[settings["input_len"], settings["label_len"], settings["output_len"]],
        train_days=settings["train_size"],
        val_days=settings["val_size"]
    )

    scaler1 = train_dataset1.scaler
    model1 = WPFModel(settings)
    path_to_model1 = os.path.join(settings["checkpoints"], "model")
    model1.set_state_dict(paddle.load(path_to_model1))
    model1.eval()
    test_x_path = settings["path_to_test_x"]
    test_x_ds = TestDataset(filename=test_x_path, size=size)
    # get_data return:1,L,134-->1,288,134
    test_x = paddle.to_tensor(
        test_x_ds.get_data()[:, -settings["input_len"]:, :], dtype="float32")
    # Normalization
    scaled_test_x1 = scaler1.transform(test_x)
    # 1, 134,288
    pred_y_1 = model1(scaled_test_x1)
    # 1, 288, 134
    pred_y_1 = paddle.transpose(pred_y_1, [0, 2, 1])
    # 1, 288, 134
    inverse_pred_y_1 = scaler1.inverse_transform(pred_y_1)
    inverse_pred_y_1 = F.relu(inverse_pred_y_1)
    # 1, 288, 134 --> 134, 288, 1
    inverse_pred_y_1 = paddle.transpose(inverse_pred_y_1, [2, 1, 0])

    return np.array(inverse_pred_y_1)


# if __name__ == "__main__":
#     settings = prep_env()
#     # 134, 288, 1
#     pred_y = forecast(settings)
#     print(pred_y)
