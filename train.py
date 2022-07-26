import os
import time
import numpy as np
import paddle
import random
from paddle.io import DataLoader
import paddle.nn.functional as F
from DLinear import WPFModel
from dataset import WPFDataset
import logging
import loss as loss_factory
from metrics import regressor_detailed_scores
import optimization as optim
from common import EarlyStopping
from common import adjust_learning_rate
from utils import _create_if_not_exist
from prepare import prep_env


def data_augment(x, y, input, p=0.9, alpha=0.5, beta=0.5):
    """
    Regression SMOTE
    """
    # batch+x,batch_y: batch_size, L,C=134
    # input_y: batch_size, C=134, L, feature_len
    batch_size = x.shape[0]
    random_values = paddle.rand([batch_size])
    idx_to_change = random_values < p

    # ensure that first element to switch has probability > 0.5
    # beta(a,b,size)
    np_betas = np.random.beta(alpha, beta, batch_size) / 2 + 0.5
    random_betas_1 = paddle.to_tensor(
        np_betas, dtype="float32").reshape([-1, 1, 1])
    random_betas_2 = paddle.to_tensor(
        np_betas, dtype="float32").reshape([-1, 1, 1, 1])
    index_permute = paddle.randperm(batch_size)

    x[idx_to_change] = random_betas_1[idx_to_change] * x[idx_to_change]
    x[idx_to_change] += (
        1 - random_betas_1[idx_to_change]) * x[index_permute][idx_to_change]

    y[idx_to_change] = random_betas_1[idx_to_change] * y[idx_to_change]
    y[idx_to_change] += (
        1 - random_betas_1[idx_to_change]) * y[index_permute][idx_to_change]

    input[idx_to_change] = random_betas_2[idx_to_change] * input[idx_to_change]
    input[idx_to_change] += (
        1 - random_betas_2[idx_to_change]) * input[index_permute][idx_to_change]
    return x, y, input


settings = prep_env()


# logger
def getLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    work_dir = os.path.join(settings['logs_path'],
                            time.strftime("%Y-%m-%d-%H.%M", time.localtime()))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    return logger


logger = getLogger()


def train_and_val(settings, model, is_debug=False):
    """
    Desc:
        Training and validation
    Args:
        settings:env_settings
        model:model
        is_debug:False
    Returns:
        None
    """
    train_dataset = WPFDataset(
        data_path=settings["data_path"],
        filename=settings["train_filename"],
        flag='train',
        size=[settings["input_len"], settings["label_len"], settings["output_len"]],
        train_days=settings["train_size"],
        val_days=settings["val_size"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=settings["num_workers"],
        drop_last=True
    )
    val_dataset = WPFDataset(
        data_path=settings["data_path"],
        filename=settings["train_filename"],
        flag='val',
        size=[settings["input_len"], settings["label_len"], settings["output_len"]],
        train_days=settings["train_size"],
        val_days=settings["val_size"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        drop_last=False
    )

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    loss_fn = getattr(loss_factory, settings["name"])(
        **dict({"name": "FilterMSELoss"}))

    opt = optim.get_optimizer(model=model, learning_rate=settings["lr"])

    col_names = train_dataset.get_col_names()
    global_step = 0

    _create_if_not_exist(settings["checkpoints"])
    path_to_model = settings["checkpoints"]

    early_stopping = EarlyStopping(patience=settings["patient"])

    valid_records = []
    epoch_start_time = time.time()
    for epoch in range(settings["train_epochs"]):
        model.train()
        for i, (batch_x, batch_y, input_y) in enumerate(train_loader):
            # batch_x,batch_y: batch_size,input_len/label+output_len,C(134turb)
            # input_y: batch_size, 134,label+output_len, C(10features)
            batch_x = batch_x.astype('float32')
            batch_y = batch_y.astype('float32')
            input_y = input_y.astype('float32')
            batch_x, batch_y, input_y = data_augment(batch_x, batch_y, input_y)

            input_y = input_y[:, :, -settings["output_len"]:, :]
            batch_y = batch_y[:, -settings["output_len"]:, :]
            # pred_y: batch_size,134,288
            pred_y = model(batch_x)
            # batch_y: batch_size,288,134-->truth: N,134,288
            truth = paddle.transpose(batch_y, [0, 2, 1])
            loss = loss_fn(pred_y, truth, input_y, col_names)
            loss.backward()
            opt.step()
            opt.clear_gradients()
            global_step += 1
            if paddle.distributed.get_rank(
            ) == 0 and global_step % settings["log_per_steps"] == 0:
                logger.info("Step %s Train MSE-Loss: %s RMSE-Loss: %s" %
                         (global_step, loss.numpy()[0],
                          (paddle.sqrt(loss)).numpy()[0]))

        if is_debug:
            epoch_end_time = time.time()
            print("Elapsed time for epoch-{}: {}".format(epoch, epoch_end_time - epoch_start_time))
            epoch_start_time = epoch_end_time

        if paddle.distributed.get_rank() == 0:
            valid_r = validation(
                val_loader,
                val_dataset.get_raw_df(),
                train_dataset,
                col_names,
                model,
                loss_fn,
                settings)
            valid_records.append(valid_r)
            logger.info("Valid " + str(dict(valid_r)))
            val_loss = valid_r['turb_score']

            # Early Stopping if needed
            early_stopping(val_loss, model, path_to_model)
            logger.info("the best model's score is:{}".format(early_stopping.val_loss_min))
            if early_stopping.early_stop:
                print("Early stopped! ")
                break
        adjust_learning_rate(opt, epoch + 1, settings)


@paddle.no_grad()
def validation(valid_data_loader,
               raw_df,
               train_dataset,
               col_names,
               model,
               loss_fn,
               settings
               ):

    col_names = col_names
    model.eval()
    losses = []
    pred_batch = []
    gold_batch = []
    scaler = train_dataset.get_scaler()
    for batch_x, batch_y, input_y in valid_data_loader:
        # batch_x,batch_y: batch_size,input_len/label+output_len,C(134turb)
        # input_y: batch_size, 134,label+output_len, C(10features)
        batch_x = batch_x.astype('float32')
        batch_y = batch_y.astype('float32')

        # batch_size, 134,output_len
        pred_y = model(batch_x)
        # batch_size, label+output_len,C(134turb)-->N, output_len,C(134turb)
        batch_y = batch_y[:, -settings["output_len"]:, :]
        # batch_size, 134,output_len
        truth = paddle.transpose(batch_y, [0, 2, 1])
        # batch_size, 134, output_len, C(10features)
        input_y = input_y[:, :, -settings["output_len"]:, :].astype('float32')

        loss = loss_fn(pred_y, truth, input_y, col_names)
        losses.append(loss.numpy()[0])

        # batch_size, 134,output_len-->N, output_len, 134再进inverse_transform
        pred_y = paddle.transpose(pred_y, [0, 2, 1])
        inverse_pred_y = scaler.inverse_transform(pred_y)
        # batch_size, output_len-->N, 134, output_len
        inverse_pred_y = paddle.transpose(inverse_pred_y, [0, 2, 1])
        inverse_pred_y = F.relu(inverse_pred_y)

        # batch_size, 134,output_len-->N, output_len, 134
        truth = paddle.transpose(truth, [0, 2, 1])
        inverse_truth = scaler.inverse_transform(truth)
        # batch_size, output_len-->N, 134, output_len
        inverse_truth = paddle.transpose(inverse_truth, [0, 2, 1])
        pred_batch.append(inverse_pred_y.numpy())
        gold_batch.append(inverse_truth.numpy())

    # N', 134, output_len
    pred_batch = np.concatenate(pred_batch, axis=0)
    gold_batch = np.concatenate(gold_batch, axis=0)
    # N', 134, output_len, 1
    pred_batch = np.expand_dims(pred_batch, -1)
    gold_batch = np.expand_dims(gold_batch, -1)
    # 134, N', output_len, 1
    pred_batch = np.transpose(pred_batch, [1, 0, 2, 3])
    gold_batch = np.transpose(gold_batch, [1, 0, 2, 3])

    _mae, _rmse = regressor_detailed_scores(pred_batch, gold_batch,
                                            raw_df, settings)

    output_metric = {
        'farm_mae': _mae,
        'farm_rmse': _rmse,
        'turb_score': (_mae + _rmse) / 2,
        'loss': np.mean(losses),
    }

    return output_metric


if __name__ == "__main__":
    fix_seed = 3407
    random.seed(fix_seed)
    paddle.seed(fix_seed)
    np.random.seed(fix_seed)

    logger.info("The experimental settings are: \n{}".format(str(settings)))
    # Set up the initial environment
    start_train_time = time.time()
    logger.info('\n>>>>>>>Start training \n')
    model = WPFModel(settings)
    train_and_val(settings, model=model, is_debug=settings["is_debug"])
    if settings["is_debug"]:
        end_time = time.time()
        end_train_time = end_time
        print("\nTotal time in training {} turbines is "
              "{} secs".format(settings["capacity"], end_train_time - start_train_time))
