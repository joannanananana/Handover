import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.nn.utils import clip_grad_norm_
import os
import tqdm
import numpy as np
import argparse
import time
import copy
from dataset import GraspDataset
from model import BackbonePointNet2
import yaml
from utils import scene_util

with open("config/handover_config.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser("Train Handover Grasp")
parser.add_argument("--batchsize", type=int, default=cfg["network"]["train"]["batchsize"], help="input batch size")
parser.add_argument("--workers", type=int, default=cfg["network"]["train"]["workers"],
                    help="number of data loading workers")
parser.add_argument("--epoch", type=int, default=cfg["network"]["train"]["epochs"],
                    help="number of epochs for training")
parser.add_argument("--gpu", type=str, default=cfg["network"]["train"]["gpu"], help="specify gpu device")
parser.add_argument("--learning_rate", type=float, default=cfg["network"]["train"]["learning_rate"],
                    help="learning rate for training")
parser.add_argument("--optimizer", type=str, default=cfg["network"]["train"]["optimizer"], help="type of optimizer")
# parser.add_argument("--theme", type=str, default=cfg["network"]["train"]["theme"], help="type of train")
# parser.add_argument("--model_path", type=str, default=cfg["network"]["eval"]["model_path"], help="type of train")
FLAGS = parser.parse_args()

theme = "quat_loss" + str(FLAGS.batchsize)
time_now = str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
LOG_DIR = os.path.join(BASE_DIR, "experiment", time_now + "_" + theme)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FOUT_test = open(os.path.join(LOG_DIR, "log_test.txt"), "w")
LOG_FOUT_test.write(str(FLAGS) + "\n")
LOG_FOUT_train = open(os.path.join(LOG_DIR, "log_train.txt"), "w")
LOG_FOUT_train.write(str(FLAGS) + "\n")
save_model_dir = os.path.join(LOG_DIR, "checkpoints")
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)


def log_string_test(out_str):
    LOG_FOUT_test.write(out_str + "\n")
    LOG_FOUT_test.flush()
    print(out_str)


def log_string_train(out_str):
    LOG_FOUT_train.write(out_str + "\n")
    LOG_FOUT_train.flush()
    print(out_str)


def adjust_learning_rate(optimizer, epoch):
    """ Sets the learning rate to the initial LR decayed by 2 every 10 epochs """
    lr = max(FLAGS.learning_rate / (2 ** (epoch // 10)), FLAGS.learning_rate)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(optimizer, num_epochs, train_dataloader, test_dataloader, model, noise=False):
    model = model.train()
    for epoch in range(num_epochs):
        loss_list, acc_list = [], []
        for i, (data, index) in enumerate(train_dataloader):
            data_nan = torch.any(torch.tensor([torch.any(torch.isnan(data[k])).item() for k in data])).item()
            assert not data_nan, "Nan values found in dataset"
            optimizer.zero_grad()
            for k in data:
                data[k] = data[k].cuda().float()
            if noise:
                shape = data["point"].cpu().numpy().shape
                noise = []
                axis_min_values = np.min(data["point"].cpu().numpy(), axis=1)
                axis_max_values = np.max(data["point"].cpu().numpy(), axis=1)
                sigma = (axis_max_values - axis_min_values) / 20
                for bat in range(sigma.shape[0]):
                    x_sigma, y_sigma, z_sigma = sigma[bat]
                    x_noise = np.random.normal(0, x_sigma, shape[1])
                    y_noise = np.random.normal(0, y_sigma, shape[1])
                    z_noise = np.random.normal(0, z_sigma, shape[1])
                    bat_noise = np.dstack((x_noise, y_noise, z_noise)).squeeze()
                    noise.append(bat_noise)
                noise_point = torch.Tensor(noise)
                ori_point = data["point"].cpu() + noise_point
                center = torch.mean(ori_point, dim=1)
                norm_point = []
                for bat in range(ori_point.shape[0]):
                    bat_ori_point = ori_point[bat]
                    bat_center = center[bat]
                    bat_norm_point = bat_ori_point - bat_center
                    norm_point.append(bat_norm_point)
                norm_point = torch.stack(norm_point)
                ori_point = ori_point.cuda().float()
                norm_point = norm_point.cuda().float()
                pred = model(ori_point, norm_point.transpose(1, 2))
            else:
                pred = model(data["point"], data["norm_point"].transpose(1, 2))
            loss_dict, acc_dict = model.module.get_loss(pred, data)
            loss = loss_dict["total_loss"]
            loss.backward()
            # clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
            loss_list.append([v.item() for k in loss_dict if k != "total_loss" for v in loss_dict[k].values()])
            acc_list.append([v for k in acc_dict for v in acc_dict[k].values()])
            each_tax_mean_loss = np.mean(loss_list, 0)
            mean_loss = np.mean(each_tax_mean_loss.reshape(-1, 5), 0)
            each_tax_mean_acc = np.mean(acc_list, 0)
            mean_acc = np.mean(each_tax_mean_acc.reshape(-1, 10), 0)
            print("epoch:%i\t iter:%.0f/%.0f\t gpts loss:%.3f\t depth loss:%.3f\t"
                  "quat loss:%.3f\t joint loss:%.3f\t total loss:%.3f\t TP:%.0d\t FP:%.0d\t"
                  "TN:%.0d\t FN:%.0d\t acc:%.3f\t p:%.3f\t"
                  "r:%.3f\t F1:%.3f\t depth acc:%.3f\t quat acc:%.3f"
                  % (epoch, i, len(train_dataloader), mean_loss[0], mean_loss[1],
                     mean_loss[2], mean_loss[3], mean_loss[4], mean_acc[0], mean_acc[1],
                     mean_acc[2], mean_acc[3], mean_acc[4], mean_acc[5],
                     mean_acc[6], mean_acc[7], mean_acc[8], mean_acc[9]
                     ))
        log_string_train("epoch:%i\t iter:%.0f/%.0f\t gpts loss:%.3f\t depth loss:%.3f\t"
                         "quat loss:%.3f\t joint loss:%.3f\t total loss:%.3f\t TP:%.0d\t FP:%.0d\t"
                         "TN:%.0d\t FN:%.0d\t acc:%.3f\t p:%.3f\t"
                         "r:%.3f\t F1:%.3f\t depth acc:%.3f\t quat acc:%.3f"
                         % (epoch, i, len(train_dataloader), mean_loss[0], mean_loss[1],
                            mean_loss[2], mean_loss[3], mean_loss[4], mean_acc[0], mean_acc[1],
                            mean_acc[2], mean_acc[3], mean_acc[4], mean_acc[5],
                            mean_acc[6], mean_acc[7], mean_acc[8], mean_acc[9]
                            ))
        torch.save(model.state_dict(), "%s/%s_%.3d.pth" % (save_model_dir, "model", epoch))
        test(test_dataloader, model)


def test(test_loader, model, noise=False):
    model = model.eval()
    with torch.no_grad():
        loss_list, acc_list = [], []
        for i, (data, index) in enumerate(test_loader):
            for k in data:
                data[k] = data[k].cuda().float()
            if noise:
                shape = data["point"].cpu().numpy().shape
                noise = []
                axis_min_values = np.min(data["point"].cpu().numpy(), axis=1)
                axis_max_values = np.max(data["point"].cpu().numpy(), axis=1)
                sigma = (axis_max_values - axis_min_values) / 20
                for bat in range(sigma.shape[0]):
                    x_sigma, y_sigma, z_sigma = sigma[bat]
                    x_noise = np.random.normal(0, x_sigma, shape[1])
                    y_noise = np.random.normal(0, y_sigma, shape[1])
                    z_noise = np.random.normal(0, z_sigma, shape[1])
                    bat_noise = np.dstack((x_noise, y_noise, z_noise)).squeeze()
                    noise.append(bat_noise)
                noise_point = torch.Tensor(noise)
                ori_point = data["point"].cpu() + noise_point
                center = torch.mean(ori_point, dim=1)
                norm_point = []
                for bat in range(ori_point.shape[0]):
                    bat_ori_point = ori_point[bat]
                    bat_center = center[bat]
                    bat_norm_point = bat_ori_point - bat_center
                    norm_point.append(bat_norm_point)
                norm_point = torch.stack(norm_point)
                ori_point = ori_point.cuda().float()
                norm_point = norm_point.cuda().float()
                pred = model(ori_point, norm_point.transpose(1, 2))
            else:
                pred = model(data["point"], data["norm_point"].transpose(1, 2))
            loss_dict, acc_dict = model.module.get_loss(pred, data)
            loss = loss_dict["total_loss"]
            loss_list.append([v.item() for k in loss_dict if k != "total_loss" for v in loss_dict[k].values()])
            acc_list.append([v for k in acc_dict for v in acc_dict[k].values()])
            each_tax_mean_loss = np.mean(loss_list, 0)
            mean_loss = np.mean(each_tax_mean_loss.reshape(-1, 5), 0)
            each_tax_mean_acc = np.mean(acc_list, 0)
            mean_acc = np.mean(each_tax_mean_acc.reshape(-1, 10), 0)
            log_string_test("test:\t gpts loss:%.3f\t depth loss:%.3f\t"
                             "quat loss:%.3f\t joint loss:%.3f\t total loss:%.3f\t TP:%.0d\t FP:%.0d\t"
                             "TN:%.0d\t FN:%.0d\t acc:%.3f\t p:%.3f\t"
                             "r:%.3f\t F1:%.3f\t depth acc:%.3f\t quat acc:%.3f"
                             % (mean_loss[0], mean_loss[1],
                                mean_loss[2], mean_loss[3], mean_loss[4], mean_acc[0], mean_acc[1],
                                mean_acc[2], mean_acc[3], mean_acc[4], mean_acc[5],
                                mean_acc[6], mean_acc[7], mean_acc[8], mean_acc[9]
                                ))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    print("Using GPUs " + os.environ["CUDA_VISIBLE_DEVICES"])
    train_data = GraspDataset(split="train")
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=FLAGS.batchsize,
                                                   shuffle=True,
                                                   num_workers=FLAGS.workers)
    test_data = GraspDataset(split="val")
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=FLAGS.batchsize,
                                                  shuffle=True,
                                                  num_workers=FLAGS.workers)

    model = BackbonePointNet2().cuda()
    model = torch.nn.DataParallel(model)

    optim_params = [
        {
            "params": model.parameters(),
            "lr": FLAGS.learning_rate,
            "betas": (0.9, 0.999),
            "eps": 1e-08
        }
    ]

    optimizer = optim.Adam(optim_params)
    train(optimizer, 40, train_dataloader, test_dataloader, model, noise=True)
