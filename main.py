from dataLoadess import Imgdataset
from torch.utils.data import DataLoader
from models import E2ECNN, reconnet
from my_utils import generate_masks, time2file_name
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from torch.autograd import Variable
import pytorch_ssim
import ssim  # 新增：用于 MS-SSIM 计算
import csv  # NEW: for metric logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

data_path = "/media/ubuntu/DATA/sjq/SCI/DAVIS2017"
# data_path = "/media/ubuntu/DATA/sjq/SCI/ADMM-net-motion/train"
test_path1 = "/media/ubuntu/DATA/sjq/SCI/single_view_motion_flow_v6/test"  # simulation data for comparison

mask, mask_s = generate_masks(data_path)

last_train = 0
model_save_filename = 'save_model'
max_iter = 20
batch_size = 3
learning_rate = 0.0001

block_size = 256
compress_rate = 8

dataset = Imgdataset(data_path)

train_data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


net = E2ECNN(compress_rate).cuda()
# net = torch.load(
#         './model/' + model_save_filename + "/recon_net_model_epoch_20.pth")
# net = torch.load(
#         './model/' + model_save_filename + "/net_model_epoch_39.pth")

if last_train != 0:
    net = torch.load(
        './model/' + model_save_filename + "/recon_net_model_epoch_{}.pth".format(last_train))
    # recon_net = torch.load(
    #     './model/' + model_save_filename + "/recon_net_model_epoch_{}.pth".format(last_train))

loss = nn.MSELoss()
loss.cuda()
# 新增：启用 MS-SSIM 指标，保持与原始损失设计一致
ms_ssim_metric = ssim.MS_SSIM(data_range=1., channel=compress_rate)
ms_ssim_metric.cuda()


# NEW: 统一的指标记录函数，确保每轮以追加方式写入 CSV
def log_metrics(log_path, exp_name, epoch, phase, loss_value, psnr, ssim_val, lr, time_sec):
    """将单条指标写入 CSV，若文件不存在则自动写入表头。"""
    header = ['exp_name', 'epoch', 'phase', 'loss', 'psnr', 'ssim', 'learning_rate', 'time_sec']
    file_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            exp_name,
            int(epoch),
            phase,
            float(loss_value) if loss_value != '' else '',
            float(psnr) if psnr != '' else '',
            float(ssim_val) if ssim_val != '' else '',
            float(lr) if lr != '' else '',
            float(time_sec) if time_sec != '' else ''
        ])


def validate(test_path, epoch, result_path, psnr_epoch, ssim_epoch):
    # 新增：验证阶段仅做前向推理与指标统计
    net.eval()
    begin = time.time()
    test_list = os.listdir(test_path)
    psnr_sample = torch.zeros(len(test_list))
    ssim_sample = torch.zeros(len(test_list))
    loss_accum = 0.0  # NEW: accumulate validation loss
    frame_counter = 0  # NEW: count frames for average loss
    for i in range(len(test_list)):
        pic = scio.loadmat(test_path + '/' + test_list[i])

        pic = pic['orig']
        pic = pic / 255

        pic_gt = np.zeros([pic.shape[2] // compress_rate, compress_rate, block_size, block_size])
        for jj in range(pic.shape[2]):
            if jj % compress_rate == 0:
                meas_t = np.zeros([block_size, block_size])
                n = 0
            pic_t = pic[:, :, jj]
            mask_t = mask[n, :, :]

            mask_t = mask_t.cpu()
            pic_gt[jj // compress_rate, n, :, :] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t.numpy(), pic_t)

            if jj == compress_rate - 1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % compress_rate == 0 and jj != compress_rate - 1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        meas = torch.from_numpy(meas)
        pic_gt = torch.from_numpy(pic_gt)
        meas = meas.cuda()
        pic_gt = pic_gt.cuda()
        meas = meas.float()
        pic_gt = pic_gt.float()

        meas_re = torch.div(meas, mask_s)
        meas_re = torch.unsqueeze(meas_re, 1)

        out_save1 = torch.zeros([meas.shape[0], compress_rate, block_size, block_size]).cuda()
        with torch.no_grad():
            psnr_1 = 0
            ssim_1 = 0
            for ii in range(meas.shape[0]):
                begin=time.time()
                out_pic = net(mask, torch.unsqueeze(meas_re[ii, :, :, :], dim=0), block_size, compress_rate)
                end = time.time()
                print("time is {:.6f}".format(end-begin))

                out_save1[ii, :, :, :] = out_pic[0, :, :, :]

                for jj in range(compress_rate):
                    out_pic_forward = out_pic[0, jj, :, :]
                    gt_t = pic_gt[ii, jj, :, :]
                    mse_forward = loss(out_pic_forward * 255, gt_t * 255)
                    mse_forward = mse_forward.data
                    loss_accum += mse_forward.item()  # NEW: accumulate loss
                    frame_counter += 1  # NEW: count frames
                    psnr_1 += 10 * torch.log10(255 * 255 / mse_forward)
                    ssim_1 += pytorch_ssim.ssim(out_pic_forward, gt_t)

            psnr_1 = psnr_1 / (meas.shape[0] * compress_rate)
            ssim_1 = ssim_1 / (meas.shape[0] * compress_rate)
            psnr_sample[i] = psnr_1
            ssim_sample[i] = ssim_1
            if epoch % 1 == 0:
                a = test_list[i]
                name1 = result_path + '/result_' + a[0:len(a) - 4] + '{}_{:.4f}_{:.7f}'.format(epoch, psnr_1,ssim_1) + '.mat'
                scio.savemat(name1, {'pic': out_save1.cpu().numpy()})
    print("PSNR result: {:.4f}".format(torch.mean(psnr_sample)),
          "     SSIM result: {:.6f}".format(torch.mean(ssim_sample)))

    val_loss = loss_accum / max(frame_counter, 1)
    val_psnr = torch.mean(psnr_sample)
    val_ssim = torch.mean(ssim_sample)
    elapsed = time.time() - begin

    psnr_epoch.append(psnr_sample)
    ssim_epoch.append(ssim_sample)

    return psnr_epoch, ssim_epoch, val_loss, val_psnr, val_ssim, elapsed


def train(epoch, learning_rate, optimizer):
    """执行单轮训练并返回平均损失与指标。"""
    net.train()  # 新增：显式切换到训练模式
    epoch_loss = 0.0
    begin = time.time()
    psnr_acc = 0.0
    ssim_acc = 0.0
    batch_counter = 0

    for iteration, batch in enumerate(train_data_loader):
        gt, meas = Variable(batch[0]), Variable(batch[1])
        gt = gt.cuda().float()  # [batch,8,256,256]
        meas = meas.cuda().float()  # [batch,256 256]

        meas_re = torch.div(meas, mask_s)
        meas_re = torch.unsqueeze(meas_re, 1)

        optimizer.zero_grad()  # 新增：训练阶段执行反向传播前清梯度

        output = net(mask, meas_re, block_size, compress_rate)

        Loss1 = torch.sqrt(loss(output, gt))
        Loss2 = ms_ssim_metric(output * 255, gt * 255, data_range=255, size_average=False).mean()

        Loss = Loss1 + 0.1 * (1 - Loss2)

        epoch_loss += Loss.item()

        # 计算训练阶段的 PSNR/SSIM 便于日志记录
        mse_batch = torch.mean((output - gt) ** 2)
        psnr_batch = 10 * torch.log10(255 * 255 / mse_batch)
        ssim_batch = pytorch_ssim.ssim(output, gt)
        psnr_acc += psnr_batch.item()
        ssim_acc += ssim_batch.item()
        batch_counter += 1

        if iteration % 1000 == 0:
            print("======>Iteration {} at epoch {}, the loss is {:.8f}, ssim 2 is {:.4f}"
                  .format(iteration, epoch, Loss.item(), Loss2.item()))

        Loss.backward()  # 新增：反向传播
        optimizer.step()  # 新增：参数更新

    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))

    train_loss = epoch_loss / len(train_data_loader)
    train_psnr = psnr_acc / max(batch_counter, 1)
    train_ssim = ssim_acc / max(batch_counter, 1)
    train_time = end - begin

    return train_loss, train_psnr, train_ssim, train_time


def checkpoint(epoch, model_path):
    model_out_path = './' + model_path + '/' + "net_model_epoch_{}.pth".format(epoch)
    torch.save(net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def main(learning_rate):
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    psnr_epoch = []
    ssim_epoch = []
    psnr_max = 0
    exp_name = os.path.basename(result_path)  # 新增：用于日志标识当前实验
    log_file = os.path.join(result_path, 'training_log.csv')

    # 新增：优化器在训练循环外创建，防止每轮重新初始化
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

    for epoch in range(last_train + 1, max_iter + 1):
        print("epoch ", epoch)

        # 新增：保持学习率计划的同时不重建优化器
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # 1. 训练一轮
        train_loss, train_psnr, train_ssim, train_time = train(epoch, learning_rate, optimizer)

        # 2. 验证一轮
        psnr_epoch, ssim_epoch, val_loss, val_psnr, val_ssim, val_time = validate(test_path1, epoch, result_path, psnr_epoch, ssim_epoch)
        psnr_mean = torch.mean(psnr_epoch[-1])

        # 新增：分别记录 train / val 的指标到 CSV
        log_metrics(log_file, exp_name, epoch, 'train', train_loss, train_psnr, train_ssim, learning_rate, train_time)
        log_metrics(log_file, exp_name, epoch, 'val', val_loss, val_psnr.item(), val_ssim.item(), learning_rate, val_time)

        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 29.5:
                checkpoint(epoch, model_path)
        if (epoch % 5 == 0 or epoch > 70):
            checkpoint(epoch, model_path)
        if (epoch % 10 == 0) and (epoch < 150):
            learning_rate = learning_rate * 0.9
            print("epoch {}, learning rate: {:.6f}".format(epoch, learning_rate))


if __name__ == '__main__':
    main(learning_rate)
