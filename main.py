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
# ms_ssim = ssim.MS_SSIM(data_range=1., channel=compress_rate)
# ms_ssim.cuda()


def validate(test_path, epoch, result_path, psnr_epoch, ssim_epoch):
    test_list = os.listdir(test_path)
    psnr_sample = torch.zeros(len(test_list))
    ssim_sample = torch.zeros(len(test_list))
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
                # recon_net.eval()
                # out_pic = recon_net(out_1)
                end = time.time()
                print("time is {:.6f}".format(end-begin))

                out_save1[ii, :, :, :] = out_pic[0, :, :, :]

                for jj in range(compress_rate):
                    out_pic_forward = out_pic[0, jj, :, :]
                    gt_t = pic_gt[ii, jj, :, :]
                    mse_forward = loss(out_pic_forward * 255, gt_t * 255)
                    mse_forward = mse_forward.data
                    # print("dataset {}, batch {}, compress_rate {}, the loss is {}".format(
                    #     i, ii, jj, mse_forward))
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

    psnr_epoch.append(psnr_sample)
    ssim_epoch.append(ssim_sample)

    return psnr_epoch, ssim_epoch


def train(epoch, learning_rate, result_path, psnr_epoch, ssim_epoch):
    epoch_loss = 0
    begin = time.time()
    # psnr_epoch = []
    # ssim_epoch = []

    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

    if __name__ == '__main__':
        for iteration, batch in enumerate(train_data_loader):
            gt, meas = Variable(batch[0]), Variable(batch[1])
            gt = gt.cuda().float()  # [batch,8,256,256]
            meas = meas.cuda().float()  # [batch,256 256]

            meas_re = torch.div(meas, mask_s)
            meas_re = torch.unsqueeze(meas_re, 1)

            batch_size1 = gt.shape[0]

            output = net(mask, meas_re, block_size, compress_rate)

            optimizer.zero_grad()

            Loss1 = torch.sqrt(loss(output, gt))
            Loss2 = ms_ssim(output*255, gt*255, data_range=255, size_average=False).mean()

            Loss = Loss1 + 0.1 * (1 - Loss2)

            epoch_loss += Loss.data

            if iteration % 1000 == 0:
                print("======>Iteration {} at epoch {}, the loss is {:.8f}, ssim 2 is {:.4f}"
                      .format(iteration, epoch, Loss.item(),  Loss2.item()))
            Loss.backward()
            optimizer.step()

        psnr_epoch, ssim_epoch = validate(test_path1, epoch, result_path, psnr_epoch, ssim_epoch)

    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))

    return psnr_epoch, ssim_epoch


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

    for epoch in range(last_train + 1, max_iter + 1):
        print("epoch ", epoch)
        # checkpoint2(epoch, model_path)
        # psnr_epoch, ssim_epoch = train(epoch, learning_rate, result_path, psnr_epoch, ssim_epoch)
        psnr_epoch, ssim_epoch = validate(test_path1, epoch, result_path, psnr_epoch, ssim_epoch)
        psnr_mean = torch.mean(psnr_epoch[-1])
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