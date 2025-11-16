import torch
import torch.nn as nn
import torch.nn.functional as F

class res_block(nn.Module):
    def __init__(self):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1))
        # self.batch1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1))
        # self.batch2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        temp = x
        out1 = self.conv1(x)
        # out1 = self.batch1(out1)
        out1 = self.relu1(out1)
        out1 = self.conv2(out1)
        # out1 = self.batch2(out1)
        out1 = out1 + temp
        output = self.relu2(out1)

        return output

class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x

class up_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        # cnn6 in paper ECCV 2020, fig.3 right
        super(up_feature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 40, 3, stride=1, padding=1),
            nn.Conv2d(40, 30, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(30, 20, 3, stride=1, padding=1),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1),
            nn.Conv2d(20, out_ch, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class motion_res(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(motion_res, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x = x1 + x
        x1 = self.conv2(x)
        x = x1 + x
        x1 = self.conv3(x)
        x = x1 + x
        return x

class motion_excitation(nn.Module):

    def __init__(self, cr):
        super(motion_excitation, self).__init__()
        compress_rate = cr
        self.channel = compress_rate  #c
        self.reduction = cr
        self.n_segment = 1
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel // self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)

        self.conv2 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel // self.reduction,
            kernel_size=3,
            padding=1,
            bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

        self.conv3 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        # self.identity = nn.Identity()

    def forward(self, x):
        nt, c, h, w = x.size()
        motion = torch.zeros(nt, c, h, w).cuda()
        motion[:, 0, :, :] = x[:, 0, :, :]
        for i in range(c-1):
            xt = x[:, i, :, :]
            xt1 = x[:, i+1, :, :]
            trans = self.conv2(xt1.view(nt, 1, h, w))
            motion[:, i+1, :, :] = trans.view(nt, h, w) - xt
        y1 = self.avg_pool(motion)  # nt, c, 1, 1
        y2 = self.conv3(y1)  # nt, c, 1, 1
        y3 = self.bn3(y2)  # nt, c, 1, 1
        y4 = self.sigmoid(y3)  # nt, c, 1, 1
        y4 = y4 - 0.5
        output = x + x * y4.expand_as(x)
        return output


class flownet_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(flownet_feature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 4, 5, stride=1, padding=2),
            nn.Conv2d(4, 4, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4, out_ch, 1, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.dconv_down1 = double_conv(in_ch, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)

        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()

    def forward(self, x):
        inputs = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)

        x = self.upsample2(conv3)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs

        return out

class flownet_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(flownet_feature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 4, 5, stride=1, padding=2),
            nn.Conv2d(4, 4, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4, out_ch, 1, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x