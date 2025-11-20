from tools import *
import torch.nn.functional as F
import torch.nn as nn
from flownet import flownet


class E2ECNN(nn.Module):
    def __init__(self, in_ch):
        super(E2ECNN, self).__init__()
        # self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_ch, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.res1_en = res_block()
        self.res2_en = res_block()
        self.res3_en = res_block()
        self.res4_en = res_block()
        self.res5_en = res_block()
        self.res6_en = res_block()
        self.res7_en = res_block()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.res7_de = res_block()
        self.res6_de = res_block()
        self.res5_de = res_block()
        self.res4_de = res_block()
        self.res3_de = res_block()
        self.res2_de = res_block()
        self.res1_de = res_block()
        self.conv4 = nn.Conv2d(64, in_ch, kernel_size=1)
        self.relu4 = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, mask, meas_re, block_size, cs_rate):
        batch_size = meas_re.shape[0]
        maskt = mask.expand([batch_size, cs_rate, block_size, block_size])
        maskt = maskt.mul(meas_re)
        xt = maskt
        data = xt

        ###################  encoder  ##################
        x1 = self.conv1(xt)
        x1 = self.relu1(x1)
        x1 = self.res1_en(x1)
        x2 = self.res2_en(x1)
        x3 = self.res3_en(x2)
        x4 = self.res4_en(x3)
        x5 = self.res5_en(x4)
        x6 = self.res6_en(x5)
        x7 = self.res7_en(x6)

        x7_1 = self.conv2(x7)
        x7_1 = self.relu2(x7_1)
        x7_1 = self.conv3(x7_1)
        x7_1 = self.relu3(x7_1)
        ###################  decoder  ##################
        x7_1 = x7 + x7_1
        x6_1 = self.res7_de(x7_1)
        x6_1 = x6 + x6_1
        x5_1 = self.res6_de(x6_1)
        x5_1 = x5 + x5_1
        x4_1 = self.res5_de(x5_1)
        x4_1 = x4 + x4_1
        x3_1 = self.res4_de(x4_1)
        x3_1 = x3 + x3_1
        x2_1 = self.res3_de(x3_1)
        x2_1 = x2 + x2_1
        x1_1 = self.res2_de(x2_1)
        x1_1 = x1 + x1_1
        out = self.res1_de(x1_1)

        out = self.conv4(out)
        out = self.tanh(out)
        out = self.relu4(out)
        output = out + data

        return output


class reconnet(nn.Module):
    def __init__(self, in_ch):
        super(reconnet, self).__init__()
        self.motion_generation = motion_excitation(in_ch)
        self.flow_generation = flownet()
        self.motion_extract = motion_res(in_ch, in_ch)
        self.unet = Unet(in_ch, in_ch)
        self.flow_extract = flownet_feature(2 * (in_ch - 1), in_ch)
        self.recon = Unet(in_ch * 3, in_ch * 3)
        self.conv = up_feature(in_ch * 3, in_ch)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        data = x
        motion = self.motion_generation(x)
        flow = self.flow_generation(x)

        motion_fea = self.motion_extract(motion)
        flow_fea = self.flow_extract(flow)
        # out = self.unet(x)

        z = torch.cat([data, motion_fea, flow_fea], dim=1)
        z = self.recon(z)

        output = self.conv(z)
        output = self.relu(output)
        output = self.tanh(output)

        return (output + data)
