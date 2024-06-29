#!/usr/bin/env python3
import torch
import torch.nn as nn


class pilotNet(nn.Module):
    def __init__(self) -> None:
        super(pilotNet, self).__init__()

        self.network = nn.Sequential(
            # 1 for filtered image
            # 3 for normal image
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 24, 5, 2),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1152, 1164),
            nn.Linear(1164, 100),
            nn.Linear(100, 50),
            nn.Linear(50, 10),
            nn.Linear(10, 2), 
        )
        
    def forward(self, x):
        return self.network(x)


class DeepPilot(nn.Module):
    def __init__(self, imgShape):
        super(DeepPilot, self).__init__()

        self.img_height = imgShape[0]
        self.img_width =  imgShape[1]
        self.num_channels = imgShape[2]

        self.cn_1 = nn.Conv2d(self.num_channels, 64, kernel_size=(7,7), stride=(2,2))
        self.po_1 = nn.MaxPool2d(kernel_size=(3,3),stride=(2,2))
        self.ln_1 = nn.BatchNorm2d(64)
        self.re_1 = nn.Conv2d(64, 64, kernel_size=(1,1))

        self.cn_2 = nn.Conv2d(64, 192, kernel_size=(3,3))
        self.ln_2 = nn.BatchNorm2d(192)
        self.po_2 = nn.MaxPool2d(kernel_size=(3,3),stride=(2,2))

        self.im_1_re_1 = nn.Conv2d(192, 96, kernel_size=(1,1))
        self.im_1_o_1 = nn.Conv2d(96, 128, kernel_size=(3,3),padding=(1,1))
        self.im_1_re_2 = nn.Conv2d(192, 16, kernel_size=(3,3))
        self.im_1_o_2 = nn.Conv2d(16, 32, kernel_size=(5,5),padding=(3,3))
        self.im_1_re_3 = nn.MaxPool2d(kernel_size=(3,3),stride=(1,1))
        self.im_1_o_3 = nn.Conv2d(192, 32, kernel_size=(1,1),padding=(1,1))
        self.im_1_o_0 = nn.Conv2d(192, 64, kernel_size=(1,1))

        self.im_2_re_1 = nn.Conv2d(128+32+32+64, 128, kernel_size=(1,1))
        self.im_2_o_1 = nn.Conv2d(128, 192, kernel_size=(3,3),padding=(1,1))
        self.im_2_re_2 = nn.Conv2d(128+32+32+64, 32, kernel_size=(1,1))
        self.im_2_o_2 = nn.Conv2d(32, 96, kernel_size=(5,5),padding=(2,2))
        self.im_2_re_3 = nn.MaxPool2d(kernel_size=(3,3),stride=(1,1))
        self.im_2_o_3 = nn.Conv2d(128+32+32+64, 64, kernel_size=(1,1),padding=(1,1))
        self.im_2_o_0 = nn.Conv2d(128+32+32+64, 128, kernel_size=(1,1))

        self.im_3_in = nn.MaxPool2d(kernel_size=(3,3),stride=(2,2))

        self.im_3_re_1 = nn.Conv2d(192+96+64+128, 96, kernel_size=(1,1))
        self.im_3_o_1 = nn.Conv2d(96, 208, kernel_size=(3,3),padding=(1,1))
        self.im_3_re_2 = nn.Conv2d(192+96+64+128, 16, kernel_size=(1,1))
        self.im_3_o_2 = nn.Conv2d(16, 48, kernel_size=(5,5),padding=(2,2))
        self.im_3_re_3 = nn.MaxPool2d(kernel_size=(3,3),stride=(1,1))
        self.im_3_o_3 = nn.Conv2d(192+96+64+128, 64, kernel_size=(1,1),padding=(1,1))
        self.im_3_o_0 = nn.Conv2d(192+96+64+128, 192, kernel_size=(1,1))

        self.last_po_1 = nn.AvgPool2d(kernel_size=(5,5),stride=(3,3))
        self.last_re_1 = nn.Conv2d(208+48+64+192, 128, kernel_size=(1,1))

        self.fc_r1 = nn.Linear(128 * 3 * 3, 1024)
        self.fc_p1 = nn.Linear(128 * 3 * 3, 1024)
        self.fc_y1 = nn.Linear(128 * 3 * 3, 1024)
        # self.fc_a1 = nn.Linear(128 * 3 * 3, 1024)

        self.fc_r2 = nn.Linear(1024, 1)
        self.fc_p2 = nn.Linear(1024, 1)
        self.fc_y2 = nn.Linear(1024, 1)
        # self.fc_a2 = nn.Linear(1024, 1)

    def forward(self, img):

        inp = self.cn_1(img)
        inp = torch.relu(inp)
        inp = self.po_1(inp)
        inp = torch.relu(inp)
        inp = self.ln_1(inp)
        inp = torch.relu(inp)
        inp = self.re_1(inp)
        inp = torch.relu(inp)
        inp = self.cn_2(inp)
        inp = torch.relu(inp)
        inp = self.ln_2(inp)
        inp = torch.relu(inp)
        inp = self.po_2(inp)
        inp = torch.relu(inp)

        icp1_out1 = self.im_1_re_1(inp)
        icp1_out1 = torch.relu(icp1_out1)
        icp1_out1 = self.im_1_o_1(icp1_out1)
        icp1_out1 = torch.relu(icp1_out1)

        icp1_out2 = self.im_1_re_2(inp)
        icp1_out2 = torch.relu(icp1_out2)
        icp1_out2 = self.im_1_o_2(icp1_out2)
        icp1_out2 = torch.relu(icp1_out2)

        icp1_out3 = self.im_1_re_3(inp)
        icp1_out3 = self.im_1_o_3(icp1_out3)
        icp1_out3 = torch.relu(icp1_out3)

        icp1_out0 = self.im_1_o_0(inp)

        icp1_out = torch.cat((icp1_out0, icp1_out1, icp1_out2, icp1_out3), dim=1)

        icp2_out1 = self.im_2_re_1(icp1_out)
        icp2_out1 = torch.relu(icp2_out1)
        icp2_out1 = self.im_2_o_1(icp2_out1)
        icp2_out1 = torch.relu(icp2_out1)

        icp2_out2 = self.im_2_re_2(icp1_out)
        icp2_out2 = torch.relu(icp2_out2)
        icp2_out2 = self.im_2_o_2(icp2_out2)
        icp2_out2 = torch.relu(icp2_out2)

        icp2_out3 = self.im_2_re_3(icp1_out)
        icp2_out3 = self.im_2_o_3(icp2_out3)
        icp2_out3 = torch.relu(icp2_out3)

        icp2_out0 = self.im_2_o_0(icp1_out)

        icp2_out = torch.cat((icp2_out0, icp2_out1, icp2_out2, icp2_out3), dim=1)

        icp3_in = self.im_3_in(icp2_out)

        icp3_out1 = self.im_3_re_1(icp3_in)
        icp3_out1 = torch.relu(icp3_out1)
        icp3_out1 = self.im_3_o_1(icp3_out1)
        icp3_out1 = torch.relu(icp3_out1)

        icp3_out2 = self.im_3_re_2(icp3_in)
        icp3_out2 = torch.relu(icp3_out2)
        icp3_out2 = self.im_3_o_2(icp3_out2)
        icp3_out2 = torch.relu(icp3_out2)

        icp3_out3 = self.im_3_re_3(icp3_in)
        icp3_out3 = self.im_3_o_3(icp3_out3)
        icp3_out3 = torch.relu(icp3_out3)

        icp3_out0 = self.im_3_o_0(icp3_in)

        icp3_out = torch.cat((icp3_out0, icp3_out1, icp3_out2, icp3_out3), dim=1)

        out = self.last_po_1(icp3_out)
        out = self.last_re_1(out)
        out = torch.relu(out)
        
        out = out.reshape(out.size(0), -1)

        rOut = self.fc_r1(out)
        rOut = torch.relu(rOut)
        rOut = self.fc_r2(rOut)

        pOut = self.fc_p1(out)
        pOut = torch.relu(pOut)
        pOut = self.fc_p2(pOut)

        yOut = self.fc_y1(out)
        yOut = torch.relu(yOut)
        yOut = self.fc_y2(yOut)

        # aOut = self.fc_a1(out)
        # aOut = torch.relu(aOut)
        # aOut = self.fc_a2(aOut)

        # out_final = torch.cat((rOut, pOut, yOut, aOut), dim=1)
        out_final = (rOut + pOut + yOut) / 3

        return out_final