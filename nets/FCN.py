import os
from torchvision.models import vgg16_bn
import torch.nn as nn
import torch


class FCN8s(nn.Module):
    # 定义双线性插值，作为转置卷积的初始化权重参数
    def __init__(self,num_classes=10):
        super(FCN8s, self).__init__()

        # (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (2): ReLU(inplace=True)
        # (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (5): ReLU(inplace=True)
        # (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        pretrained_net = vgg16_bn(pretrained=False)
        # pool1 按照上述图片命名
        self.pool1 = pretrained_net.features[:7]
        self.pool2 = pretrained_net.features[7:14]
        self.pool3 = pretrained_net.features[14:24]
        self.pool4 = pretrained_net.features[24:34]
        self.pool5 = pretrained_net.features[34:]

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):  # x = 352, 480，3 = height, width, channels
        # s1第一个下采样块的输出特征图
        # x为输入图像
        # layer1 为Vgg网络下，第一个下采样块和之前的结构
        s1 = self.pool1(x)  # >>> s1 = 176, 240，64 # 1/2
        s2 = self.pool2(s1)  # >>> s2 = 88, 120, 128 # 1/4
        s3 = self.pool3(s2)  # >>> s3 = 44, 60, 256  # 1/8
        s4 = self.pool4(s3)  # >>> s4 = 22, 30, 512  # 1/16
        s5 = self.pool5(s4)  # >>> s5 = 11, 15, 512  # 1/32 通道数增加到512，到700左右就行了，不闭增加过多
        # relu 用来防止梯度消失
        # bn 层用来，使数据保持高斯分布
        # bn层 的里面是relu层，外面是转置卷积层， relu内部接转置卷积结果
        scores = self.relu(self.deconv1(s5))  # h,w,n = 22, 30, 512   1/16
        scores = self.bn1(scores + s4)  # h,w,n = 22, 30, 512   1/16
        scores = self.relu(self.deconv2(scores))  # h,w,n = 44 , 60, 256   1/8
        scores = self.bn2(scores + s3)
        scores = self.bn3(self.relu(self.deconv3(scores)))  # h,w,n = 88, 120, 128   1/4
        scores = self.bn4(self.relu(self.deconv4(scores)))  # h,w,n = 176, 240, 64   1/2
        scores = self.bn5(self.relu(self.deconv5(scores)))  # h,w,n = 352, 480, 32    1/1
        return self.classifier(scores)  # h,w,n= 352, 480, 5  1/1



if __name__ == '__main__':
    a=torch.randn(2,3,320,416)
    net=FCN8s(num_classes=2)
    print(net(a).shape)


