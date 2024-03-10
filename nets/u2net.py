import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self,in_channel=3,out_channel=3,dilation=1):
        super(ConvBlock, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=dilation,dilation=dilation),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv(x)

def Upsample(x,y):
    return F.interpolate(x,size=y.shape[2:],mode='bilinear', align_corners=False)

class RSU7(nn.Module):
    def __init__(self,input_channel,middle_channel,output_channel):
        super(RSU7, self).__init__()
        self.enconv1=ConvBlock(in_channel=input_channel,out_channel=output_channel,dilation=1)
        self.enconv2=ConvBlock(in_channel=output_channel,out_channel=middle_channel,dilation=1)
        self.pool0=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv3=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv4=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv5=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv6=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.pool4=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv7=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.enconv8=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=2)

        self.deconv1=ConvBlock(in_channel=2*middle_channel,out_channel=middle_channel,dilation=1)
        self.deconv2 = ConvBlock(in_channel=2 * middle_channel, out_channel=middle_channel, dilation=1)
        self.deconv3 = ConvBlock(in_channel=2 * middle_channel, out_channel=middle_channel, dilation=1)
        self.deconv4 = ConvBlock(in_channel=2 * middle_channel, out_channel=middle_channel, dilation=1)
        self.deconv5 = ConvBlock(in_channel=2 * middle_channel, out_channel=middle_channel, dilation=1)
        self.deconv6=ConvBlock(in_channel=2*middle_channel,out_channel=output_channel,dilation=1)
    def forward(self,x):
        out1=self.enconv1(x)
        out2=self.enconv2(out1)
        out2_1=self.pool0(out2)
        out3=self.enconv3(out2_1)
        out3=self.pool1(out3)
        out4=self.enconv4(out3)
        out4=self.pool2(out4)
        out5=self.enconv5(out4)
        out5=self.pool3(out5)
        out6=self.enconv6(out5)
        out6=self.pool4(out6)
        out7=self.enconv7(out6)
        out8=self.enconv8(out7)

        out9=torch.cat((out7,out8),dim=1)
        out9=self.deconv1(out9)
        out9= Upsample(out9, out6)
        out10=torch.cat((out6,out9),dim=1)
        out10=self.deconv2(out10)
        out10= Upsample(out10, out5)
        out11=torch.cat((out5,out10),dim=1)
        out11=self.deconv3(out11)
        out11 = Upsample(out11, out4)
        out12=torch.cat((out4,out11),dim=1)
        out12=self.deconv4(out12)
        out12=Upsample(out12,out3)
        out13=torch.cat((out3,out12),dim=1)
        out13=self.deconv5(out13)
        out13=Upsample(out13,out2)
        out14=torch.cat((out2,out13),dim=1)
        out14=self.deconv6(out14)
        out=out14+out1
        return out

class RSU6(nn.Module):
    def __init__(self,input_channel,middle_channel,output_channel):
        super(RSU6, self).__init__()
        self.enconv1=ConvBlock(in_channel=input_channel,out_channel=output_channel,dilation=1)
        self.enconv2=ConvBlock(in_channel=output_channel,out_channel=middle_channel,dilation=1)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv3=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv4=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv5=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.pool4=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv6=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.enconv7=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=2)
        self.deconv1=ConvBlock(in_channel=2*middle_channel,out_channel=middle_channel,dilation=1)
        self.deconv2 = ConvBlock(in_channel=2 * middle_channel, out_channel=middle_channel, dilation=1)
        self.deconv3 = ConvBlock(in_channel=2 * middle_channel, out_channel=middle_channel, dilation=1)
        self.deconv4 = ConvBlock(in_channel=2 * middle_channel, out_channel=middle_channel, dilation=1)
        self.deconv5 = ConvBlock(in_channel=2 * middle_channel, out_channel=output_channel, dilation=1)

    def forward(self,x):
        out1=self.enconv1(x)
        out2=self.enconv2(out1)
        out2_1=self.pool1(out2)
        out3=self.enconv3(out2_1)
        out3=self.pool2(out3)
        out4=self.enconv4(out3)
        out4=self.pool3(out4)
        out5=self.enconv5(out4)
        out5=self.pool4(out5)
        out6=self.enconv6(out5)
        out7=self.enconv7(out6)

        out8=torch.cat((out7,out6),dim=1)
        out8=self.deconv1(out8)
        out8=Upsample(out8,out5)
        out9=torch.cat((out8,out5),dim=1)
        out9=self.deconv2(out9)
        out9=Upsample(out9,out4)
        out10=torch.cat((out9,out4),dim=1)
        out10=self.deconv3(out10)
        out10=Upsample(out10,out3)
        out11=torch.cat((out10,out3),dim=1)
        out11=self.deconv4(out11)
        out11=Upsample(out11,out2)
        out12=torch.cat((out11,out2),dim=1)
        out12=self.deconv5(out12)
        out=out12+out1
        return out

class RSU5(nn.Module):
    def __init__(self,input_channel,middle_channel,output_channel):
        super(RSU5, self).__init__()
        self.enconv1=ConvBlock(in_channel=input_channel,out_channel=output_channel,dilation=1)
        self.enconv2=ConvBlock(in_channel=output_channel,out_channel=middle_channel,dilation=1)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv3=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv4=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv5=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.enconv6=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=2)
        self.deconv1=ConvBlock(in_channel=2*middle_channel,out_channel=middle_channel,dilation=1)
        self.deconv2 = ConvBlock(in_channel=2 * middle_channel, out_channel=middle_channel, dilation=1)
        self.deconv3 = ConvBlock(in_channel=2 * middle_channel, out_channel=middle_channel, dilation=1)
        self.deconv4 = ConvBlock(in_channel=2 * middle_channel, out_channel=output_channel, dilation=1)

    def forward(self,x):
        out1=self.enconv1(x)
        out2=self.enconv2(out1)
        out2_1=self.pool1(out2)
        out3=self.enconv3(out2_1)
        out3=self.pool2(out3)
        out4=self.enconv4(out3)
        out4=self.pool3(out4)
        out5=self.enconv5(out4)
        out6=self.enconv6(out5)
        out7=torch.cat((out6,out5),dim=1)
        out7=self.deconv1(out7)
        out7=Upsample(out7,out4)
        out8=torch.cat((out7,out4),dim=1)
        out8=self.deconv2(out8)
        out8=Upsample(out8,out3)
        out9=torch.cat((out8,out3),dim=1)
        out9=self.deconv3(out9)
        out9=Upsample(out9,out2)
        out10=torch.cat((out9,out2),dim=1)
        out10=self.deconv4(out10)
        out=out10+out1
        return out

class RSU4(nn.Module):
    def __init__(self,input_channel,middle_channel,output_channel):
        super(RSU4, self).__init__()
        self.enconv1=ConvBlock(in_channel=input_channel,out_channel=output_channel,dilation=1)
        self.enconv2=ConvBlock(in_channel=output_channel,out_channel=middle_channel,dilation=1)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv3=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv4=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=1)
        self.enconv5=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=2)
        self.deconv1=ConvBlock(in_channel=2*middle_channel,out_channel=middle_channel,dilation=1)
        self.deconv2 = ConvBlock(in_channel=2 * middle_channel, out_channel=middle_channel,dilation=1)
        self.deconv3=ConvBlock(in_channel=2*middle_channel,out_channel=output_channel,dilation=1)

    def forward(self,x):
        out1=self.enconv1(x)
        out2=self.enconv2(out1)
        out2_1=self.pool1(out2)
        out3=self.enconv3(out2_1)
        out3=self.pool2(out3)
        out4=self.enconv4(out3)
        out5=self.enconv5(out4)
        out6=torch.cat((out5,out4),dim=1)
        out6=self.deconv1(out6)
        out6=Upsample(out6,out3)
        out7=torch.cat((out6,out3),dim=1)
        out7=self.deconv2(out7)
        out7=Upsample(out7,out2)
        out8=torch.cat((out7,out2),dim=1)
        out9=self.deconv3(out8)
        out=out9+out1
        return out

class RSU4F(nn.Module):
    def __init__(self,input_channel,middle_channel,output_channel):
        super(RSU4F, self).__init__()
        self.conv1=ConvBlock(in_channel=input_channel,out_channel=output_channel,dilation=1)
        self.conv2=ConvBlock(in_channel=output_channel,out_channel=middle_channel,dilation=1)
        self.conv3=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=2)
        self.conv4=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=4)
        self.conv5=ConvBlock(in_channel=middle_channel,out_channel=middle_channel,dilation=8)
        self.conv6=ConvBlock(in_channel=2*middle_channel,out_channel=middle_channel,dilation=4)
        self.conv7=ConvBlock(in_channel=2*middle_channel,out_channel=middle_channel,dilation=2)
        self.conv8=ConvBlock(in_channel=2*middle_channel,out_channel=output_channel,dilation=1)

    def forward(self,x):
        out1=self.conv1(x)
        out2=self.conv2(out1)
        out3=self.conv3(out2)
        out4=self.conv4(out3)
        out5=self.conv5(out4)
        out6=self.conv6(torch.cat((out5,out4),dim=1))
        out7=self.conv7(torch.cat((out6,out3),dim=1))
        out8=self.conv8(torch.cat((out7,out2),dim=1))
        out=out8+out1
        return out


class U2Net(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(U2Net, self).__init__()
        self.enconv1=RSU7(input_channel=in_channel,middle_channel=32,output_channel=64)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv2=RSU6(input_channel=64,middle_channel=32,output_channel=128)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv3=RSU5(input_channel=128,middle_channel=64,output_channel=256)
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv4=RSU4(input_channel=256,middle_channel=128,output_channel=512)
        self.pool4=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv5=RSU4F(input_channel=512,middle_channel=256,output_channel=512)
        self.pool5=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.enconv6=RSU4F(input_channel=512,middle_channel=256,output_channel=512)
        self.side6=nn.Conv2d(512,out_channel,kernel_size=3,padding=1)
        self.deconv5=RSU4F(input_channel=1024,middle_channel=256,output_channel=512)
        self.side5=nn.Conv2d(512,out_channel,kernel_size=3,padding=1)
        self.deconv4=RSU4(input_channel=1024,middle_channel=128,output_channel=256)
        self.side4=nn.Conv2d(256,out_channel,kernel_size=3,padding=1)
        self.deconv3=RSU5(input_channel=512,middle_channel=64,output_channel=128)
        self.side3=nn.Conv2d(128,out_channel,kernel_size=3,padding=1)
        self.deconv2=RSU6(input_channel=256,middle_channel=32,output_channel=64)
        self.side2=nn.Conv2d(64,out_channel,kernel_size=3,padding=1)
        self.deconv1=RSU7(input_channel=128,middle_channel=16,output_channel=64)
        self.side1=nn.Conv2d(64,out_channel,kernel_size=3,padding=1)
        self.out=nn.Conv2d(out_channel*6,out_channel,kernel_size=1)
    def forward(self,x):
        out1=self.enconv1(x)
        out1x=self.pool1(out1)
        out2=self.enconv2(out1x)
        out2x=self.pool2(out2)
        out3=self.enconv3(out2x)
        out3x=self.pool3(out3)
        out4=self.enconv4(out3x)
        out4x=self.pool4(out4)
        out5=self.enconv5(out4x)
        out5x=self.pool5(out5)
        out6=self.enconv6(out5x)
        out6=Upsample(out6,out5)
        out5_1=self.deconv5(torch.cat((out6,out5),dim=1))
        out5_2=Upsample(out5_1,out4)
        out4_1=self.deconv4(torch.cat((out5_2,out4),dim=1))
        out4_2=Upsample(out4_1,out3)
        out3_1=self.deconv3(torch.cat((out4_2,out3),dim=1))
        out3_2=Upsample(out3_1,out2)
        out2_1=self.deconv2(torch.cat((out3_2,out2),dim=1))
        out2_2=Upsample(out2_1,out1)
        out1_1=self.deconv1(torch.cat((out2_2,out1),dim=1))

        outside1=torch.sigmoid(self.side1(out1_1))
        outside2=self.side2(out2_1)
        outside2=torch.sigmoid(Upsample(outside2,outside1))
        outside3 = self.side3(out3_1)
        outside3=torch.sigmoid(Upsample(outside3,outside1))
        outside4 = self.side4(out4_1)
        outside4=torch.sigmoid(Upsample(outside4,outside1))
        outside5 = self.side5(out5_1)
        outside5=torch.sigmoid(Upsample(outside5,outside1))
        outside6=self.side6(out6)
        outside6=torch.sigmoid(Upsample(outside6,outside1))
        out=self.out(torch.cat((outside1,outside2,outside3,outside4,outside5,outside6),dim=1))
        return out


if __name__ == '__main__':
    a=torch.randn(2,3,320,416)
    net=U2Net(3,2)
    print(net(a).shape)