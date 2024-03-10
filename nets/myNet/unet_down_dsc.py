import torch
import torch.nn as nn

# 注意力机制
# 注意力机制
# 注意力机制
# 注意力机制
# 注意力机制

class CAM(nn.Module):
    def __init__(self, in_channel, reduction=8):
        super(CAM, self).__init__()
        # 通道注意力机制
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel // reduction, out_features=in_channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力机制
        maxout = self.max_pool(x)
        maxout = self.mlp(maxout.view(maxout.size(0), -1))
        avgout = self.avg_pool(x)
        avgout = self.mlp(avgout.view(avgout.size(0), -1))
        channel_out = self.sigmoid(maxout + avgout)
        channel_out = channel_out.view(x.size(0), x.size(1), 1, 1)
        out = channel_out * x
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


# 通道注意力机制
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class CBAM(nn.Module):
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        super(CBAM, self).__init__()
        #通道注意力机制
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #空间注意力机制
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size ,stride=1,padding=kernel_size//2,bias=False)

    def forward(self,x):
        #通道注意力机制
        maxout=self.max_pool(x)
        maxout=self.mlp(maxout.view(maxout.size(0),-1))
        avgout=self.avg_pool(x)
        avgout=self.mlp(avgout.view(avgout.size(0),-1))
        channel_out=self.sigmoid(maxout+avgout)
        channel_out=channel_out.view(x.size(0),x.size(1),1,1)
        channel_out=channel_out*x
        #空间注意力机制
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        out=out*channel_out
        return out


# BSC-U结构
# BSC-U结构
# BSC-U结构
# BSC-U结构
# BSC-U结构

# 需要指定输入通道数和输出的通道数 不改变宽高
class Pointwise_Conv(nn.Module):
    def __init__(self, inner_channels, out_channels):
        super(Pointwise_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=inner_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                              groups=1)

    def forward(self, x):
        out = self.conv(x)
        return out


# 需要指定输入通道数、卷积核、步长、padding值    输出的通道数与输入的通道数相同
class Depthwise_Conv(nn.Module):
    def __init__(self, inner_channels, kernel_size=3, stride=1, padding=1):
        super(Depthwise_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=inner_channels, out_channels=inner_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=inner_channels)

    def forward(self, x):
        out = self.conv(x)
        return out


# 需要指定 inner_channels, out_channels, kernel_size, stride, padding
class DSC(nn.Module):
    def __init__(self, inner_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.depthwise_conv_block = Depthwise_Conv(inner_channels=inner_channels, kernel_size=kernel_size,
                                                   stride=stride, padding=padding)
        self.pointwise_conv_block = Pointwise_Conv(inner_channels=inner_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.depthwise_conv_block(x)
        x = self.pointwise_conv_block(x)
        return x


class BSC_U(nn.Module):
    def __init__(self, inner_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BSC_U, self).__init__()
        self.depthwise_conv_block = Depthwise_Conv(inner_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                   padding=padding)
        self.pointwise_conv_block = Pointwise_Conv(inner_channels=inner_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.pointwise_conv_block(x)
        x = self.depthwise_conv_block(x)
        return x


class BSC_S(nn.Module):
    def __init__(self, inner_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BSC_S, self).__init__()
        self.pointwise_conv_block = Pointwise_Conv(inner_channels=inner_channels, out_channels=inner_channels)
        self.pointwise_conv_block_2 = Pointwise_Conv(inner_channels=inner_channels, out_channels=out_channels)
        self.depthwise_conv_block = Depthwise_Conv(inner_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                   padding=padding)

    def forward(self, x):
        x = self.pointwise_conv_block(x)
        x = self.pointwise_conv_block_2(x)
        x = self.depthwise_conv_block(x)
        return x


class BSC_S_Ratio(nn.Module):
    def __init__(self, inner_channels, out_channels, kernel_size=3, stride=1, padding=1, mid_ratio=2.5):
        super(BSC_S_Ratio, self).__init__()
        self.pointwise_conv_block = Pointwise_Conv(inner_channels=inner_channels,
                                                   out_channels=int(inner_channels * mid_ratio))
        self.pointwise_conv_block_2 = Pointwise_Conv(inner_channels=int(inner_channels * mid_ratio),
                                                     out_channels=out_channels)
        self.depthwise_conv_block = Depthwise_Conv(inner_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                   padding=padding)

    def forward(self, x):
        x = self.pointwise_conv_block(x)
        x = self.pointwise_conv_block_2(x)
        x = self.depthwise_conv_block(x)
        return x


class DSC_And_BSC(nn.Module):
    def __init__(self, inner_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DSC_And_BSC, self).__init__()
        self.DSC_branch = DSC(inner_channels=inner_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.BSC_branch = BSC_U(inner_channels=inner_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding)
        self.conv = Pointwise_Conv(inner_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        dsc_branch_res = self.DSC_branch(x)
        bsc_branch_res = self.BSC_branch(x)
        x = self.conv(dsc_branch_res + bsc_branch_res)
        return x





# 注意力机制层
# 注意力机制层
# 注意力机制层
# 注意力机制层
# 注意力机制层

class ABR(nn.Module):
    def __init__(self, inner_channels_list):
        super(ABR, self).__init__()
        self.attention_block_1 = CoordAtt(inp=inner_channels_list[0], oup=inner_channels_list[0])
        self.attention_block_2 = CoordAtt(inp=inner_channels_list[1], oup=inner_channels_list[1])
        self.attention_block_3 = CoordAtt(inp=inner_channels_list[2], oup=inner_channels_list[2])
        self.attention_block_4 = CoordAtt(inp=inner_channels_list[3], oup=inner_channels_list[3])
        self.attention_block_5 = CoordAtt(inp=inner_channels_list[4], oup=inner_channels_list[4])

    def forward(self, features_list):
        feat1 = self.attention_block_1(features_list[0])
        feat2 = self.attention_block_2(features_list[1])
        feat3 = self.attention_block_3(features_list[2])
        feat4 = self.attention_block_4(features_list[3])
        feat5 = self.attention_block_5(features_list[4])
        return [feat1, feat2, feat3, feat4, feat5]

class CBAM(nn.Module):
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        super(CBAM, self).__init__()
        #通道注意力机制
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #空间注意力机制
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size ,stride=1,padding=kernel_size//2,bias=False)

    def forward(self,x):
        #通道注意力机制
        maxout=self.max_pool(x)
        maxout=self.mlp(maxout.view(maxout.size(0),-1))
        avgout=self.avg_pool(x)
        avgout=self.mlp(avgout.view(avgout.size(0),-1))
        channel_out=self.sigmoid(maxout+avgout)
        channel_out=channel_out.view(x.size(0),x.size(1),1,1)
        channel_out=channel_out*x
        #空间注意力机制
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        out=out*channel_out
        return out

class CBAMLayer(nn.Module):
    def __init__(self):
        super(CBAMLayer, self).__init__()
        self.CBAM_1 = CBAM(in_channel=64)
        self.CBAM_2 = CBAM(in_channel=128)
        self.CBAM_3 = CBAM(in_channel=256)
        self.CBAM_4 = CBAM(in_channel=512)
        self.CBAM_5 = CBAM(in_channel=512)

    def forward(self,features_list):
        feat1 = self.CBAM_1(features_list[0])
        feat2 = self.CBAM_2(features_list[1])
        feat3 = self.CBAM_3(features_list[2])
        feat4 = self.CBAM_4(features_list[3])
        feat5 = self.CBAM_5(features_list[4])
        return [feat1, feat2, feat3, feat4, feat5]

# 大尺度卷积核层
# 大尺度卷积核层
# 大尺度卷积核层
# 大尺度卷积核层


#  def __init__(self, inner_channels, out_channels, kernel_size=3, stride=1, padding=1):
class LargeKernelBSC_S(nn.Module):
    def __init__(self, inner_channels_list):
        super(LargeKernelBSC_S, self).__init__()
        self.largerKernelBSC_S_1 = BSC_S_Ratio(inner_channels=inner_channels_list[0],
                                               out_channels=inner_channels_list[0], kernel_size=21, padding=10,
                                               stride=1)
        self.largerKernelBSC_S_2 = BSC_S_Ratio(inner_channels=inner_channels_list[1],
                                               out_channels=inner_channels_list[1], kernel_size=21, padding=10,
                                               stride=1)
        self.largerKernelBSC_S_3 = BSC_S_Ratio(inner_channels=inner_channels_list[2],
                                               out_channels=inner_channels_list[2], kernel_size=21, padding=10,
                                               stride=1)
        self.largerKernelBSC_S_4 = BSC_S_Ratio(inner_channels=inner_channels_list[3],
                                               out_channels=inner_channels_list[3], kernel_size=21, padding=10,
                                               stride=1)
        self.largerKernelBSC_S_5 = BSC_S_Ratio(inner_channels=inner_channels_list[4],
                                               out_channels=inner_channels_list[4], kernel_size=21, padding=10,
                                               stride=1)

    def forward(self, features_list):
        feat1 = self.largerKernelBSC_S_1(features_list[0]) + features_list[0]
        feat2 = self.largerKernelBSC_S_2(features_list[1]) + features_list[1]
        feat3 = self.largerKernelBSC_S_3(features_list[2]) + features_list[2]
        feat4 = self.largerKernelBSC_S_4(features_list[3]) + features_list[3]
        feat5 = self.largerKernelBSC_S_5(features_list[4]) + features_list[4]
        return [feat1, feat2, feat3, feat4, feat5]


class BSC_S_Muti_Cat(nn.Module):
    def __init__(self, inner_channels):
        super(BSC_S_Muti_Cat, self).__init__()
        self.BSC_S_Muti_Cat_1 = BSC_S(inner_channels=inner_channels, out_channels=int(inner_channels / 4),
                                      kernel_size=1, padding=0, stride=1)
        self.BSC_S_Muti_Cat_2 = BSC_S(inner_channels=inner_channels, out_channels=int(inner_channels / 4),
                                      kernel_size=3, padding=1, stride=1)
        self.BSC_S_Muti_Cat_3 = BSC_S(inner_channels=inner_channels, out_channels=int(inner_channels / 4),
                                      kernel_size=5, padding=2, stride=1)
        self.BSC_S_Muti_Cat_4 = BSC_S(inner_channels=inner_channels, out_channels=int(inner_channels / 4),
                                      kernel_size=7, padding=3, stride=1)
        # self.BSC_S_Muti_last = BSC_S(inner_channels=inner_channels*4, out_channels=inner_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = torch.cat(
            [self.BSC_S_Muti_Cat_1(x), self.BSC_S_Muti_Cat_2(x), self.BSC_S_Muti_Cat_3(x), self.BSC_S_Muti_Cat_4(x)],
            dim=1)
        # x = self.BSC_S_Muti_last(x)
        return x

class BSC_S_Muti_Cat_Layer(nn.Module):
    def __init__(self, inner_channels_list):
        # [64, 128, 256, 512, 512]
        super(BSC_S_Muti_Cat_Layer, self).__init__()
        self.BSC_S_Muti_Cat_Block_1 = BSC_S_Muti_Cat(inner_channels=inner_channels_list[0])
        self.BSC_S_Muti_Cat_Block_2 = BSC_S_Muti_Cat(inner_channels=inner_channels_list[1])
        self.BSC_S_Muti_Cat_Block_3 = BSC_S_Muti_Cat(inner_channels=inner_channels_list[2])
        self.BSC_S_Muti_Cat_Block_4 = BSC_S_Muti_Cat(inner_channels=inner_channels_list[3])
        self.BSC_S_Muti_Cat_Block_5 = BSC_S_Muti_Cat(inner_channels=inner_channels_list[4])

    def forward(self, features_list):
        feat1 = self.BSC_S_Muti_Cat_Block_1(features_list[0])
        feat2 = self.BSC_S_Muti_Cat_Block_2(features_list[1])
        feat3 = self.BSC_S_Muti_Cat_Block_3(features_list[2])
        feat4 = self.BSC_S_Muti_Cat_Block_4(features_list[3])
        feat5 = self.BSC_S_Muti_Cat_Block_5(features_list[4])
        return [feat1, feat2, feat3, feat4, feat5]




class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        if in_size != 1024:
            self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            # self.up = nn.ConvTranspose2d(in_channels=in_size - out_size, out_channels=in_size - out_size, kernel_size=3, padding=1, stride=2, output_padding=(1,1))
            self.relu = nn.ReLU(inplace=True)
        else:
            block = BSC_S_Ratio
            self.conv1 = block(in_size, out_size)
            self.conv2 = block(out_size, out_size)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            # self.up = nn.ConvTranspose2d(in_channels=in_size - out_size, out_channels=in_size - out_size, kernel_size=3, padding=1, stride=2, output_padding=(1,1))
            self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs



# 特征融合层
# 特征融合层
# 特征融合层
# 特征融合层


class multi_fusion(nn.Module):
    def __init__(self):
        super(multi_fusion, self).__init__()
        self.maxpooling_down_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpooling_down_4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.maxpooling_down_8 = nn.MaxPool2d(kernel_size=8, stride=8)
        # self.conv_1 = nn.Conv2d(960,512,kernel_size=3,padding=1)
        # self.conv_2 = nn.Conv2d(960, 512, kernel_size=3,padding=1)
        # self.conv_1 = nn.Conv2d(960, 512, kernel_size=1)
        # self.conv_2 = nn.Conv2d(960, 512, kernel_size=1)
        self.conv_1 = BSC_S_Ratio(960, 512, kernel_size=3, padding=1, stride=1)
        self.conv_2 = BSC_S_Ratio(960, 512, kernel_size=3, padding=1, stride=1)

    def forward(self, inputs):
        feat1 = inputs[0]
        feat2 = inputs[1]
        feat3 = inputs[2]
        feat4 = inputs[3]
        feat5 = inputs[4]
        feat4 = torch.cat([self.maxpooling_down_8(feat1), self.maxpooling_down_4(feat2), self.maxpooling_down_2(feat3), feat4], dim=1)
        feat4 = self.conv_1(feat4)

        feat5 = torch.cat([self.maxpooling_down_2(self.maxpooling_down_8(feat1)), self.maxpooling_down_2(self.maxpooling_down_4(feat2)), self.maxpooling_down_2(self.maxpooling_down_2(feat3)), feat5], dim=1)
        feat5 = self.conv_2(feat5)

        return [feat1,feat2,feat3,feat4,feat5]

class multi_fusion2(nn.Module):
    def __init__(self):
        super(multi_fusion2, self).__init__()
        self.maxpooling_down_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpooling_down_4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.maxpooling_down_8 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.maxpooling_down_16 = nn.MaxPool2d(kernel_size=16, stride=16)
        self.conv_1 = nn.Conv2d(960,512,kernel_size=1)
        self.conv_2 = nn.Conv2d(960, 512, kernel_size=1)

    def forward(self, inputs):
        feat1 = inputs[0]
        feat2 = inputs[1]
        feat3 = inputs[2]
        feat4 = inputs[3]
        feat5 = inputs[4]
        feat4 = torch.cat([self.maxpooling_down_8(feat1), self.maxpooling_down_4(feat2), self.maxpooling_down_2(feat3), feat4], dim=1)
        feat4 = self.conv_1(feat4)

        feat5 = torch.cat([self.maxpooling_down_16(feat1), self.maxpooling_down_8(feat2), self.maxpooling_down_4(feat3), feat5], dim=1)
        feat5 = self.conv_2(feat5)

        return [feat1,feat2,feat3,feat4,feat5]


class myVgg(nn.Module):
    def __init__(self, inner_channels=3,
                 model_structure=[64, 64, 'down', 128, 128, 'down', 256, 256, 256, 'down', 512, 512, 512, 'down', 512,
                                  512, 512], block_type='DSC', batch_norm=False):
        super(myVgg, self).__init__()
        block_type_list = ['DSC', 'BSC_U', 'BSC_S', 'BSC_S_Ratio', 'DSC_And_BSC']
        assert block_type in block_type_list, 'block type must be in [DSC, BSC_U, BSC_S, BSC_S_Ratio, DSC_And_BSC]'

        if block_type == 'DSC':
            block = DSC
        elif block_type == 'BSC_U':
            block = BSC_U
        elif block_type == 'BSC_S':
            block = BSC_S
        elif block_type == 'BSC_S_Ratio':
            block = BSC_S_Ratio
        elif block_type == 'DSC_And_BSC':
            block = DSC_And_BSC

        layers = []

        if False:
            for index in range(0, len(model_structure)):
                if model_structure[index] == 'down':
                    conv2d = nn.Conv2d(in_channels=inner_channels, out_channels=model_structure[index + 1],
                                       kernel_size=2, stride=2, padding=0)
                    inner_channels = model_structure[index + 1]
                else:
                    conv2d = block(inner_channels=inner_channels, out_channels=model_structure[index])
                    inner_channels = model_structure[index]
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(inner_channels), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            self.features = nn.Sequential(*layers)
        else:
            for index in range(0, len(model_structure)):
                if model_structure[index] == 'down':
                    conv2d = nn.Conv2d(in_channels=inner_channels, out_channels=model_structure[index + 1],
                                       kernel_size=2, stride=2, padding=0)
                    # conv2d = nn.MaxPool2d(kernel_size=2, stride=2)
                    inner_channels = model_structure[index + 1]
                else:
                    if model_structure[index] == 512:
                        conv2d = block(inner_channels=inner_channels, out_channels=model_structure[index])
                    else:
                        conv2d = nn.Conv2d(in_channels=inner_channels, out_channels=model_structure[index],
                                           kernel_size=3, padding=1, stride=1)
                    inner_channels = model_structure[index]
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(inner_channels), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            self.features = nn.Sequential(*layers)

    def forward(self, x):
        feat1 = self.features[:4](x)
        feat2 = self.features[4:10](feat1)
        feat3 = self.features[10:18](feat2)
        feat4 = self.features[18:26](feat3)
        feat5 = self.features[26:](feat4)
        return [feat1, feat2, feat3, feat4, feat5]

class UNet_DSC_VGG(nn.Module):
    def __init__(self, num_classes=21):
        super(UNet_DSC_VGG, self).__init__()
        self.vgg = myVgg(
            model_structure=[64, 64, 'down', 128, 128, 'down', 256, 256, 256, 'down', 512, 512, 512, 'down', 512, 512,
                             512], block_type='BSC_S_Ratio', batch_norm=False)

        # self.multi_fusion_Layer = multi_fusion()
        # self.largetKernelLayer = LargeKernelBSC_S(inner_channels_list=[64, 128, 256, 512, 512])
        # #
        # self.CBAMLayer = CBAMLayer()

        # self.CBAM1 = CBAM(in_channel=64)
        # self.CBAM2 = CBAM(in_channel=128)
        # self.CBAM3 = CBAM(in_channel=256)
        # self.CBAM4 = CBAM(in_channel=512)

        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]

        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        self.up_conv = None
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        # [feat1, feat2, feat3, feat4, feat5] = self.CBAMLayer([feat1, feat2, feat3, feat4, feat5])
        # [feat1, feat2, feat3, feat4, feat5] = self.multi_fusion_Layer([feat1, feat2, feat3, feat4, feat5])
        # [feat1, feat2, feat3, feat4, feat5] = self.largetKernelLayer.forward([feat1, feat2, feat3, feat4, feat5])
        up4 = self.up_concat4(feat4, feat5)
        # up4 = self.CBAM4(up4)
        up3 = self.up_concat3(feat3, up4)
        # up3 = self.CBAM3(up3)
        up2 = self.up_concat2(feat2, up3)
        # up2 = self.CBAM2(up2)
        up1 = self.up_concat1(feat1, up2)
        # up1 = self.CBAM1(up1)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        return final


if __name__ == '__main__':
    inputs = torch.randn(2, 3, 320, 320)
    net = UNet_DSC_VGG(num_classes=2)
    # print(net)
    out = net(inputs)
    total = sum(p.numel() for p in net.parameters())
    print("Total params: %.2f" % (total / 1000000))
