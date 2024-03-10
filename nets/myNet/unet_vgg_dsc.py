import torch
import torch.nn as nn


# 需要指定输入通道数和输出的通道数 不改变宽高
class Pointwise_Conv(nn.Module):
    def __init__(self, inner_channels, out_channels):
        super(Pointwise_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=inner_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
    def forward(self, x):
        out = self.conv(x)
        return out

# 需要指定输入通道数、卷积核、步长、padding值    输出的通道数与输入的通道数相同
class Depthwise_Conv(nn.Module):
    def __init__(self, inner_channels, kernel_size=3, stride=1, padding=1):
        super(Depthwise_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=inner_channels, out_channels=inner_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=inner_channels)
    def forward(self, x):
        out = self.conv(x)
        return out

# 需要指定 inner_channels, out_channels, kernel_size, stride, padding
class DSC(nn.Module):
    def __init__(self, inner_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.depthwise_conv_block = Depthwise_Conv(inner_channels=inner_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pointwise_conv_block = Pointwise_Conv(inner_channels=inner_channels, out_channels=out_channels)
    def forward(self, x):
        x = self.depthwise_conv_block(x)
        x = self.pointwise_conv_block(x)
        return x

class BSC_U(nn.Module):
    def __init__(self, inner_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BSC_U, self).__init__()
        self.depthwise_conv_block = Depthwise_Conv(inner_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
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
        self.depthwise_conv_block = Depthwise_Conv(inner_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        x = self.pointwise_conv_block(x)
        x = self.pointwise_conv_block_2(x)
        x = self.depthwise_conv_block(x)
        return x

class BSC_S_Ratio(nn.Module):
    def __init__(self, inner_channels, out_channels, kernel_size=3, stride=1, padding=1, mid_ratio=1):
        super(BSC_S_Ratio, self).__init__()
        self.pointwise_conv_block = Pointwise_Conv(inner_channels=inner_channels, out_channels=inner_channels * mid_ratio)
        self.pointwise_conv_block_2 = Pointwise_Conv(inner_channels=inner_channels * mid_ratio, out_channels=out_channels)
        self.depthwise_conv_block = Depthwise_Conv(inner_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        x = self.pointwise_conv_block(x)
        x = self.pointwise_conv_block_2(x)
        x = self.depthwise_conv_block(x)
        return x

class DSC_And_BSC(nn.Module):
    def __init__(self, inner_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DSC_And_BSC, self).__init__()
        self.DSC_branch = DSC(inner_channels=inner_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.BSC_branch = BSC_U(inner_channels=inner_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = Pointwise_Conv(inner_channels=out_channels, out_channels=out_channels)
    def forward(self, x):
        dsc_branch_res = self.DSC_branch(x)
        bsc_branch_res = self.BSC_branch(x)
        x = self.conv(dsc_branch_res + bsc_branch_res)
        return x


class myVgg(nn.Module):
    def __init__(self, inner_channels=3, model_structure=[64, 64, 'down', 128, 128, 'down', 256, 256, 256, 'down', 512, 512, 512, 'down', 512, 512, 512], block_type='DSC', batch_norm=False):
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
        for index in range(0, len(model_structure)):
            if model_structure[index] == 'down':
                conv2d = nn.Conv2d(in_channels=inner_channels, out_channels=model_structure[index + 1], kernel_size=2, stride=2, padding=0)
                inner_channels = model_structure[index + 1]
            else:
                conv2d = block(inner_channels=inner_channels, out_channels=model_structure[index])
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


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        # unetUp(1024, 512)
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
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


class UNet_DSC_VGG(nn.Module):
    def __init__(self, num_classes=21):
        super(UNet_DSC_VGG, self).__init__()
        self.vgg = myVgg(model_structure=[64, 64, 'down', 128, 128, 'down', 256, 256, 256, 'down', 512, 512, 512, 'down', 512, 512, 512],  block_type='BSC_S', batch_norm=False)
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

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final


class Feature_PN(nn.Module):
    def __init__(self, feature1_sizes, feature2_sizes):
        super(Feature_PN, self).__init__()
        self.conv_left = nn.Conv2d(in_channels=feature1_sizes, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv_left_2 = nn.Conv2d(in_channels=feature2_sizes, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, feature1, feature2):
        feature1 = self.conv_left(feature1)
        feature1 = self.up(feature1)
        feature2 = self.conv_left_2(feature2)
        out = feature1 + feature2
        return out

class DSC_VGG_FPN(nn.Module):
    def __init__(self, num_classes=2):
        super(DSC_VGG_FPN, self).__init__()
        self.backbone = myVgg(model_structure=[64, 64, 'down', 128, 128, 'down', 256, 256, 256, 'down', 512, 512, 512, 'down', 512, 512, 512],  block_type='BSC_S', batch_norm=False)
        self.up_add1  = Feature_PN(feature1_sizes=512, feature2_sizes=512)
        self.up_add2 = Feature_PN(feature1_sizes=256, feature2_sizes=256)
        self.up_add3 = Feature_PN(feature1_sizes=256, feature2_sizes=128)
        self.up_add4 = Feature_PN(feature1_sizes=256, feature2_sizes=64)

        self.pred = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.pred2 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.backbone.forward(inputs)
        out = self.up_add1(feat5, feat4)
        out = self.up_add2(out, feat3)
        out = self.up_add3(out, feat2)
        out = self.up_add4(out, feat1)

        out = self.pred(out)
        out = self.pred2(out)

        return out

if __name__ == '__main__':
    inputs = torch.randn(2,3,320,320)
    #
    # p_conv = Pointwise_Conv(3, 64)
    # outs = p_conv(inputs)
    # print(outs.shape)
    #
    # d_conv = Depthwise_Conv(inner_channels=3, kernel_size=3, stride=1, padding=1)
    # outs = d_conv(inputs)
    # print(outs.shape)
    #
    # dsc_block = DSC(inner_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    # outpus = dsc_block(inputs)
    # print(outpus.shape)
    #
    # bsc_u_block = BSC_U(inner_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    # outpus = bsc_u_block(inputs)
    # print(outpus.shape)
    #
    # bsc_s_block = BSC_S(inner_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    # outpus = bsc_s_block(inputs)
    # print(outpus.shape)
    #
    # bsc_s_ratio_block = BSC_S_Ratio(inner_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, mid_ratio=32)
    # outpus = bsc_s_ratio_block(inputs)
    # print(outpus.shape)
    #
    # DSC_And_BSC_block = DSC_And_BSC(inner_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    # outpus = DSC_And_BSC_block(inputs)
    # print(outpus.shape)

    # net = myVgg(model_structure=[64, 64, 'down', 128, 128, 'down', 256, 256, 256, 'down', 512, 512, 512, 'down', 512, 512, 512],  block_type='DSC_And_BSC', batch_norm=False)
    # print(net)
    # out = net(inputs)
    # print(out.shape)
    #
    # total = sum(p.numel() for p in net.parameters())
    # print("Total params: %.2f" % (total / 1000000))

    net = DSC_VGG_FPN(num_classes=2)
    print(net)
    out = net(inputs)
    print(out.shape)
    total = sum(p.numel() for p in net.parameters())
    print("Total params: %.2f" % (total / 1000000))