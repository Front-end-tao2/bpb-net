import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self._initialize_weights()

    def forward(self, x):
        feat1 = self.features[:4](x)
        feat2 = self.features[4:10](feat1)
        feat3 = self.features[10:18](feat2)
        feat4 = self.features[18:26](feat3)
        feat5 = self.features[26:](feat4)
        return [feat1, feat2, feat3, feat4, feat5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []

    for index in range(0, len(cfg)):
        if cfg[index] == 'down':
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=cfg[index+1], kernel_size=2, stride=2, padding=0)
            in_channels = cfg[index+1]

        else:
            conv2d = nn.Conv2d(in_channels = in_channels, out_channels = cfg[index], kernel_size = 3, stride = 1, padding = 1)
            in_channels = cfg[index]

        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)



# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    'D': [64, 64, 'down', 128, 128, 'down', 256, 256, 256, 'down', 512, 512, 512, 'down', 512, 512, 512]
}


def VGG16(in_channels=3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm=False, in_channels=in_channels), **kwargs)
    return model



class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        # unetUp(1024, 512)
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet_vgg_allconv(nn.Module):
    def __init__(self, num_classes=21):
        super(Unet_vgg_allconv, self).__init__()
        self.vgg = VGG16()
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




if __name__ == "__main__":
    a = torch.randn(2, 3, 320, 320)
    net = Unet_vgg_allconv(num_classes=3)
    print(net(a).shape)
    # total = sum(p.numel() for p in net.parameters())
    # print("Total params: %.3f" % (total/1000000))
    # 18.001
