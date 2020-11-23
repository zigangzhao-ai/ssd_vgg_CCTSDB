import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
from torch.nn.parameter import Parameter
from data.config import USE_ECA, USE_SE
from attention import Bottleneck, SEModule, ECAModule, Upsample
import os


#############################【SSD中融合特征显著性模块CBAM】######################
class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        # 经过修改的vgg网络
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            #self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            self.detect = Detect()
    

    # =====se_fpn新增==================
        # pool2到conv4_3  扩张卷积，尺度少一半
        self.DilationConv_128_128 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2,
                                              stride=2)
        # conv4_3到conv4_3  尺度不变
        self.conv_512_256 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        # fc7 到 conv4_3    反卷积上采样，尺度大一倍
        #self.DeConv_1024_128 = nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=2, stride=2)
        self.upsample_1024_1024 = Upsample(38)
        self.conv_1024_128 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1)

        # conv4_3 到FC7  扩张卷积，尺度少一半
        self.DilationConv_512_256 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=2, dilation=2,
                                              stride=2)
        # FC7到FC7 尺度不变
        self.conv_1024_512 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
        # conv8_2 到 FC7    反卷积上采样，尺度大一倍  10->19
        self.upsample_512_512 = Upsample(19)
        self.conv_512_256_fc7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)

        # conv5_3到conv8_2,扩张卷积，尺度少一半
        self.DilationConv_512_128_2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=2, dilation=2,
                                                stride=2)
        # conv8_2到conv8_2 尺度不变
        self.conv_512_256_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        # conv9_2到conv8_2
        self.upsample_256_256_2 = Upsample(10)
        self.conv_256_128_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)

        # conv9_2到conv9_2 尺度不变
        self.conv_256_128_9_2_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.upsample_256_256_10_2 = Upsample(5)
        self.conv_256_128_9_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1,stride=1)

        #conv10_2到conv10_2 尺度不变
        # self.conv_256_128_10_2_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        # self.upsample_256_256_11_2 = Upsample(3)
        # self.conv_256_128_10_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)

        # 平滑层
        # self.smooth = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)       
        # self.smooth2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=1)
        # self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)

        # 通道数BN层的参数是输出通道数out_channels
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(256)

        # CBAM模块【6个特征层：512 512 512 256 256 256 】
        # self.CBAM1 = Bottleneck(512)
        # self.CBAM2 = Bottleneck(512)
        # self.CBAM3 = Bottleneck(512)
        # self.CBAM4 = Bottleneck(256)
        # self.CBAM5 = Bottleneck(256)
        # self.CBAM6 = Bottleneck(256)
        
        # SE模块【6个特征层：512 512 512 256 256 256 】
        self.SE1 = SEModule(512)
        self.SE2 = SEModule(1024)
        self.SE3 = SEModule(512)
        self.SE4 = SEModule(256)
        self.SE5 = SEModule(256)
        self.SE6 = SEModule(256) 
        
        # ECA模块【6个特征层：512 512 512 256 256 256 】   
        # self.ECA1 = ECAModule(512)
        # self.ECA2 = ECAModule(512)
        # self.ECA3 = ECAModule(512)
        # self.ECA4 = ECAModule(256)
        # self.ECA5 = ECAModule(256)
        # self.ECA6 = ECAModule(256) 


    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        
        # 原论文中vgg的conv4_3，relu之后加入L2 Normalization正则化，然后保存feature map
        # apply vgg up to conv4_3 relu
        # 将vgg层的feature map保存
        # k的范围为0-22
        # =========开始保存 所需的所有中间信息

        # 保存pool2（pool下标从1开始）的结果
        # 经过maxpool，所以不需要L2Norm正则化
        for k in range(10):
            x = self.vgg[k](x)
        sources.append(x)


        # apply vgg up to conv4_3 relu
        for k in range(10, 23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to conv5_3 relu
        for k in range(23, 30):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        #apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        
        # 此时sources保存了所有中间结果，论文中的pool2、conv4_3、conv5_3、fc7、conv8_2、conv9_2、conv10_2、conv11_2
        # sources_final保存各层融合之后的最终结果
        sources_final = list()
        # con4_3层融合结果  self.bn1(self.conv1(x)) 在通道维度上融合
        # conv4_fp = torch.cat((F.relu(self.bn(self.DilationConv_128_128(sources[0])), inplace=True),
        #                       F.relu(self.conv_512_256(sources[1]), inplace=True),
        #                       F.relu(self.DeConv_1024_128(sources[3]), inplace=True)), 1) 

        conv4_fp = torch.cat((F.relu(self.bn(self.DilationConv_128_128(sources[0])), inplace=True),
                              F.relu(self.conv_512_256(sources[1]), inplace=True),
                              F.relu(self.conv_1024_128(self.upsample_1024_1024(sources[3])), inplace=True)), 1) 
        
        conv4_fp = F.relu(conv4_fp, inplace=True)
        if USE_SE:
            sources_final.append(self.SE1(conv4_fp))
        else:
            sources_final.append(conv4_fp)
        # FC7层融合结果
        # fc7_fp = torch.cat((F.relu(self.bn(self.DilationConv_512_128(sources[1])), inplace=True),
        #                     F.relu(self.conv_1024_256(sources[3]), inplace=True),
        #                     F.relu(self.DeConv_512_128(sources[4]), inplace=True)), 1)
        
        fc7_fp = torch.cat((F.relu(self.bn1(self.DilationConv_512_256(sources[1])), inplace=True),
                            F.relu(self.conv_1024_512(sources[3]), inplace=True),
                            F.relu(self.conv_512_256_fc7(self.upsample_512_512(sources[4])), inplace=True)), 1)
        
        # sources_final.append(F.relu(self.smooth(fc7_fp) , inplace=True))
        fc7_fp = F.relu(fc7_fp, inplace=True)
        if USE_SE:
            sources_final.append(self.SE2(fc7_fp))
        else:
            sources_final.append(fc7_fp)
        # conv8_2层融合结果
        # conv8_fp = torch.cat((F.relu(self.bn(self.DilationConv_512_128_2(sources[2])), inplace=True),
        #                       F.relu(self.conv_512_256_2(sources[4]), inplace=True),
        #                       F.relu(self.DeConv_256_128_2(sources[5]), inplace=True)), 1)
        
        conv8_fp = torch.cat((F.relu(self.bn(self.DilationConv_512_128_2(sources[2])), inplace=True),
                              F.relu(self.conv_512_256_2(sources[4]), inplace=True),
                              F.relu(self.conv_256_128_2(self.upsample_256_256_2(sources[5])), inplace=True)), 1)
        # sources_final.append(F.relu(self.smooth(conv8_fp) , inplace=True))
        conv8_fp = F.relu(conv8_fp, inplace=True)
        if USE_SE:
            sources_final.append(self.SE3(conv8_fp))
        else:
            sources_final.append(conv8_fp)

        #conv9_2层融合
        conv9_fp = torch.cat((F.relu(self.conv_256_128_9_2_1(sources[5]), inplace=True),
                              F.relu(self.conv_256_128_9_2(self.upsample_256_256_10_2(sources[6])), inplace=True)), 1)

        conv9_fp = F.relu(conv9_fp, inplace=True)
        
        if USE_SE:
            sources_final.append(self.SE4(conv9_fp))
        else:
            sources_final.append(conv9_fp)

        #conv10_2层融合
        # conv10_fp = torch.cat((F.relu(self.conv_256_128_10_2_1 (sources[6]), inplace=True),
        #                       F.relu(self.conv_256_128_10_2(self.upsample_256_256_11_2(sources[7])), inplace=True)), 1)
        # conv10_fp = F.relu(conv10_fp, inplace=True)

        if USE_SE:
            sources_final.append(self.SE5(sources[6]))
        # conv11_2
            sources_final.append(self.SE6(sources[7]))
        else:
            sources_final.append(sources[6])
        # conv11_2
            sources_final.append(sources[7])

        # for x in sources_final:
        #     print(x.shape)



        # apply multibox head to source layers
        # print(self.loc)
        for (x, l, c) in zip(sources_final, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            # output = self.detect(
            #     loc.view(loc.size(0), -1, 4),                  #loc preds
            #     self.softmax(conf.view(conf.size(0), -1,
            #                  self.num_classes)),               #conf preds
            #     self.priors.type(type(x.data))                 #default boxes
            # )
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        _, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    # 传入的修改过的vgg网络用于预测的网络是21层以及倒数第二层
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        # 按照fp-ssd论文，将1024改为512通道
        if k == 1:
            in_channels = 1024
        else:
            in_channels = vgg[v].out_channels

        loc_layers += [nn.Conv2d(in_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    # [x::y] 从下标x开始，每隔y取值
    # 论文中新增层也是每隔一个层添加一个预测层
    # 将新增的额外层中的预测层也添加上   start=2：下标起始位置
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=2):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
