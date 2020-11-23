'''
code by zzg 2020-11-12
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
from data.config import USE_ECA, USE_SE, USE_CBAM
from attention import Bottleneck, SEModule, ECAModule, Upsample


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
    

    # =====bobo新增==================
        # conv9_2到conv8_2
        self.upsample_256_256 = Upsample(10)
        self.conv_256_512 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1)
        #conv8_2到conv8_2
        self.conv_512_512_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)

        ##add后512-->1024 conv8_2到conv7 上采样，尺度大一倍
        self.upsample_512_512 = Upsample(19)
        self.conv_512_1024 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1)

        # conv7到conv7  尺度不变
        self.conv_1024_1024 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1)
        # conv7 到 conv4_3  上采样，尺度大一倍
        self.upsample_1024_1024 = Upsample(38)
        self.conv_1024_512 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
     
        # conv4_3 到con4_3 尺度不变
        self.conv_512_512_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1,  stride=1)
        

        # 平滑层
        self.smooth = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.smooth1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=1)

        # 通道数BN层的参数是输出通道数out_channels
        # self.bn = nn.BatchNorm2d(128)

        # CBAM模块【6个特征层：512 512 512 256 256 256 】
        if USE_CBAM:
            self.CBAM1 = Bottleneck(512)
            self.CBAM2 = Bottleneck(1024)
            self.CBAM3 = Bottleneck(512)
            self.CBAM4 = Bottleneck(256)
            self.CBAM5 = Bottleneck(256)
            self.CBAM6 = Bottleneck(256)

        if USE_SE:
            self.SE1 = SEModule(512)
            self.SE2 = SEModule(1024)
            self.SE3 = SEModule(512)
            self.SE4 = SEModule(256)
            self.SE5 = SEModule(256)
            self.SE6 = SEModule(256) 


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
        attention = list()
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
        for k in range(10,23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to conv5_3 relu
        for k in range(23,30):
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

        if USE_SE:
            attention.append(sources[0])
            attention.append(self.SE1(sources[1]))
            attention.append(sources[2])
            attention.append(self.SE2(sources[3]))
            attention.append(self.SE3(sources[4]))
            attention.append(self.SE4(sources[5]))
            attention.append(self.SE5(sources[6]))
            attention.append(self.SE6(sources[7]))
       

        
        # 此时sources保存了所有中间结果，论文中的pool2、conv4_3、conv5_3、fc7、conv8_2、conv9_2、conv10_2、conv11_2
        # sources_final保存各层融合之后的最终结果
        sources_final = list()
        
   
        # conv8_2层融合结果  self.bn1(self.conv1(x)) 在通道维度上融合
        conv8_fp1 = self.conv_256_512(self.upsample_256_256(attention[5])) + self.conv_512_512_1(attention[4])

        conv8_fp = self.smooth(conv8_fp1)
  
        # conv7层融合结果
        fc7_fp1 =  self.conv_512_1024(self.upsample_512_512(conv8_fp1)) + self.conv_1024_1024(attention[3])
        fc7_fp = self.smooth1(fc7_fp1)
     
        # conv4_3层融合结果
        conv4_fp = self.conv_1024_512(self.upsample_1024_1024(fc7_fp1)) + self.conv_512_512_2(attention[1])
      
        conv4_fp = self.smooth(conv4_fp)


        if USE_CBAM:
            # print("use cbam")
            sources_final.append(self.CBAM1(conv4_fp))
            sources_final.append(self.CBAM2(fc7_fp))
            sources_final.append(self.CBAM3(conv8_fp))
            sources_final.append(self.CBAM4(sources[5]))
            sources_final.append(self.CBAM5(sources[6]))
            sources_final.append(self.CBAM6(sources[7]))
        # if USE_SE:
        #     sources_final.append(self.SE1(conv4_fp))
        #     sources_final.append(self.SE2(fc7_fp))
        #     sources_final.append(self.SE3(conv8_fp))
        #     sources_final.append(self.SE4(sources[5]))
        #     sources_final.append(self.SE5(sources[6]))
        #     sources_final.append(self.SE6(sources[7]))

        else:
            # print("no cbam")
            sources_final.append(conv4_fp)
            sources_final.append(fc7_fp)
            sources_final.append(conv8_fp)
            sources_final.append(attention[5])
            sources_final.append(attention[6])
            sources_final.append(attention[7])



        # apply multibox head to source layers
        for (x, l, c) in zip(sources_final, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),                  # loc preds
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
        other, ext = os.path.splitext(base_file)
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
    # 传入的修改过的vgg网络用于预测的网络是21层以及 倒数第二层
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
