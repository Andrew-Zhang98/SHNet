import torch
import torchvision.models as models
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from .ResNet import ResNet50
from .Vquery import HyperRegionConv as HRconv
from .Vquery import QueryGenerationModule as QueryGenerationModule
from .transformer.transformer_predictor import TransformerPredictor


class SaliencyFormer(nn.Module):
    def __init__(self):
        super(SaliencyFormer, self).__init__()

        self.query_num = 3

        self.base_C = 64

        self.resnet = ResNet50()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.T_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=self.base_C, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.base_C),
            nn.ReLU(inplace=True)
        )

        self.T_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=self.base_C, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.base_C),
            nn.ReLU(inplace=True)
        )

        self.T_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=self.base_C, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.base_C),
            nn.ReLU(inplace=True)
        )
        self.T_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=self.base_C, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.base_C),
            nn.ReLU(inplace=True)
        )

        self.up_conv4_1 = HRconv(self.base_C, self.base_C, self.query_num)
        self.c_conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_C * 2, out_channels=self.base_C, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.base_C),
            nn.ReLU(inplace=True)
        )

        self.up_conv4_2 = HRconv(self.base_C, self.base_C, self.query_num)
        self.c_conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_C * 2, out_channels=self.base_C, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.base_C),
            nn.ReLU(inplace=True)
        )

        self.up_deconv_5 = HRconv(self.base_C, self.base_C, self.query_num)

        self.c_deconv_layer_5_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_C * 2, out_channels=self.base_C, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.base_C),
            nn.ReLU(inplace=True)
        )

        self.up_deconv_6 = HRconv(self.base_C, self.base_C, self.query_num)

        self.c_deconv_layer_5_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_C + 64, out_channels=self.base_C, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.base_C),
            nn.ReLU(inplace=True),
            self.upsample2
        )

        self.up_deconv_7 = HRconv(self.base_C, self.base_C, self.query_num)

        self.predict_layer_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_C, out_channels=1, kernel_size=3, padding=1, bias=True)
        )
        self.predict_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_C, out_channels=1, kernel_size=3, padding=1, bias=True)
        )
        self.predict_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_C, out_channels=1, kernel_size=3, padding=1, bias=True)
        )
        self.predict_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_C, out_channels=1, kernel_size=3, padding=1, bias=True)
        )

        self.predict_layer_final = nn.Sequential(
            nn.Conv2d(in_channels=self.base_C, out_channels=1, kernel_size=3, padding=1, bias=True)
        )

        self.queryG = QueryGenerationModule('S', 2048, self.query_num, 1.0, query_channel=128)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        self.trans_pre = TransformerPredictor(in_channels=2048,
                                              hidden_dim=128,
                                              num_queries=self.query_num,
                                              nheads=8,
                                              dropout=0.1,
                                              dim_feedforward=2048,
                                              enc_layers=0,
                                              dec_layers=6,
                                              mask_dim=256,
                                              pre_norm=False,
                                              deep_supervision=True,
                                              enforce_input_project=True,
                                              base_c=self.base_C)

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        b = x.shape[0]
        # Backbone
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3_1 = self.resnet.layer3(x2)
        x4_1 = self.resnet.layer4(x3_1)

        # Decoder
        x1_t = self.T_layer1(x1)
        x2_t = self.T_layer2(x2)
        x3_1_t = self.T_layer3(x3_1)
        x4_1_t = self.T_layer4(x4_1)

        query = self.queryG(x4_1)
        signal_f = x4_1

        region_layer_kernel = self.trans_pre(signal_f, query)
        w4_1, w4_2, w5, w6, w7 = region_layer_kernel

        x4_1_t = self.upsample2(x4_1_t)
        x4_1_u_0, x4_1_u_0_mask = self.up_conv4_1(x4_1_t, w4_1)

        c4_1_u = torch.cat((x4_1_u_0, x3_1_t), dim=1)
        x4_1_u_1 = self.c_conv4_1(c4_1_u)

        x4_1_u_1 = self.upsample2(x4_1_u_1)
        x4_1_u_2, x4_1_u_2_mask = self.up_conv4_2(x4_1_u_1, w4_2)

        c4_2_u = torch.cat((x4_1_u_2, x2_t), dim=1)
        x4_1_u = self.c_conv4_2(c4_2_u)

        h5_f, h_5_f_mask = self.up_deconv_5(x4_1_u, w5)
        h5_f = self.upsample2(h5_f)

        h_5_2c = torch.cat((h5_f, x1_t), 1)

        h_5_2f = self.c_deconv_layer_5_2(h_5_2c)

        h_5_2u, h_5_2u_mask = self.up_deconv_6(h_5_2f, w6)

        h_5_1c = torch.cat((h_5_2u, x), 1)
        h_5_1c = self.c_deconv_layer_5_1(h_5_1c)

        h_5_1f, h_5_1f_mask = self.up_deconv_7(h_5_1c, w7)
        h_5_1f = self.upsample2(h_5_1f)

        pred_mask_0 = self.upsample8(self.predict_layer_0(x4_1_u_1))
        pred_mask_1 = self.upsample8(self.predict_layer_1(x4_1_u))
        pred_mask_2 = self.upsample4(self.predict_layer_2(h_5_2f))
        pred_mask_3 = self.upsample2(self.predict_layer_3(h_5_1c))

        pred_mask = self.predict_layer_final(h_5_1f)

        masks = [x4_1_u_0_mask, x4_1_u_2_mask, h_5_f_mask, h_5_2u_mask, h_5_1f_mask]

        pred_contour, contours = None, None

        return [pred_mask, masks], [pred_contour, contours], [pred_mask_0, pred_mask_1, pred_mask_2, pred_mask_3], None

        # import pdb; pdb.set_trace()

    def hard_softmax(self, logits):
        y = F.softmax(logits, 1)
        _, max_value_indexes = y.data.max(1, keepdim=True)
        y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)
        y = Variable(y_hard - y.data) + y

        return y

    def group_conv(self, x, kernel, in_c):
        b, c, h, w = x.size()
        kernel = kernel.reshape(-1, in_c, 3, 3)
        x = x.view(1, -1, h, w)
        x = F.conv2d(x, kernel, stride=1, padding=1, groups=b).view(b, 64, h, w)
        return x

    def group_region_conv(self, x, kernel, in_c):
        b, c, h, w = x.size()
        kernel = kernel.reshape(-1, in_c, 3, 3)
        x = x.view(1, -1, h, w)
        x = F.conv2d(x, kernel, stride=1, padding=1, groups=b).view(b, 64, h, w)
        return x

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()

        pretrained_dict = torch.load("/4T/wenhu/IJCV/resnet50_ibn_a-d9d0bb7b.pth")
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif 'num_batches_tracked' in k:
                all_params[k] = torch.tensor([0.0])
            else:
                print("missing", k)
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        print("load pretrain resnet50")
        self.resnet.load_state_dict(all_params)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
