import torch
from torch import nn
from torch.nn import functional as F

from .gumblemodule import GumbelSoftmax2D


class HyperRegionConv(nn.Module):

    def __init__(self, in_channel, out_channel, region_num):
        super().__init__()
        self.RN = region_num
        self.in_c = in_channel
        self.out_c = out_channel
        self.mask_conv = nn.Conv2d(in_channels=in_channel, out_channels=region_num, kernel_size=3, padding=1, bias=True)

        self.GS = GumbelSoftmax2D(hard=False)

        self.out = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, x, kernel):
        b, c, h, w = x.size()
        masks = self.mask_conv(x)
        if self.training:
            masks = self.GS(masks, gumbel=True)
        else:
            masks = self.GS(masks, gumbel=False)

        x = x.unsqueeze(1).repeat(1, self.RN, 1, 1, 1)  # for region
        x = x.view(1, -1, h, w)  # for adaptivity group region
        kernel = kernel.reshape(-1, self.in_c, 3, 3)
        x = F.conv2d(x, kernel, stride=1, padding=1, groups=b * self.RN).view(b, self.RN, self.out_c, h, w)
        x = torch.sum(x * masks.unsqueeze(2), dim=1)
        return self.out(x), masks


class QueryGeneration_spacial(nn.Module):
    def __init__(self, in_channel, query_num, query_base, query_channel=256):
        super(QueryGeneration_spacial, self).__init__()

        self.query_num = query_num
        self.query_base = query_base
        self.query_channel = query_channel - int(query_channel * query_base)

        self.base_features = nn.Embedding(self.query_num, self.query_channel)

        if not self.query_base == 0:
            self.base_fea_cat = nn.Embedding(self.query_num, int(query_channel * query_base))

        self.q_g = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=self.query_channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.query_channel),
            nn.ReLU(inplace=True)
        )

        self.k_g = nn.Sequential(
            nn.Linear(self.query_channel, self.query_channel),
            nn.BatchNorm1d(self.query_channel),
            nn.ReLU(inplace=True)
        )

        self.out_g = nn.Sequential(
            nn.Linear(self.query_channel, self.query_channel),
            nn.BatchNorm1d(self.query_channel),
            nn.ReLU(inplace=True)
        )

        self.spacial_fc = nn.Linear(64, self.query_num)

        # self.linear_v

    def forward(self, x):
        b = x.size(0)
        x = self.q_g(x)
        x = x.flatten(2)
        x = self.spacial_fc(x)

        k = self.k_g(self.base_features.weight)

        weight = torch.einsum('bcn, mc -> bmn', x, k)
        weight = F.softmax(weight, -1)
        query = torch.einsum('bmn, nc -> bmc', weight, self.base_features.weight)

        query = self.out_g(query.view(-1, self.query_channel)).view(b, self.query_num, self.query_channel)
        query = query + self.base_features.weight.unsqueeze(0).repeat(b, 1, 1)
        if not self.query_base == 0:
            query = torch.cat([query, self.base_fea_cat.weight.unsqueeze(0).repeat(b, 1, 1)], dim=-1)
        return query


class QueryGeneration_channel(nn.Module):
    def __init__(self, in_channel, query_num, query_base, query_channel=256):
        super(QueryGeneration_channel, self).__init__()

        self.query_num = query_num
        self.query_base = query_base
        self.query_channel = query_channel - int(query_channel * query_base)

        self.base_features = nn.Embedding(self.query_num, self.query_channel)
        if not self.query_base == 0:
            self.base_fea_cat = nn.Embedding(self.query_num, int(query_channel * query_base))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.q_g = nn.Sequential(
            nn.Linear(in_channel, self.query_channel * self.query_num, bias=True),
            nn.BatchNorm1d(self.query_channel * self.query_num),
            nn.ReLU(inplace=True)
        )

        self.k_g = nn.Sequential(
            nn.Linear(self.query_channel, self.query_channel),
            nn.BatchNorm1d(self.query_channel),
            nn.ReLU(inplace=True)
        )

        self.out_g = nn.Sequential(
            nn.Linear(self.query_channel, self.query_channel),
            nn.BatchNorm1d(self.query_channel),
            nn.ReLU(inplace=True)
        )

        # self.linear_v

    def forward(self, x):
        b = x.size(0)
        x = self.pool(x).view(b, -1)
        x = self.q_g(x).view(b, self.query_num, self.query_channel)

        k = self.k_g(self.base_features.weight)

        weight = torch.einsum('bnc, mc -> bnm', x, k)
        weight = F.softmax(weight, -1)
        query = torch.einsum('bnm, mc -> bnc', weight, self.base_features.weight)

        query = self.out_g(query.view(-1, self.query_channel)).view(b, self.query_num, self.query_channel)
        query = query + self.base_features.weight.unsqueeze(0).repeat(b, 1, 1)
        if not self.query_base == 0:
            query = torch.cat([query, self.base_fea_cat.weight.unsqueeze(0).repeat(b, 1, 1)], dim=-1)
        return query


class QueryGenerationModule(nn.Module):
    def __init__(self, query_type, in_channel, query_num, query_base, query_channel=256):
        super(QueryGenerationModule, self).__init__()

        self.query_num = query_num
        self.query_base = query_base
        if self.query_base == 1:
            self.base_features = nn.Embedding(self.query_num, query_channel)
        else:
            if query_type in ['S', 's']:
                self.queryG = QueryGeneration_spacial(in_channel, query_num, query_base, query_channel)
            elif query_type in ['C', 'c']:
                self.queryG = QueryGeneration_channel(in_channel, query_num, query_base, query_channel)

    def forward(self, x):
        b = x.size(0)
        if self.query_base == 1:
            return self.base_features.weight.unsqueeze(0).repeat(b, 1, 1)
        else:
            return self.queryG(x)
