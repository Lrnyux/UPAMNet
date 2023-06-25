import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


class APReLU(nn.Module):
    def __init__(self, in_channels):
        super(APReLU, self).__init__()
        self.in_channels=in_channels
        self.gap_min_branch=nn.AdaptiveAvgPool2d(1)
        self.gap_max_branch=nn.AdaptiveAvgPool2d(1)
        # self.bn_squeeze=nn.BatchNorm2d(self.in_channels)
        # self.bn_excitation=nn.BatchNorm2d(self.in_channels)
        self.fc_squeeze=nn.Linear(self.in_channels*2,self.in_channels)
        self.fc_excitation=nn.Linear(self.in_channels,self.in_channels)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_squeeze.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.fc_excitation.weight, mode='fan_in')

    def forward(self, x):
        N, C, H, W = x.size()
        x_min = (x-x.abs())*0.5
        x_max = F.relu(x)
        x_min_gap = self.gap_min_branch(x_min)
        x_max_gap = self.gap_max_branch(x_max)
        x_concat = torch.cat((x_min_gap,x_max_gap),dim=1).view(N,C*2)
        x_squeeze = self.fc_squeeze(x_concat).view(N,C,1,1)
        # x_squeeze = self.bn_squeeze(x_squeeze)
        x_squeeze = F.relu(x_squeeze).view(N,C)
        x_excitation = self.fc_excitation(x_squeeze).view(N,C,1,1)
        # x_excitation = self.bn_excitation(x_excitation)
        sigma = self.sigmoid(x_excitation)
        output = F.relu(x)+0.5*sigma.expand_as(x)*(x-x.abs())
        return output


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Dilated_Conv2d(nn.Module):
    def __init__(self, dim, kernel, padding, index_u_mask, index_v_mask):
        super(Dilated_Conv2d, self).__init__()
        weight = torch.ones(dim,dim,kernel, kernel)
        self.dim = dim
        self.weight = torch.nn.Parameter(weight, requires_grad=True)
        self.idx_u_mk = index_u_mask
        self.idx_v_mk = index_v_mask
        self.padding = padding

    def forward(self, x):
        self.weight.requires_grad = False
        self.weight[:,:,self.idx_u_mk,self.idx_v_mk]=0.0
        self.weight.requires_grad = True
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        return F.conv2d(x, self.weight)





# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.0, norm_groups=32):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input




class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, norm_groups=32, dropout=0.0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x):
        x = self.res_block(x)
        if (self.with_attn):
            # print('========')
            # print(x.shape)
            x = self.attn(x)
            # print(x.shape)
            # print('========')
        return x

class OrientationAttention(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim_in = dim
        self.dim_out = dim
        self.conv_in_1 = nn.Conv2d(dim,dim,3,padding=1)
        # self.conv_in_2 = Dilated_ConvVH(kernel_size=5,padding=2)
        # self.conv_in_3 = Dilated_ConvD(kernel_size=5,padding=2)
        self.conv_in_2 = Dilated_Conv2d(dim=dim, kernel=5, padding=2, index_u_mask=[0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4],
                       index_v_mask=[0, 1, 3, 4, 0, 1, 3, 4, 0, 1, 3, 4, 0, 1, 3, 4])
        self.conv_in_3 = Dilated_Conv2d(dim=dim,kernel=5,padding=2,index_u_mask=[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                            index_v_mask=[1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 0, 2, 4, 1, 2, 3])

        self.conv_weight_1 = nn.Conv2d(dim*3, dim*3, 1)
        self.conv_weight_2 = nn.Conv2d(dim*3, dim*3, 1)

        self.sigmoid = nn.Sigmoid()
        self.conv_out = nn.Conv2d(3*dim,dim,3,padding=1)


    def forward(self,x):
        x_conv_1 = self.conv_in_1(x)
        x_conv_2 = self.conv_in_2(x)
        x_conv_3 = self.conv_in_3(x)
        x_cat = torch.cat([x_conv_1,x_conv_2,x_conv_3],dim=1)

        x_ptx = F.adaptive_avg_pool2d(x_cat,(1,1))
        x_ptx = self.sigmoid(self.conv_weight_2(F.relu(self.conv_weight_1(x_ptx))))

        x_cat_attn = x_cat * x_ptx
        x_out = self.conv_out(F.relu(x_cat_attn))
        return x + x_out


class PositionAttention(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim_in = dim
        self.dim_out = dim
        self.conv_in = nn.Conv2d(dim, dim, 3, padding=1)
        self.aprulu = APReLU(in_channels=dim)
        self.conv_out = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out = self.conv_in(x)
        out = self.aprulu(out)
        out = self.conv_out(out)
        out = self.sigmoid(out)
        return x * out



class UPAMNet(nn.Module):
    def __init__(
            self,
            in_channel=1,
            out_channel=1,
            inner_channel=16,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8),
            attn_res=[16,32,64,128],
            res_blocks=1,
            dropout=0.1,
            image_size=256
    ):
        super().__init__()

        kernel_size = 3
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]


        self.res_blocks = res_blocks

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for res_idx in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(OrientationAttention(dim=pre_channel))
                downs.append(Downsample(pre_channel))


                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:

                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        ups.append(PositionAttention(pre_channel))
        self.ups = nn.ModuleList(ups)
        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)




    def forward(self, x):
        feats = []
        for idx in range(len(self.downs)):
            layer = self.downs[idx]
            if isinstance(layer,OrientationAttention):
                x = layer(x)
            else :
                x = layer(x)
                feats.append(x)

        for layer in self.mid:
            x = layer(x)

        for layer in self.ups:
            # print(x.shape)
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1))
            else:
                x = layer(x)

        return self.final_conv(x)

if __name__ == '__main__':
    import time

    import argparse
    parser = argparse.ArgumentParser()

    # =========for hyper parameters===
    args = parser.parse_args()
    #
    net = UPAMNet(inner_channel=16, norm_groups=16, channel_mults=[1,2,4,8],
                    attn_res=[16,32,64], res_blocks=2)
    #
    net = net.cuda()
    print(net)

    input = torch.ones((1,1,256,256))

    input = input.cuda()


    t0 = time.time()
    output = net(input)
    print(time.time() - t0)
    print(output.shape)

    tal_num = sum(p.numel() for p in net.parameters())
    print(tal_num)


