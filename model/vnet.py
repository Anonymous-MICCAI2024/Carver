import enum
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAttention(nn.Module):
    def __init__(self, channel, depth=64):
        super(DepthAttention,self).__init__()
        reduction = channel
        self.avg_pool = nn.AdaptiveAvgPool3d((depth,1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel*depth,(channel*depth) // reduction),
            nn.ReLU(inplace=True),
            nn.Linear((channel*depth)// reduction,channel*depth),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,d,_,_ = x.size()
        y = self.avg_pool(x).view(b,c*d)
        y = self.fc(y).view(b,c,d,1,1)
        return x*y.expand_as(x)



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _,_ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)



class DaDoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depth=None, norm_layer=None):
        super(DaDoubleConv3D,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = norm_layer(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(out_channels)
        self.da = DepthAttention(out_channels,depth)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.da(out)
        out = self.relu(out)
        
        return out


class DaSeDoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depth=None, norm_layer=None):
        super(DaSeDoubleConv3D,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = norm_layer(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(out_channels)
        self.da = DepthAttention(out_channels,depth)
        self.se = SELayer(out_channels,reduction=16)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.da(out)
        out = self.se(out)
        out = self.relu(out)
        
        return out


class SeDoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depth=None, norm_layer=None):
        super(SeDoubleConv3D,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = norm_layer(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(out_channels)
        self.se = SELayer(out_channels,reduction=16)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = self.relu(out)
        
        return out
    
class   ResSeDoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depth=None, norm_layer=None):
        super(ResSeDoubleConv3D,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = norm_layer(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(out_channels)
        self.se = SELayer(out_channels,reduction=16)
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if residual.size() != out.size():
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out


class ResDaSeDoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depth=None, norm_layer=None):
        super(ResDaSeDoubleConv3D,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = norm_layer(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(out_channels)
        self.da = DepthAttention(out_channels,depth)
        self.se = SELayer(out_channels,reduction=16)
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.da(out)
        out = self.se(out)

        if residual.size() != out.size():
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out



class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depth=None, norm_layer=None):
        super(DoubleConv3D,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class ResDoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depth=None, norm_layer=None):
        super(ResDoubleConv3D,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = norm_layer(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(out_channels)
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if residual.size() != out.size():
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

#-------------------------------------------

class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder, depth, norm_layer=None):
        super(Down3D,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            conv_builder(in_channels, out_channels, depth=depth, norm_layer=norm_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


#-------------------------------------------

class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder, depth, trilinear=True, norm_layer=None):
        super(Up3D,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        # if bilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = conv_builder(in_channels, out_channels, in_channels // 2, depth=depth, norm_layer=norm_layer)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = conv_builder(in_channels, out_channels, depth=depth, norm_layer=norm_layer)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print(x1.size())
        # input is CDHW
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffD // 2, diffD - diffD // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffW // 2, diffW - diffW // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#-------------------------------------------

class Tail3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tail3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#-------------------------------------------

class VNet(nn.Module):
    def __init__(self, stem, down, up, tail, width, depth, conv_builder, norm_layer=None, n_blocks=4, in_channels=1, classes=2, trilinear=True,dropout_rate=0.2):
        super(VNet, self).__init__()

        assert len(width) == n_blocks + 1
        assert len(depth) == n_blocks + 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        self.n_blocks = n_blocks
        factor = 2 if trilinear else 1

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.inc = stem(in_channels, width[0], depth=depth[0])
        
        for i in range(n_blocks):
            if i < n_blocks - 1:
                self.down_blocks.append(down(width[i], width[i+1], conv_builder, depth=depth[i+1], norm_layer=norm_layer))
                self.up_blocks .append(up(width[n_blocks - i], width[n_blocks - i - 1] // factor, conv_builder, depth=depth[n_blocks - i - 1], trilinear=trilinear, norm_layer=norm_layer))
            else:
                self.down_blocks.append(down(width[i], width[i+1]//factor, conv_builder, depth=depth[i+1], norm_layer=norm_layer))
                self.up_blocks .append(up(width[n_blocks - i], width[n_blocks - i - 1], conv_builder, depth=depth[n_blocks - i - 1], trilinear=trilinear, norm_layer=norm_layer))

        self.dropout = nn.Dropout(p=0.2) if dropout_rate > 0. else nn.Identity()
        self.outc = tail(width[0], classes)

    def forward(self, x):
        x = self.inc(x)

        skip_out = [x]

        for i in range(self.n_blocks):
            x = self.down_blocks[i](x)
            if i < self.n_blocks - 1:
                skip_out.append(x)

        skip_out = skip_out[::-1]
        for i in range(self.n_blocks):
            x = self.up_blocks[i](x,skip_out[i])

        x = self.dropout(x)
        logits = self.outc(x)
        return logits



def vnet(init_depth=128,**kwargs):
    return VNet(stem=DoubleConv3D,
                down=Down3D,
                up=Up3D,
                tail=Tail3D,
                width=[32,64,128,256,512],
                depth=[init_depth,init_depth//2,init_depth//4,init_depth//8,init_depth//16],
                conv_builder=DoubleConv3D,
                norm_layer=nn.BatchNorm3d,
                n_blocks=4,
                **kwargs)

def vnet_res(init_depth=128,**kwargs):
    return VNet(stem=DoubleConv3D,
                down=Down3D,
                up=Up3D,
                tail=Tail3D,
                width=[32,64,128,256,512],
                depth=[init_depth,init_depth//2,init_depth//4,init_depth//8,init_depth//16],
                conv_builder=ResDoubleConv3D,
                norm_layer=nn.BatchNorm3d,
                n_blocks=4,
                **kwargs)

def vnet_lite(init_depth=128,**kwargs):
    return VNet(stem=DoubleConv3D,
                down=Down3D,
                up=Up3D,
                tail=Tail3D,
                width=[32,64,128,256],
                depth=[init_depth,init_depth//2,init_depth//4,init_depth//8],
                conv_builder=DoubleConv3D,
                n_blocks=3,
                **kwargs)



def vnet_da(init_depth=128,**kwargs):
    return VNet(stem=DoubleConv3D,
                down=Down3D,
                up=Up3D,
                tail=Tail3D,
                width=[32,64,128,256,512],
                depth=[init_depth,init_depth//2,init_depth//4,init_depth//8,init_depth//16],
                conv_builder=DaDoubleConv3D,
                **kwargs)

def vnet_se(init_depth=128,**kwargs):
    return VNet(stem=DoubleConv3D,
                down=Down3D,
                up=Up3D,
                tail=Tail3D,
                width=[32,64,128,256,512],
                depth=[init_depth,init_depth//2,init_depth//4,init_depth//8,init_depth//16],
                conv_builder=SeDoubleConv3D,
                **kwargs)

def vnet_da_se(init_depth=128,**kwargs):
    return VNet(stem=DoubleConv3D,
                down=Down3D,
                up=Up3D,
                tail=Tail3D,
                width=[32,64,128,256,512],
                depth=[init_depth,init_depth//2,init_depth//4,init_depth//8,init_depth//16],
                conv_builder=DaSeDoubleConv3D,
                **kwargs)

def vnet_res_se(init_depth=128,**kwargs):
    return VNet(stem=DoubleConv3D,
                down=Down3D,
                up=Up3D,
                tail=Tail3D,
                width=[32,64,128,256,512],
                depth=[init_depth,init_depth//2,init_depth//4,init_depth//8,init_depth//16],
                conv_builder=ResSeDoubleConv3D,
                **kwargs)

def vnet_res_da_se(init_depth=128,**kwargs):
    return VNet(stem=DoubleConv3D,
                down=Down3D,
                up=Up3D,
                tail=Tail3D,
                width=[32,64,128,256,512],
                depth=[init_depth,init_depth//2,init_depth//4,init_depth//8,init_depth//16],
                conv_builder=ResDaSeDoubleConv3D,
                **kwargs)