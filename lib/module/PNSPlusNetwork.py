import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# contributie start
import torch.utils.checkpoint as checkpoint
# contributie end

from lib.module.LightRFB import LightRFB
from lib.module.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.module.PNSPlusModule import NS_Block
#from lib.module.ConvNeXt import convnext_tiny, convnext_base, convnext_small
from lib.module.MedNeXt import create_mednext_encoder
from lib.module.KAN import KANBlock, PatchEmbed

# contributie start
# Shared across DataParallel replicas since it's a module-level global, not an
# instance attribute (which would get reset on every replicate() call).
NAN_STATS = {'frames': 0, 'pixels': 0}
# contributie end


class conbine_feature(nn.Module):
    def __init__(self):
        super(conbine_feature, self).__init__()
        self.up2_high = DilatedParallelConvBlockD2(32, 16)
        self.up2_low = nn.Conv2d(24, 16, 1, stride=1, padding=0, bias=False)
        self.up2_bn2 = nn.GroupNorm(2, 16)
        self.up2_act = nn.Mish()
        self.refine = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, bias=False), nn.GroupNorm(2, 16), nn.Mish())

    def forward(self, low_fea, high_fea):
        high_fea = self.up2_high(high_fea)
        low_fea = self.up2_bn2(self.up2_low(low_fea))
        refine_feature = self.refine(self.up2_act(high_fea + low_fea))
        return refine_feature


class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, nIn, nOut, add=False):
        super(DilatedParallelConvBlockD2, self).__init__()
        n = int(np.ceil(nOut / 2.))
        n2 = nOut - n

        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(n2, n2, 3, stride=1, padding=2, dilation=2, bias=False)

        self.bn = nn.GroupNorm(nOut // 16, nOut)
        self.add = add

    def forward(self, input):
        in0 = self.conv0(input)
        in1, in2 = torch.chunk(in0, 2, dim=1)
        b1 = self.conv1(in1)
        b2 = self.conv2(in2)
        output = torch.cat([b1, b2], dim=1)

        if self.add:
            output = input + output
        output = self.bn(output)

        return output


class PNSNet(nn.Module):
    def __init__(self, bn_out, use_kan):
        super(PNSNet, self).__init__()
        # contributie start
        self.feature_extractor = create_mednext_encoder(
            'B', in_channels=3, n_channels=64, kernel_size=3, dim='2d'
        )
        # contributie end
        # self.feature_extractor = convnext_base(pretrained=True, in_22k=True,  num_classes=21841, drop_path_rate=0.2)

        self.High_RFB = LightRFB(channels_in=1024)
        self.Low_RFB = LightRFB(channels_in=512, channels_mid=128, channels_out=24)

        self.High_drop = nn.Dropout2d(0.5)
        self.Low_drop = nn.Dropout2d(0.5)

        # self.squeeze = nn.Sequential(nn.Conv2d(1024, 32, 1), nn.GroupNorm(2, 32), nn.Mish(inplace=True))
        self.decoder = conbine_feature()
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(16, 1, kernel_size=1, bias=False))
        self.NSB_global = NS_Block(bn_out=bn_out, channels_in=32, radius=[3, 3, 3, 3], dilation=[3, 4, 3, 4])
        self.NSB_local = NS_Block(bn_out=bn_out, channels_in=32, radius=[3, 3, 3, 3], dilation=[1, 2, 1, 2])
        self.up_sample_low = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up_sample_high = nn.ConvTranspose2d(1024, 1024, kernel_size=4 if use_kan else 2, stride=4 if use_kan else 2)

        self.patch_embed_h_1 = PatchEmbed(img_size=256 // 8, patch_size=3, stride=2, in_chans=1024, embed_dim=1024)
        
        self.block_h_1 = nn.ModuleList([KANBlock(
            dim=1024
            )])

        self.block_h_2 = nn.ModuleList([KANBlock(
            dim=32
            )])
        
        self.norm_h_1 = nn.LayerNorm(1024)
        
        self.norm_h_2 = nn.LayerNorm(32)

        self.use_kan = use_kan

    def forward(self, x):

        origin_shape = x.shape
        x = x.view(-1, *origin_shape[2:])

        B = x.shape[0]

        #print(x)
        #print(x.shape)

        # x = self.feature_extractor.downsample_layers[0](x)
        # x = self.feature_extractor.stages[0](x)

        # x = self.feature_extractor.downsample_layers[1](x)
        # x = self.feature_extractor.stages[1](x)

        # Extract anchor, low-level, and high-level features.
        # low_feature = self.feature_extractor.downsample_layers[2](x)
        # low_feature = self.feature_extractor.stages[2](low_feature)

        # high_feature = self.feature_extractor.downsample_layers[3](low_feature)
        # high_feature = self.feature_extractor.stages[3](high_feature)

        # contributie start
        x = self.feature_extractor.stem(x)
        x = self.feature_extractor.iterative_checkpoint(self.feature_extractor.enc_block_0, x)
        x = self.feature_extractor.down_0(x)

        x = self.feature_extractor.iterative_checkpoint(self.feature_extractor.enc_block_1, x)
        x = self.feature_extractor.down_1(x)

        x = self.feature_extractor.enc_block_2(x)
        x = self.feature_extractor.down_2(x)

        low_feature = self.feature_extractor.enc_block_3(x)

        high_feature = self.feature_extractor.down_3(low_feature)
        high_feature = self.feature_extractor.bottleneck(high_feature)
        # contributie end

        if self.use_kan:
            high_feature, H, W = self.patch_embed_h_1(high_feature)
            for i, blk in enumerate(self.block_h_1):
                # high_feature = blk(high_feature, H, W)
                # contributie start
                high_feature = checkpoint.checkpoint(blk, high_feature, H, W)
            #     contrbutie end
            high_feature = self.norm_h_1(high_feature)
            high_feature = high_feature.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        high_feature = self.up_sample_high(high_feature)

        high_feature = self.High_RFB(high_feature)

        high_feature_H, high_feature_W = high_feature.shape[2:4]

        t_h = high_feature

        low_feature = self.up_sample_low(low_feature)


        # Reduce the channel dimension.
        low_feature = self.Low_RFB(low_feature)

        # Reshape into temporal formation.
        high_feature = high_feature.view(*origin_shape[:2], *high_feature.shape[1:])
        low_feature = low_feature.view(*origin_shape[:2], *low_feature.shape[1:])


        # Feature Separation.
        high_feature_global = high_feature[:, 0, ...].unsqueeze(dim=1).repeat(1, 6, 1, 1, 1)
        high_feature_local = high_feature[:, 1:7, ...]
        low_feature = low_feature[:, 1:7, ...]


        # First NS Block.
        high_feature_1 = self.NSB_global(high_feature_global, high_feature_local) + high_feature_local
        # Second NS Block.
        high_feature_2 = self.NSB_local(high_feature_1, high_feature_1) + high_feature_1


        # Residual Connection.
        high_feature = high_feature_2 + high_feature_local


        # Reshape back into spatial formation.
        high_feature = high_feature.contiguous().view(-1, *high_feature.shape[2:])
        low_feature = low_feature.contiguous().view(-1, *low_feature.shape[2:])

        if self.use_kan:
            B, _, H, W = high_feature.shape
            high_feature = high_feature.flatten(2).transpose(1,2)
            for i, blk in enumerate(self.block_h_2):
                high_feature = blk(high_feature, H, W)
            high_feature = self.norm_h_2(high_feature)
            high_feature = high_feature.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            #daria
            #la train am primit warning ca val default a lui align_corners s-a schimbat din false in true => l-am pus false explicit
            high_feature = nn.Mish()(F.interpolate(high_feature.float(), size=(high_feature_H, high_feature_W), mode='bilinear', align_corners=False)).to(high_feature.dtype)
            #end daria

        to_slice = t_h.shape[0] - high_feature.shape[0]
        
        high_feature = high_feature + t_h[to_slice:]

        # Resize high-level feature to the same as low-level feature.
        high_feature = F.interpolate(high_feature.float(), size=(low_feature.shape[-2], low_feature.shape[-1]),
                                     mode="bilinear",
                                     align_corners=False)

        # UNet-like decoder.
        out = self.decoder(low_feature.clone(), high_feature.clone())

        out = torch.sigmoid(
            F.interpolate(self.SegNIN(out).float(), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear",
                          align_corners=False))

        #contributie
        # nan_to_num must come before clamp: clamp(NaN) = NaN in PyTorch, so NaN would still reach BCE
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            NAN_STATS['frames'] += 1
            NAN_STATS['pixels'] += nan_mask.sum().item()
        out = torch.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
        out = out.clamp(min=1e-7, max=1 - 1e-7)
        #contributie

        return out


if __name__ == "__main__":
    a = torch.randn(1, 6, 3, 256, 448).cuda()
    mobile = PNSNet().cuda()
    print(mobile(a).shape)
