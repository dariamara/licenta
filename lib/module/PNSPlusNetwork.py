import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.module.LightRFB import LightRFB
from lib.module.PNSPlusModule import NS_Block
from lib.module.KAN import KANBlock, PatchEmbed
#contributie
from lib.module.SwinTransformer import SwinBackbone
#contributie

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
        #contributie
        # Swin-B backbone: outputs low_feature (512ch, 1/16) and high_feature (1024ch, 1/32)
        self.feature_extractor = SwinBackbone(
            img_size=(224, 448),
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            window_size=7,
            drop_path_rate=0.2,
        )
        #contributie
        self.High_RFB = LightRFB(channels_in=1024)
        self.Low_RFB = LightRFB(channels_in=512, channels_mid=128, channels_out=24)

        self.High_drop = nn.Dropout2d(0.5)
        self.Low_drop = nn.Dropout2d(0.5)

        self.squeeze = nn.Sequential(nn.Conv2d(1024, 32, 1), nn.GroupNorm(2, 32), nn.Mish(inplace=True))
        self.decoder = conbine_feature()
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(16, 1, kernel_size=1, bias=False))
        self.NSB_global = NS_Block(bn_out=bn_out, channels_in=32, radius=[3, 3, 3, 3], dilation=[3, 4, 3, 4])
        self.NSB_local = NS_Block(bn_out=bn_out, channels_in=32, radius=[3, 3, 3, 3], dilation=[1, 2, 1, 2])
        self.up_sample_low = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        #contributie
        # Swin high_feature is at 1/32 scale; stride=2 brings it to 1/16 matching low_feature.
        # The previous stride=4 was for the old backbone which further downsampled via KAN patch_embed.
        self.up_sample_high = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        #contributie

        #contributie
        # KAN block 1: applied directly to Swin high_feature tokens (no extra patch_embed
        # downsampling needed because Swin already provides 1/32-scale tokens).
        self.block_h_1 = nn.ModuleList([KANBlock(dim=1024)])
        #contributie

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

        #contributie
        # Extract multi-scale features with Swin-B backbone.
        # low_feature:  (B, 512,  H/16, W/16) — e.g. (B, 512, 14, 28) for 224x448 input
        # high_feature: (B, 1024, H/32, W/32) — e.g. (B, 1024, 7, 14) for 224x448 input
        low_feature, high_feature = self.feature_extractor(x)
        #contributie

        #contributie
        if self.use_kan:
            # Flatten Swin tokens and apply KAN directly — no extra patch_embed step
            # because the Swin features are already at a compact 1/32 spatial scale.
            B_k, C_k, H_k, W_k = high_feature.shape   # (B, 1024, 7, 14)
            high_feature_tokens = high_feature.flatten(2).transpose(1, 2)  # (B, H*W, 1024)
            for blk in self.block_h_1:
                high_feature_tokens = blk(high_feature_tokens, H_k, W_k)
            high_feature_tokens = self.norm_h_1(high_feature_tokens)
            high_feature = high_feature_tokens.reshape(B_k, H_k, W_k, -1).permute(0, 3, 1, 2).contiguous()
            # (B, 1024, 7, 14) — spatial shape unchanged
        #contributie

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
            high_feature = nn.Mish()(F.interpolate(high_feature, size=(high_feature_H, high_feature_W), mode='bilinear', align_corners=False))
            #end daria

        to_slice = t_h.shape[0] - high_feature.shape[0]
        
        high_feature = high_feature + t_h[to_slice:]

        # Resize high-level feature to the same as low-level feature.
        high_feature = F.interpolate(high_feature, size=(low_feature.shape[-2], low_feature.shape[-1]),
                                     mode="bilinear",
                                     align_corners=False)

        # UNet-like decoder.
        out = self.decoder(low_feature.clone(), high_feature.clone())

        out = torch.sigmoid(
            F.interpolate(self.SegNIN(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear",
                          align_corners=False))

        return out


#contributie
if __name__ == "__main__":
    # 1 batch, 7 clips (anchor + 6 local), 3 channels, 224x448 (Swin-compatible size)
    a = torch.randn(1, 7, 3, 224, 448).cuda()
    mobile = PNSNet(bn_out=(14, 28), use_kan=False).cuda()
    print(mobile(a).shape)
#contributie
