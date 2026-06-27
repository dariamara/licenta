import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.module.PNSPlusModule import NS_Block
from lib.module.KAN import KANBlock, D_ConvLayer
#contributie
from lib.module.SwinTransformer import SwinBackbone
#contributie


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


#contributie
class PNSNet(nn.Module):
    """Swin-UKAN: Swin Transformer backbone with U-KAN bottleneck and
    multi-scale skip-connection decoder.

    Architecture (for 224×448 input):
      Swin backbone  →  feat0 (128, 56×112)
                        feat1 (256, 28×56)
                        feat2 (512, 14×28)   [low-level Swin output]
                        feat3 (1024, 7×14)   [high-level Swin output]

      U-KAN KAN bottleneck on feat3  →  feat3_kan (1024, 7×14)

      Temporal NS_Block pathway (squeezed to 32ch, spatial 14×28):
          feat3_kan → upsample → squeeze(1024→32) → NS_Block →
          temporal_refined (B*6, 32, 14×28)

      Expand temporal back to 512ch for decoder entry:
          expand_temporal(32→512) → dec (512, 14×28)

      U-KAN decoder with skip connections:
          dec + feat2  →  dblock1 KAN  →  D_ConvLayer(512→256)  →  upsample
          + feat1      →  dblock2 KAN  →  D_ConvLayer(256→128)  →  upsample
          + feat0      →                  D_ConvLayer(128→64)   →  upsample
                                          D_ConvLayer(64→32)    →  upsample
                                          Conv2d(32→1)          →  sigmoid
    """

    def __init__(self, bn_out, use_kan=True, no_kan=False):
        super(PNSNet, self).__init__()

        # --- Swin-B backbone (now returns 4 spatial feature maps) ---
        self.feature_extractor = SwinBackbone(
            img_size=(224, 448),
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            window_size=7,
            drop_path_rate=0.2,
        )

        # --- U-KAN KAN bottleneck (applied to feat3 tokens) ---
        self.block1 = nn.ModuleList([KANBlock(dim=1024, no_kan=no_kan)])
        self.norm3 = nn.LayerNorm(1024)

        # --- Temporal processing pathway (existing NS_Block logic, unchanged) ---
        # Upsample feat3_kan from 7×14 to 14×28 before temporal squeeze
        self.up_sample_high = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.High_drop = nn.Dropout2d(0.5)
        # Squeeze 1024→32 for efficient temporal aggregation via NS_Block
        self.squeeze = nn.Sequential(
            nn.Conv2d(1024, 32, 1),
            nn.GroupNorm(2, 32),
            nn.Mish(inplace=True),
        )
        self.NSB_global = NS_Block(bn_out=bn_out, channels_in=32,
                                   radius=[3, 3, 3, 3], dilation=[3, 4, 3, 4])
        self.NSB_local  = NS_Block(bn_out=bn_out, channels_in=32,
                                   radius=[3, 3, 3, 3], dilation=[1, 2, 1, 2])

        # Expand temporal refined features (32ch) back to 512ch to match feat2 channels
        # so the U-KAN skip-connection addition (torch.add) is channel-compatible.
        self.expand_temporal = nn.Conv2d(32, 512, 1)

        # --- U-KAN decoder KAN blocks ---
        # dblock1: applied at 14×28 after adding feat2 skip (512ch)
        self.dblock1 = nn.ModuleList([KANBlock(dim=512, no_kan=no_kan)])
        self.dnorm1 = nn.LayerNorm(512)
        # dblock2: applied at 28×56 after adding feat1 skip (256ch)
        self.dblock2 = nn.ModuleList([KANBlock(dim=256, no_kan=no_kan)])
        self.dnorm2 = nn.LayerNorm(256)

        # --- U-KAN decoder conv stages ---
        # Each D_ConvLayer reduces channels; F.interpolate ×2 follows each stage.
        self.decoder1 = D_ConvLayer(512, 256)   # 14×28  → 28×56
        self.decoder2 = D_ConvLayer(256, 128)   # 28×56  → 56×112
        self.decoder3 = D_ConvLayer(128, 64)    # 56×112 → 112×224
        self.decoder4 = D_ConvLayer(64, 32)     # 112×224→ 224×448

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

        self.use_kan = use_kan

    def forward(self, x):
        origin_shape = x.shape               # (B, T, 3, H, W)
        x = x.view(-1, *origin_shape[2:])    # (B*T, 3, H, W)
        BT = x.shape[0]
        B  = origin_shape[0]

        # --- Swin backbone: 4 feature maps ---
        # feat0: (BT, 128,  56, 112)
        # feat1: (BT, 256,  28,  56)
        # feat2: (BT, 512,  14,  28)
        # feat3: (BT, 1024,  7,  14)
        feat0, feat1, feat2, feat3 = self.feature_extractor(x)

        # --- U-KAN KAN bottleneck on feat3 ---
        if self.use_kan:
            B_k, C_k, H_k, W_k = feat3.shape          # (BT, 1024, 7, 14)
            feat3_tokens = feat3.flatten(2).transpose(1, 2)   # (BT, 98, 1024)
            for blk in self.block1:
                feat3_tokens = blk(feat3_tokens, H_k, W_k)
            feat3_tokens = self.norm3(feat3_tokens)
            feat3_kan = feat3_tokens.reshape(B_k, H_k, W_k, C_k).permute(0, 3, 1, 2).contiguous()
            # (BT, 1024, 7, 14)
        else:
            feat3_kan = feat3

        # --- Temporal NS_Block pathway ---
        # Upsample to 14×28 to match bn_out=(14,28) expected by NS_Block
        high_f = self.up_sample_high(feat3_kan)   # (BT, 1024, 14, 28)
        high_f = self.High_drop(high_f)
        high_f = self.squeeze(high_f)              # (BT, 32, 14, 28)

        high_f = high_f.view(B, origin_shape[1], *high_f.shape[1:])  # (B, T, 32, 14, 28)
        t_h = high_f                               # save full-T tensor for residual

        # Split into global (anchor) and local (clip) frames
        high_global = high_f[:, 0, ...].unsqueeze(1).repeat(1, origin_shape[1] - 1, 1, 1, 1)
        high_local  = high_f[:, 1:, ...]

        high_1 = self.NSB_global(high_global, high_local) + high_local
        high_2 = self.NSB_local(high_1, high_1) + high_1
        high_temporal = (high_2 + high_local).contiguous().view(-1, *high_2.shape[2:])
        # (B*6, 32, 14, 28)

        # Residual from pre-temporal path (aligned to B*6 frames)
        to_slice = BT - high_temporal.shape[0]     # removes the anchor frame
        t_h_flat = t_h.contiguous().view(-1, *t_h.shape[2:])
        high_temporal = high_temporal + t_h_flat[to_slice:]

        # --- U-KAN decoder ---
        # Expand temporal (32ch) back to 512ch to be added with feat2 (512ch)
        dec = self.expand_temporal(high_temporal)  # (B*6, 512, 14, 28)

        # Decoder stage 1: skip from feat2 (512ch, 14×28)
        dec = torch.add(dec, feat2[to_slice:])     # (B*6, 512, 14, 28)
        B6, C, H, W = dec.shape
        dec_t = dec.flatten(2).transpose(1, 2)     # (B*6, H*W, 512)
        for blk in self.dblock1:
            dec_t = blk(dec_t, H, W)
        dec_t = self.dnorm1(dec_t)
        dec = dec_t.reshape(B6, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B*6, 512, 14, 28)
        dec = F.relu(F.interpolate(self.decoder1(dec), scale_factor=2, mode='bilinear', align_corners=False))
        # (B*6, 256, 28, 56)

        # Decoder stage 2: skip from feat1 (256ch, 28×56)
        dec = torch.add(dec, feat1[to_slice:])     # (B*6, 256, 28, 56)
        B6, C, H, W = dec.shape
        dec_t = dec.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            dec_t = blk(dec_t, H, W)
        dec_t = self.dnorm2(dec_t)
        dec = dec_t.reshape(B6, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B*6, 256, 28, 56)
        dec = F.relu(F.interpolate(self.decoder2(dec), scale_factor=2, mode='bilinear', align_corners=False))
        # (B*6, 128, 56, 112)

        # Decoder stage 3: skip from feat0 (128ch, 56×112)
        dec = torch.add(dec, feat0[to_slice:])     # (B*6, 128, 56, 112)
        dec = F.relu(F.interpolate(self.decoder3(dec), scale_factor=2, mode='bilinear', align_corners=False))
        # (B*6, 64, 112, 224)

        # Decoder stage 4: upsample to full resolution
        dec = F.relu(F.interpolate(self.decoder4(dec), scale_factor=2, mode='bilinear', align_corners=False))
        # (B*6, 32, 224, 448)

        out = torch.sigmoid(self.final_conv(dec))  # (B*6, 1, 224, 448)
        return out
#contributie


if __name__ == "__main__":
    # 1 batch, 7 clips (anchor + 6 local), 3 channels, 224×448 (Swin-compatible size)
    a = torch.randn(1, 7, 3, 224, 448).cuda()
    mobile = PNSNet(bn_out=(14, 28), use_kan=True, no_kan=False).cuda()
    print(mobile(a).shape)
