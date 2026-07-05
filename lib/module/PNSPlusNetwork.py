import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lib.module.PNSPlusModule import NS_Block
#contributie
from lib.module.KAN import KANBlock, PatchEmbed, ConvLayer, D_ConvLayer
from lib.module.ConvNeXtV2 import convnextv2_base
#contributie


#contributie
class PNSNet(nn.Module):
    def __init__(self, bn_out=(32, 56)):
        super(PNSNet, self).__init__()

        self.feature_extractor = convnextv2_base(pretrained=True, in_22k=True, num_classes=21841, drop_path_rate=0.2)

        # embed_dims: [low_feature_ch, intermediate_ch, bottleneck_ch]
        # chosen to match ConvNeXtV2 base output channels (512, 768, 1024)
        embed_dims = [512, 768, 1024]

        # U-KAN encoder: two tokenized KAN stages on top of ConvNeXtV2
        # patch_embed3: high_feature (1024ch, 8x14) -> 768ch, keeps spatial size (stride=1)
        self.patch_embed3 = PatchEmbed(img_size=8, patch_size=3, stride=1,
                                       in_chans=embed_dims[2], embed_dim=embed_dims[1])
        # patch_embed4: 768ch 8x14 -> 1024ch 4x7 (stride=2 halves spatial)
        self.patch_embed4 = PatchEmbed(img_size=8, patch_size=3, stride=2,
                                       in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.block1 = nn.ModuleList([KANBlock(dim=embed_dims[1])])
        self.block2 = nn.ModuleList([KANBlock(dim=embed_dims[2])])

        self.norm3 = nn.LayerNorm(embed_dims[1])
        self.norm4 = nn.LayerNorm(embed_dims[2])

        # U-KAN decoder: KAN blocks at each skip-connection stage
        self.dblock1 = nn.ModuleList([KANBlock(dim=embed_dims[1])])
        self.dblock2 = nn.ModuleList([KANBlock(dim=embed_dims[0])])

        self.dnorm3 = nn.LayerNorm(embed_dims[1])
        self.dnorm4 = nn.LayerNorm(embed_dims[0])

        # U-KAN conv decoder layers (D_ConvLayer = depthwise-style double conv)
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])   # 1024 -> 768
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])   # 768  -> 512
        self.decoder3 = D_ConvLayer(embed_dims[0], 32)              # 512  -> 32  (to temporal resolution)
        self.decoder4 = D_ConvLayer(32, 16)
        self.decoder5 = D_ConvLayer(16, 16)

        # Temporal NS blocks preserved from original PNSNet
        # bn_out must match the spatial size of features entering NSB: (H//8, W//8)
        # e.g. for input (256, 448): bn_out=(32, 56)
        self.NSB_global = NS_Block(bn_out=bn_out, channels_in=32, radius=[3, 3, 3, 3], dilation=[3, 4, 3, 4])
        self.NSB_local  = NS_Block(bn_out=bn_out, channels_in=32, radius=[3, 3, 3, 3], dilation=[1, 2, 1, 2])

        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(16, 1, kernel_size=1, bias=False))

        # trade compute for memory on the heavy ConvNeXtV2 stages
        self.use_checkpoint = True

    def _run_stage(self, stage, x):
        # gradient checkpointing: recompute activations in backward instead of storing them
        if self.use_checkpoint and self.training:
            return checkpoint(stage, x)
        return stage(x)

    def forward(self, x):
        origin_shape = x.shape                       # (B, 7, 3, H, W)
        x = x.view(-1, *origin_shape[2:])            # (B*7, 3, H, W)
        B = x.shape[0]

        # ConvNeXtV2 backbone: extract multi-scale features
        x = self.feature_extractor.downsample_layers[0](x)
        x = self._run_stage(self.feature_extractor.stages[0], x)
        x = self.feature_extractor.downsample_layers[1](x)
        x = self._run_stage(self.feature_extractor.stages[1], x)
        low_feature = self.feature_extractor.downsample_layers[2](x)
        low_feature = self._run_stage(self.feature_extractor.stages[2], low_feature)    # (B*7, 512, H/16, W/16)
        high_feature = self.feature_extractor.downsample_layers[3](low_feature)
        high_feature = self._run_stage(self.feature_extractor.stages[3], high_feature)  # (B*7, 1024, H/32, W/32)

        # U-KAN encoder stage 4: tokenize high_feature, produce skip t4
        out, H4, W4 = self.patch_embed3(high_feature)
        for blk in self.block1:
            out = blk(out, H4, W4)
        out = self.norm3(out)
        t4 = out.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()  # (B*7, 768, H4, W4)

        # U-KAN bottleneck: further tokenize t4
        out, H5, W5 = self.patch_embed4(t4)
        for blk in self.block2:
            out = blk(out, H5, W5)
        out = self.norm4(out)
        out = out.reshape(B, H5, W5, -1).permute(0, 3, 1, 2).contiguous()  # (B*7, 1024, H5, W5)

        # U-KAN decoder stage 4: upsample bottleneck + skip t4 + KAN
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear', align_corners=False))
        out = torch.add(out, t4)
        _, _, H4, W4 = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H4, W4)
        out = self.dnorm3(out)
        out = out.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()  # (B*7, 768, H4, W4)

        # U-KAN decoder stage 3: upsample + skip low_feature (t3) + KAN
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear', align_corners=False))
        out = torch.add(out, low_feature)
        _, _, H3, W3 = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H3, W3)
        out = self.dnorm4(out)
        out = out.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()  # (B*7, 512, H3, W3)

        # Decoder stage 2: upsample to temporal-processing resolution (H/8, W/8)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear', align_corners=False))
        # (B*7, 32, H/8, W/8)

        t_h = out  # skip for temporal residual connection

        # Temporal feature separation and NS_Block aggregation (preserved from original PNSNet)
        out = out.view(*origin_shape[:2], *out.shape[1:])               # (B, 7, 32, H/8, W/8)
        high_feature_global = out[:, 0, ...].unsqueeze(1).repeat(1, 6, 1, 1, 1)  # (B, 6, 32, H/8, W/8)
        high_feature_local  = out[:, 1:7, ...]                                    # (B, 6, 32, H/8, W/8)

        # NS_Block uses a custom CUDA kernel that only supports float32 -> run outside autocast
        with torch.cuda.amp.autocast(enabled=False):
            high_feature_global = high_feature_global.float()
            high_feature_local = high_feature_local.float()
            high_feature_1 = self.NSB_global(high_feature_global, high_feature_local) + high_feature_local
            high_feature_2 = self.NSB_local(high_feature_1, high_feature_1) + high_feature_1
        out = (high_feature_2 + high_feature_local).contiguous().view(-1, *high_feature_2.shape[2:])
        # (B*6, 32, H/8, W/8)

        to_slice = t_h.shape[0] - out.shape[0]   # = B (remove anchor frame from skip)
        out = out + t_h[to_slice:].to(out.dtype)

        # Final decoder stages
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear', align_corners=False))
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear', align_corners=False))

        out = torch.sigmoid(
            F.interpolate(self.SegNIN(out), size=(origin_shape[-2], origin_shape[-1]),
                          mode='bilinear', align_corners=False)
        )
        return out
#contributie


if __name__ == "__main__":
    #contributie
    a = torch.randn(1, 7, 3, 256, 448).cuda()
    mobile = PNSNet(bn_out=(32, 56)).cuda()
    print(mobile(a).shape)
    #contributie
