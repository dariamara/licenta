import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.module.PNSPlusModule import NS_Block
#contributie
from lib.module.KAN import KANBlock, D_ConvLayer
from lib.module.MSCAN import create_mscan_encoder


class PNSNet(nn.Module):
    """
    SegNeXt-encoded U-KAN video model.

    Encoder  : MSCAN-small backbone (4 scales).
    Bottleneck: U-KAN tokenised-KAN block at 512 ch (stride-32).
    Temporal  : squeeze 512→32 → NS-Block global/local → expand 32→512.
    Decoder   : U-KAN KAN-augmented decoder with skip connections from MSCAN
                feats[2] (320 ch), feats[1] (128 ch), feats[0] (64 ch).

    Input  : (B, 6, 3, H, W)
    Output : (B*5, 1, H, W)  — frame 0 is used as global reference only.
    """

    def __init__(self, bn_out, use_kan=True):
        super(PNSNet, self).__init__()
        no_kan = not use_kan

        # MSCAN backbone
        # outputs: [64@s4, 128@s8, 320@s16, 512@s32] for 'small' variant
        self.feature_extractor = create_mscan_encoder(variant='small', in_chans=3, drop_path_rate=0.1)

        # drop-path rates spread across the 3 KAN blocks (bottleneck + 2 decoder)
        dpr = [x.item() for x in torch.linspace(0, 0.1, 3)]

        # U-KAN bottleneck — tokenised KAN at 512 ch
        self.norm_bot  = nn.LayerNorm(512)
        self.block_bot = nn.ModuleList([
            KANBlock(dim=512, drop=0., drop_path=dpr[0], no_kan=no_kan)
        ])

        # Squeeze 512→32 for the temporal NS-Block
        self.squeeze = nn.Sequential(
            nn.Conv2d(512, 32, 1),
            nn.GroupNorm(2, 32),
            nn.Mish(inplace=True),
        )
        # Expand 32→512 to re-enter the U-KAN decoder
        self.expand = nn.Conv2d(32, 512, 1, bias=False)

        # Temporal NS-Blocks (operate at 32 ch, stride-32 spatial resolution)
        self.NSB_global = NS_Block(bn_out=bn_out, channels_in=32,
                                   radius=[3, 3, 3, 3], dilation=[3, 4, 3, 4])
        self.NSB_local  = NS_Block(bn_out=bn_out, channels_in=32,
                                   radius=[3, 3, 3, 3], dilation=[1, 2, 1, 2])

        # U-KAN decoder stage 1: 512→320, skip with feats[2] (320 ch @ stride-16)
        self.decoder1 = D_ConvLayer(512, 320)
        self.dblock1  = nn.ModuleList([
            KANBlock(dim=320, drop=0., drop_path=dpr[1], no_kan=no_kan)
        ])
        self.dnorm1 = nn.LayerNorm(320)

        # U-KAN decoder stage 2: 320→128, skip with feats[1] (128 ch @ stride-8)
        self.decoder2 = D_ConvLayer(320, 128)
        self.dblock2  = nn.ModuleList([
            KANBlock(dim=128, drop=0., drop_path=dpr[2], no_kan=no_kan)
        ])
        self.dnorm2 = nn.LayerNorm(128)

        # U-KAN decoder stages 3-5: plain conv, skip with feats[0] (64 ch @ stride-4)
        self.decoder3 = D_ConvLayer(128, 64)
        self.decoder4 = D_ConvLayer(64, 32)
        self.decoder5 = D_ConvLayer(32, 16)

        self.final   = nn.Conv2d(16, 1, kernel_size=1)
        self.use_kan = use_kan

    def forward(self, x):
        origin_shape = x.shape                    # (B, 6, 3, H, W)
        x = x.view(-1, *origin_shape[2:])         # (B*6, 3, H, W)

        # ── Encoder ──────────────────────────────────────────────────────────
        feats = self.feature_extractor(x)
        t1  = feats[0]   # (B*6,  64, H/4,  W/4)
        t2  = feats[1]   # (B*6, 128, H/8,  W/8)
        t3  = feats[2]   # (B*6, 320, H/16, W/16)
        bot = feats[3]   # (B*6, 512, H/32, W/32)

        # ── U-KAN tokenised bottleneck ────────────────────────────────────────
        B_tok, _, H_tok, W_tok = bot.shape
        out = bot.flatten(2).transpose(1, 2)      # (B*6, H*W tokens, 512)
        for blk in self.block_bot:
            out = blk(out, H_tok, W_tok)
        out = self.norm_bot(out)
        out = (out.reshape(B_tok, H_tok, W_tok, -1)
                  .permute(0, 3, 1, 2).contiguous())  # (B*6, 512, H/32, W/32)

        # ── Temporal NS-Block ─────────────────────────────────────────────────
        out_32 = self.squeeze(out)                # (B*6, 32, H/32, W/32)
        out_32 = out_32.view(*origin_shape[:2], *out_32.shape[1:])  # (B, 6, 32, ...)

        high_global = out_32[:, 0, ...].unsqueeze(1).repeat(1, 6, 1, 1, 1)  # (B, 6, 32, ...)
        high_local  = out_32[:, 1:7, ...]                                    # (B, 5, 32, ...)

        high_1 = self.NSB_global(high_global, high_local) + high_local
        high_2 = self.NSB_local(high_1, high_1) + high_1
        out_32 = (high_2 + high_local).contiguous()                          # (B, 5, 32, ...)
        out_32 = out_32.view(-1, *out_32.shape[2:])                          # (B*5, 32, H/32, W/32)

        # Expand back to 512 ch for the U-KAN decoder
        out = self.expand(out_32)                 # (B*5, 512, H/32, W/32)

        # Align skip tensors to the 5 local frames (drop the global reference frame)
        B = origin_shape[0]
        t3 = t3[B:]   # (B*5, 320, H/16, W/16)
        t2 = t2[B:]   # (B*5, 128, H/8,  W/8)
        t1 = t1[B:]   # (B*5,  64, H/4,  W/4)

        # ── U-KAN decoder stage 1: 512→320, skip +t3, KAN block ──────────────
        out = F.relu(F.interpolate(self.decoder1(out),
                                   scale_factor=(2, 2), mode='bilinear', align_corners=False))
        out = torch.add(out, t3)
        _, _, H_d1, W_d1 = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H_d1, W_d1)
        out = self.dnorm1(out)
        out = (out.reshape(-1, H_d1, W_d1, 320)
                  .permute(0, 3, 1, 2).contiguous())

        # ── U-KAN decoder stage 2: 320→128, skip +t2, KAN block ──────────────
        out = F.relu(F.interpolate(self.decoder2(out),
                                   scale_factor=(2, 2), mode='bilinear', align_corners=False))
        out = torch.add(out, t2)
        _, _, H_d2, W_d2 = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H_d2, W_d2)
        out = self.dnorm2(out)
        out = (out.reshape(-1, H_d2, W_d2, 128)
                  .permute(0, 3, 1, 2).contiguous())

        # ── U-KAN decoder stages 3-5: conv only, skip +t1 at stage 3 ─────────
        out = F.relu(F.interpolate(self.decoder3(out),
                                   scale_factor=(2, 2), mode='bilinear', align_corners=False))
        out = torch.add(out, t1)

        out = F.relu(F.interpolate(self.decoder4(out),
                                   scale_factor=(2, 2), mode='bilinear', align_corners=False))
        out = F.relu(F.interpolate(self.decoder5(out),
                                   scale_factor=(2, 2), mode='bilinear', align_corners=False))

        out = torch.sigmoid(
            F.interpolate(self.final(out),
                          size=(origin_shape[-2], origin_shape[-1]),
                          mode='bilinear', align_corners=False)
        )
        return out                                # (B*5, 1, H, W)
#contributie


if __name__ == "__main__":
    a = torch.randn(1, 6, 3, 256, 448).cuda()
    model = PNSNet(bn_out=(256 // 32, 448 // 32), use_kan=True).cuda()
    out = model(a)
    print(out.shape)   # expect (5, 1, 256, 448)
