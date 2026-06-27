# contributie
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.module.KAN import KANBlock, PatchEmbed
from lib.module.MaxViTBackbone import MaxViTBackbone
from lib.module.PNSPlusModule import NS_Block


# ── Encoder building block ────────────────────────────────────────────────────

class ConvLayer(nn.Module):
    """Conv2d → BatchNorm2d → ReLU  (no pooling — caller handles stride)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ── Decoder building block ────────────────────────────────────────────────────

class D_ConvLayer(nn.Module):
    """Bilinear upsample to a target spatial size, then Conv2d → BN → ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.conv(x)


# ── Main network ──────────────────────────────────────────────────────────────

class MaxViTUKAN(nn.Module):
    """
    U-KAN decoder on top of a pretrained MaxViT encoder.

    Encoder hierarchy (spatial sizes for 256×448 input):
        t1  : [B*T,  32, H,    W   ]   early conv, full resolution
        t2  : [B*T,  64, H/2,  W/2 ]   early conv after MaxPool
        t3  : [B*T, 256, H/8,  W/8 ]   MaxViT low  feature, projected
        t4  : [B*T, 320, H/16, W/16]   MaxViT high feature, projected
        bot : [B*T, 512, H/32, W/32]   KAN bottleneck (PatchEmbed + KANBlock)

    Temporal processing (NS_Block) is applied on t3 (32ch squeezed),
    matching the logic of the original PNSNet.  After NS_Block only the
    video_time_clips local frames are kept; all encoder features are sliced
    accordingly before entering the decoder.

    Decoder:
        d1 [320, H/16] ← upsample(bot) + t4       + KANBlock
        d2 [256, H/8 ] ← upsample(d1)  + t3_att   + KANBlock
        d3 [ 64, H/2 ] ← upsample(d2)  + t2        (4× upsample bridges H/4 gap)
        d4 [ 32, H   ] ← upsample(d3)  + t1
        d5 [ 32, H   ] ← refinement conv
        out[  1, H   ] ← Conv(32→1) + Sigmoid
    """

    def __init__(
        self,
        img_size=(256, 448),
        embed_dims=(256, 320, 512),
        video_time_clips=6,
        drop_path_rate=0.1,
    ):
        super().__init__()
        H, W = img_size
        self.video_time_clips = video_time_clips
        self.img_size = img_size

        # ── Early conv encoder ────────────────────────────────────────────
        self.encoder1 = ConvLayer(3, 32)       # t1 : [B*T,  32, H,   W  ]
        self.pool1    = nn.MaxPool2d(2, 2)     # H  → H/2
        self.encoder2 = ConvLayer(32, 64)      # t2 : [B*T,  64, H/2, W/2]

        # ── MaxViT backbone (pretrained) ──────────────────────────────────
        self.backbone = MaxViTBackbone()
        # backbone param name prefix → "backbone.*" for optimizer splitting

        # ── Project MaxViT features to U-KAN channel sizes ────────────────
        self.proj_t3 = nn.Conv2d(512,  embed_dims[0], kernel_size=1, bias=False)
        self.proj_t4 = nn.Conv2d(1024, embed_dims[1], kernel_size=1, bias=False)

        nn.init.kaiming_normal_(self.proj_t3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj_t4.weight, mode='fan_out', nonlinearity='relu')

        # ── Temporal processing: squeeze t3 to 32ch for NS_Block ─────────
        # LightRFB is not used in U-KAN; we squeeze via 1×1 conv instead.
        self.squeeze_t3 = nn.Sequential(
            nn.Conv2d(embed_dims[0], 32, kernel_size=1, bias=False),
            nn.GroupNorm(2, 32),
            nn.Mish(),
        )
        self.expand_t3 = nn.Conv2d(32, embed_dims[0], kernel_size=1, bias=False)

        # NS_Block operates on squeezed t3: 32ch, spatial (H/8, W/8)
        bn_out_t3 = (H // 8, W // 8)
        self.NSB_global = NS_Block(
            bn_out=bn_out_t3, channels_in=32,
            radius=[3, 3, 3, 3], dilation=[3, 4, 3, 4],
        )
        self.NSB_local = NS_Block(
            bn_out=bn_out_t3, channels_in=32,
            radius=[3, 3, 3, 3], dilation=[1, 2, 1, 2],
        )

        # ── KAN bottleneck ────────────────────────────────────────────────
        # PatchEmbed: t4 [H/16, W/16] → tokens at [H/32, W/32]
        self.patch_embed_bot = PatchEmbed(
            img_size=(H // 16, W // 16),
            patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2],
        )
        self.bottleneck_block = nn.ModuleList([
            KANBlock(dim=embed_dims[2], drop_path=drop_path_rate)
        ])
        self.bottleneck_norm = nn.LayerNorm(embed_dims[2])

        # ── Decoder stage 1: 512 → 320, H/32 → H/16, KANBlock ───────────
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.dblock1  = nn.ModuleList([KANBlock(dim=embed_dims[1], drop_path=drop_path_rate)])
        self.dnorm1   = nn.LayerNorm(embed_dims[1])

        # ── Decoder stage 2: 320 → 256, H/16 → H/8, KANBlock ────────────
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.dblock2  = nn.ModuleList([KANBlock(dim=embed_dims[0], drop_path=drop_path_rate)])
        self.dnorm2   = nn.LayerNorm(embed_dims[0])

        # ── Decoder stage 3: 256 → 64, H/8 → H/2 (4× upsample) ─────────
        self.decoder3 = D_ConvLayer(embed_dims[0], 64)

        # ── Decoder stage 4: 64 → 32, H/2 → H ───────────────────────────
        self.decoder4 = D_ConvLayer(64, 32)

        # ── Decoder stage 5: refinement conv at full resolution ───────────
        self.decoder5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── Segmentation head ─────────────────────────────────────────────
        self.final = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _kan_block(self, x, blocks, norm):
        """Flatten spatial → run KANBlocks + norm → reshape back to [B,C,H,W]."""
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # [B, H*W, C]
        for blk in blocks:
            x = blk(x, H, W)
        x = norm(x)
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def _select_local(self, feat, B, T_total):
        """
        Select the video_time_clips local frames (indices 1 … T) per batch group.
        Input  : [B*T_total, C, H, W]
        Output : [B*video_time_clips, C, H, W]
        """
        f5d = feat.view(B, T_total, *feat.shape[1:])
        local = f5d[:, 1:1 + self.video_time_clips, ...]
        return local.contiguous().view(-1, *feat.shape[1:])

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x):
        origin_shape = x.shape                          # [B, T_total, 3, H, W]
        B, T_total   = origin_shape[0], origin_shape[1]
        x_flat = x.view(-1, *origin_shape[2:])          # [B*T_total, 3, H, W]

        # ── ENCODER ──────────────────────────────────────────────────────
        t1 = self.encoder1(x_flat)                      # [B*T, 32, H,   W  ]
        t2 = self.encoder2(self.pool1(t1))              # [B*T, 64, H/2, W/2]

        low_feat, high_feat = self.backbone(x_flat)     # [B*T, 512, H/8], [B*T, 1024, H/16]
        t3 = self.proj_t3(low_feat)                     # [B*T, 256, H/8,  W/8 ]
        t4 = self.proj_t4(high_feat)                    # [B*T, 320, H/16, W/16]

        # ── KAN BOTTLENECK ────────────────────────────────────────────────
        x_bot, H_b, W_b = self.patch_embed_bot(t4)     # [B*T, N, 512]
        for blk in self.bottleneck_block:
            x_bot = blk(x_bot, H_b, W_b)
        x_bot = self.bottleneck_norm(x_bot)
        bottleneck = (
            x_bot.reshape(-1, H_b, W_b, x_bot.shape[-1])
                 .permute(0, 3, 1, 2).contiguous()
        )                                               # [B*T, 512, H/32, W/32]

        # ── TEMPORAL PROCESSING on t3 ─────────────────────────────────────
        t3_sq   = self.squeeze_t3(t3)                           # [B*T, 32, H/8, W/8]
        t3_5d   = t3_sq.view(B, T_total, *t3_sq.shape[1:])     # [B, T, 32, H/8, W/8]

        t3_global = (
            t3_5d[:, 0, ...].unsqueeze(1)
                             .repeat(1, self.video_time_clips, 1, 1, 1)
        )                                                       # [B, T_clips, 32, H/8, W/8]
        t3_local_raw = t3_5d[:, 1:1 + self.video_time_clips, ...]

        # Two-pass NS_Block (same pattern as original PNSNet)
        t3_att1  = self.NSB_global(t3_global, t3_local_raw) + t3_local_raw
        t3_att2  = self.NSB_local(t3_att1, t3_att1) + t3_att1
        t3_att   = (t3_att2 + t3_local_raw).contiguous()       # [B, T_clips, 32, H/8, W/8]

        t3_att   = t3_att.view(-1, *t3_att.shape[2:])          # [B*T_clips, 32, H/8, W/8]
        t3_att   = self.expand_t3(t3_att)                      # [B*T_clips, 256, H/8, W/8]

        # ── ALIGN SKIP CONNECTIONS TO LOCAL FRAMES ────────────────────────
        # All encoder features were [B*T_total, ...]; keep only local frames.
        t1_l   = self._select_local(t1,         B, T_total)    # [B*T_clips, 32, H,   W  ]
        t2_l   = self._select_local(t2,         B, T_total)    # [B*T_clips, 64, H/2, W/2]
        t4_l   = self._select_local(t4,         B, T_total)    # [B*T_clips, 320, H/16, W/16]
        bot_l  = self._select_local(bottleneck, B, T_total)    # [B*T_clips, 512, H/32, W/32]

        # ── DECODER ──────────────────────────────────────────────────────
        # Stage 1: 512 → 320, H/32 → H/16, skip + KAN
        d1 = self.decoder1(bot_l, target_size=(t4_l.shape[2], t4_l.shape[3]))
        d1 = d1 + t4_l
        d1 = self._kan_block(d1, self.dblock1, self.dnorm1)

        # Stage 2: 320 → 256, H/16 → H/8, skip + KAN
        d2 = self.decoder2(d1, target_size=(t3_att.shape[2], t3_att.shape[3]))
        d2 = d2 + t3_att
        d2 = self._kan_block(d2, self.dblock2, self.dnorm2)

        # Stage 3: 256 → 64, H/8 → H/2 (4× upsample bridges the H/4 gap)
        d3 = self.decoder3(d2, target_size=(t2_l.shape[2], t2_l.shape[3]))
        d3 = d3 + t2_l

        # Stage 4: 64 → 32, H/2 → H
        d4 = self.decoder4(d3, target_size=(t1_l.shape[2], t1_l.shape[3]))
        d4 = d4 + t1_l

        # Stage 5: refinement at full resolution
        d5 = self.decoder5(d4)

        # Segmentation head + resize to original input resolution
        out = self.final(d5)
        out = torch.sigmoid(
            F.interpolate(out, size=(origin_shape[-2], origin_shape[-1]),
                          mode='bilinear', align_corners=False)
        )
        return out                                              # [B*T_clips, 1, H, W]


if __name__ == "__main__":
    a = torch.randn(1, 7, 3, 256, 448).cuda()
    model = MaxViTUKAN(img_size=(256, 448), video_time_clips=6).cuda()
    out = model(a)
    print("Output shape:", out.shape)   # expected: [6, 1, 256, 448]
# contributie
