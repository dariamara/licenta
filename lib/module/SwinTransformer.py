#contributie
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ---------------------------------------------------------------------------
# Utility functions (copied from Swin-Transformer/models/swin_transformer.py)
# ---------------------------------------------------------------------------

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ---------------------------------------------------------------------------
# WindowAttention
# (copied from Swin-Transformer/models/swin_transformer.py, lines 77-156)
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative
    position bias. Supports both shifted and non-shifted windows.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add a learnable bias to q, k, v. Default: True
        qk_scale (float | None): Override default qk scale of head_dim**-0.5 if set.
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                   + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self):
        return (f'dim={self.dim}, window_size={self.window_size}, '
                f'num_heads={self.num_heads}')


# ---------------------------------------------------------------------------
# SwinTransformerBlock
# (copied from Swin-Transformer/models/swin_transformer.py, lines 175-294)
# ---------------------------------------------------------------------------

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to q, k, v. Default: True
        qk_scale (float | None): Override default qk scale of head_dim**-0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, \
            "shift_size must be in [0, window_size)"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # MLP: two linear layers with GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
                                  .masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = window_reverse(attn_windows, self.window_size, H, W)

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self):
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, "
                f"num_heads={self.num_heads}, window_size={self.window_size}, "
                f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}")


# ---------------------------------------------------------------------------
# PatchMerging
# (copied from Swin-Transformer/models/swin_transformer.py, lines 315-352)
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    r""" Patch Merging Layer — halves spatial resolution and doubles channels.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, \
            f"x size ({H}*{W}) must be even for PatchMerging."

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


# ---------------------------------------------------------------------------
# BasicLayer
# (copied from Swin-Transformer/models/swin_transformer.py, lines 364-423)
# ---------------------------------------------------------------------------

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add learnable bias to q, k, v. Default: True
        qk_scale (float | None): Override default qk scale of head_dim**-0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float | list[float]): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None): Downsample layer at end of layer. Default: None
        use_checkpoint (bool): Use gradient checkpointing to save memory. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim,
                                         norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self):
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, "
                f"depth={self.depth}")


# ---------------------------------------------------------------------------
# SwinPatchEmbed
# (adapted from Swin-Transformer/models/swin_transformer.py, lines 437-482)
# The size assertion is removed so non-square images work at inference.
# Renamed to SwinPatchEmbed to avoid collision with KAN.PatchEmbed.
# ---------------------------------------------------------------------------

class SwinPatchEmbed(nn.Module):
    r""" Image to Patch Embedding.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int | tuple): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                               img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # Size assertion removed — supports any H×W that is divisible by patch_size.
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, Ph*Pw, C
        if self.norm is not None:
            x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# SwinBackbone — wraps the 4 Swin stages and returns two spatial feature maps
# ---------------------------------------------------------------------------

class SwinBackbone(nn.Module):
    """Swin Transformer used as a dense feature extraction backbone.

    Returns two spatial feature maps used by PNSNet:
      - low_feature:  (B, embed_dim*4,  H/16, W/16)
      - high_feature: (B, embed_dim*8, H/32, W/32)

    Default configuration is Swin-B (embed_dim=128) which gives exactly
    512-ch (low) and 1024-ch (high) outputs — matching LightRFB expectations.

    Pretrained weights (ImageNet-22k):
        swin_base_patch4_window7_224_22k.pth
    Call load_pretrained(path) after construction to load them.

    Window-size compatibility:
        With img_size=(224, 448) and patch_size=4 the patch grid is [56, 112].
        All 4 stage resolutions ([56,112], [28,56], [14,28], [7,14]) are
        divisible by window_size=7, so the default settings are safe.
    """

    def __init__(self, img_size=(224, 448), patch_size=4, in_chans=3,
                 embed_dim=128,
                 depths=(2, 2, 18, 2),
                 num_heads=(4, 8, 16, 32),
                 window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        # --- Patch embedding ---
        self.patch_embed = SwinPatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution  # e.g. [56, 112]

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, sum(depths))]

        # --- 4 Swin stages ---
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                  self.patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # Layer norms applied before reshaping to spatial maps
        self.norm_low  = norm_layer(int(embed_dim * 4))  # 512 for Swin-B
        self.norm_high = norm_layer(int(embed_dim * 8))  # 1024 for Swin-B
        #contributie
        # Extra norms for the two shallow skip features exposed for the U-KAN decoder.
        # These are applied to the pre-PatchMerging tokens of stages 0 and 1.
        self.norm_feat0 = norm_layer(int(embed_dim))          # 128 for Swin-B
        self.norm_feat1 = norm_layer(int(embed_dim * 2))      # 256 for Swin-B
        #contributie

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            feat0:        (B, embed_dim,    H/4,  W/4)   — 128-ch for Swin-B  (pre-merge skip)
            feat1:        (B, embed_dim*2,  H/8,  W/8)   — 256-ch for Swin-B  (pre-merge skip)
            low_feature:  (B, embed_dim*4,  H/16, W/16)  — 512-ch for Swin-B
            high_feature: (B, embed_dim*8,  H/32, W/32)  — 1024-ch for Swin-B
        """
        B = x.shape[0]

        x = self.patch_embed(x)          # (B, Ph*Pw, C)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        pr = self.patches_resolution     # e.g. [56, 112]

        #contributie
        # Stage 0 — run transformer blocks, capture pre-merge tokens as feat0,
        # then apply PatchMerging separately to continue the Swin pipeline.
        for blk in self.layers[0].blocks:
            if self.layers[0].use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        H0, W0 = pr[0], pr[1]           # 56, 112
        feat0 = self.norm_feat0(x).transpose(1, 2).view(B, -1, H0, W0)
        # (B, 128, 56, 112) for Swin-B with img_size=(224,448)
        if self.layers[0].downsample is not None:
            x = self.layers[0].downsample(x)

        # Stage 1 — same pattern: blocks → feat1 → downsample.
        for blk in self.layers[1].blocks:
            if self.layers[1].use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        H1, W1 = pr[0] // 2, pr[1] // 2  # 28, 56
        feat1 = self.norm_feat1(x).transpose(1, 2).view(B, -1, H1, W1)
        # (B, 256, 28, 56) for Swin-B with img_size=(224,448)
        if self.layers[1].downsample is not None:
            x = self.layers[1].downsample(x)
        #contributie

        # --- low_feature: 1/16 of input, embed_dim*4 channels ---
        H_low = pr[0] // 4               # e.g. 56//4 = 14
        W_low = pr[1] // 4               # e.g. 112//4 = 28
        low_feature = self.norm_low(x).transpose(1, 2).view(B, -1, H_low, W_low)
        # (B, 512, 14, 28) for Swin-B with img_size=(224,448)

        x = self.layers[2](x)            # (B, Ph/8 * Pw/8, 8C)   after PatchMerging
        x = self.layers[3](x)            # (B, Ph/8 * Pw/8, 8C)   no downsample

        # --- high_feature: 1/32 of input, embed_dim*8 channels ---
        H_high = pr[0] // 8              # e.g. 56//8 = 7
        W_high = pr[1] // 8              # e.g. 112//8 = 14
        high_feature = self.norm_high(x).transpose(1, 2).view(B, -1, H_high, W_high)
        # (B, 1024, 7, 14) for Swin-B with img_size=(224,448)

        #contributie
        return feat0, feat1, low_feature, high_feature
        #contributie

    def load_pretrained(self, checkpoint_path):
        """Load ImageNet-22k pretrained weights, ignoring the classification head."""
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt)

        # Drop keys that belong to the classifier head (not present in backbone)
        keys_to_delete = [k for k in list(state_dict.keys())
                          if k.startswith('head') or k.startswith('norm.')
                          or k.startswith('avgpool')]
        for k in keys_to_delete:
            del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"[SwinBackbone] Loaded: {checkpoint_path}")
        print(f"[SwinBackbone] Missing keys  : {missing}")
        print(f"[SwinBackbone] Unexpected keys: {unexpected}")
#contributie
