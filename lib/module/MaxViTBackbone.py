# 3contributie
import torch.nn as nn
import timm


class MaxViTBackbone(nn.Module):
    """
    PyTorch MaxViT backbone (via timm) producing two feature pyramid levels:
      - low_feature:  [B, 512,  H/8,  W/8]
      - high_feature: [B, 1024, H/16, W/16]

    These channel counts match what LightRFB expects in PNSPlusNetwork.

    Requirements:
        pip install timm>=0.9.0

    Input constraint: H and W must both be divisible by 8 (MaxViT window_size=8
    is baked into the tf_256 pretrained weights used here). For the default
    training size of 256x448 all intermediate resolutions (128x224, 64x112,
    32x56, 16x28) satisfy this constraint.

    If you switch to a different model_name, re-verify that
    self.encoder.feature_info.channels() returns [low_ch, high_ch] matching
    the out_indices you pass (default: indices 1 and 2 → strides 8 and 16).
    """

    def __init__(
        self,
        model_name: str = 'maxvit_tiny_tf_256.in1k',
        pretrained: bool = True,
    ):
        super().__init__()

        # features_only=True: the encoder returns a list of intermediate feature
        # maps instead of classification logits.
        # out_indices selects which pyramid levels are returned:
        #   index 0 → stride  4  (H/4)
        #   index 1 → stride  8  (H/8)   ← low_feature
        #   index 2 → stride 16  (H/16)  ← high_feature
        #   index 3 → stride 32  (H/32)
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2),
        )

        # Read actual channel counts produced by the encoder so the code
        # stays correct if you swap to a larger MaxViT variant.
        ch = self.encoder.feature_info.channels()   # e.g. [128, 256] for Tiny
        low_ch, high_ch = ch[0], ch[1]

        # 1x1 convolutions project to the channel sizes that LightRFB expects.
        self.proj_low  = nn.Conv2d(low_ch,  512,  kernel_size=1, bias=False)
        self.proj_high = nn.Conv2d(high_ch, 1024, kernel_size=1, bias=False)

        nn.init.kaiming_normal_(self.proj_low.weight,  mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj_high.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        Args:
            x: FloatTensor [B, 3, H, W]
        Returns:
            low_feature:  FloatTensor [B, 512,  H/8,  W/8]
            high_feature: FloatTensor [B, 1024, H/16, W/16]
        """
        feats = self.encoder(x)                        # list of 2 tensors
        low_feature  = self.proj_low(feats[0])
        high_feature = self.proj_high(feats[1])
        return low_feature, high_feature
# 3contributie
