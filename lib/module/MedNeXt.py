"""
Adapted from the official MedNeXt repo (nnunet_mednext/network_architecture/mednextv1/MedNextV1.py).
Original license/credit: MedNeXt authors.

Only change from the original MedNextV1.py: the import below now points at the
local MedNeXtBlocks.py (copied verbatim from blocks.py) instead of the
nnunet_mednext package path, since we're not using the rest of the nnU-Net
framework.

The full `MedNeXt` class is kept as-is below for reference / future use (e.g.
if you ever want a standalone encoder-decoder MedNeXt model). The class that
PNSPlusNetwork.py actually imports and uses is `MedNeXtEncoder`, defined at
the bottom of this file -- it's the first half of `MedNeXt` (stem + 4 encoder
stages + bottleneck) with the decoder half (up_X / dec_block_X / out_X)
removed, since PNSNet already supplies its own decoder.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from lib.module.MedNeXtBlocks import *


class MedNeXt(nn.Module):

    def __init__(self,
        in_channels: int,
        n_channels: int,
        n_classes: int,
        exp_r: int = 4,                            # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                      # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,             # Additional 'res' connection on up and down convs
        checkpoint_style: bool = None,            # Either inside block or outside block
        block_counts: list = [2,2,2,2,2,2,2,2,2], # Can be used to test staging ratio:
                                            # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type = 'group',
        dim = '3d',                                # 2d or 3d
        grn = False
    ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d

        self.stem = conv(in_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[0])]
        )

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[1])]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[2])]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[3])]
        )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8*n_channels,
            out_channels=16*n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*16,
                out_channels=n_channels*16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[4])]
        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=16*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[5])]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[6])]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4*n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[7])]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2*n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[8])]
        )

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels*2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels*4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels*8, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=n_channels*16, n_classes=n_classes, dim=dim)

        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x):

        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)

            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)
            del x_res_3, x_up_3

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)
            del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)
            del x_res_1, x_up_1

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)

        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            dec_x = x_res_3 + x_up_3
            x = self.dec_block_3(dec_x)

            if self.do_ds:
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1
            x = self.dec_block_1(dec_x)
            if self.do_ds:
                x_ds_1 = self.out_1(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x


#contributie
def create_mednext_encoder(model_id, in_channels=3, n_channels=64, kernel_size=3, dim='2d'):
    """
    Factory that returns the full MedNeXt (encoder + decoder weights) with
    deep_supervision=False. PNSPlusNetwork drives it layer-by-layer using only
    the encoder half (stem, enc_block_0..3, down_0..3, bottleneck); the decoder
    weights exist but are never called.

    n_channels=64 is intentional: channel progression 64->128->256->512->1024
    so enc_block_3 outputs 512 and bottleneck outputs 1024, matching the
    channel counts expected by PNSPlusNetwork's LightRFB / up_sample_* layers.
    (The official MedNeXt presets use n_channels=32, which gives 256/512 -- wrong here.)
    """
    presets = {
        'S': dict(exp_r=2,                     block_counts=[2,2,2,2,2,2,2,2,2], checkpoint_style=None),
        'B': dict(exp_r=[2,3,4,4,4,4,4,3,2],  block_counts=[2,2,2,2,2,2,2,2,2], checkpoint_style=None),
        'M': dict(exp_r=[2,3,4,4,4,4,4,3,2],  block_counts=[3,4,4,4,4,4,4,4,3], checkpoint_style='outside_block'),
        'L': dict(exp_r=[3,4,8,8,8,8,8,4,3],  block_counts=[3,4,8,8,8,8,8,4,3], checkpoint_style='outside_block'),
    }
    cfg = presets[model_id]
    return MedNeXt(
        in_channels=in_channels,
        n_channels=n_channels,
        n_classes=1,
        exp_r=cfg['exp_r'],
        kernel_size=kernel_size,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        checkpoint_style=cfg['checkpoint_style'],
        block_counts=cfg['block_counts'],
        norm_type='group',
        dim=dim,
        grn=False,
    )
#contributie