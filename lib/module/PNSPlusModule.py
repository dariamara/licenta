from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


def _unfold_neighbors(x, radius, dilation):
    """
    For every spatial location (h, w), gather the (2*radius+1)^2 neighboring vectors
    from every time step of x, zero-padded at the borders.

    Input:  x (B, T, C, H, W)
    Output: (B, T*diameter*diameter, C, H, W), candidates ordered time-major then
            row-major within the window -- matching the original CUDA kernel's
            `cal_time * diameter*diameter + kh*diameter + kw` indexing exactly.
    """
    B, T, C, H, W = x.shape
    diameter = 2 * radius + 1
    pad = radius * dilation

    unf = F.unfold(x.reshape(B * T, C, H, W), kernel_size=diameter, dilation=dilation, padding=pad)
    unf = unf.view(B, T, C, diameter * diameter, H, W)
    unf = unf.permute(0, 1, 3, 2, 4, 5).contiguous()
    return unf.view(B, T * diameter * diameter, C, H, W)


def relevance_measuring(query, key, radius=1, dilation=1):
    """
    Pure-PyTorch replacement for the original self_cuda_backend.weight_forward CUDA kernel
    (this app runs on CPU, so the compiled extension used at training time isn't available).
    Out-of-bounds neighbors score 0 in that kernel (it always writes `sum`, which stays 0.0
    when the bounds check fails, rather than the -inf the Python wrapper pre-fills) -- which
    is exactly what F.unfold's zero-padding already gives us, so no extra masking is needed.
    """
    key_cand = _unfold_neighbors(key, radius, dilation)
    return torch.einsum('bqchw,bkchw->bqkhw', query, key_cand)


def spatial_temporal_aggregation(weight, proj, radius=1, dilation=1):
    """Pure-PyTorch replacement for self_cuda_backend.map_forward."""
    value_cand = _unfold_neighbors(proj, radius, dilation)
    return torch.einsum('bqkhw,bkchw->bqchw', weight, value_cand)


class NS_Block(nn.Module):
    def __init__(self, bn_out, channels_in=32, n_head=4, d_k=8, d_v=8, radius=[3, 3, 3, 3], dilation=[1, 3, 5, 7]):
        super(NS_Block, self).__init__()
        self.channels_in = channels_in
        self.n_head = n_head
        self.d_k = d_k
        self.radius = radius
        self.dilation = dilation
        self.query_conv = nn.Conv3d(channels_in, n_head * d_k, 1, bias=False)
        self.key_conv = nn.Conv3d(channels_in, n_head * d_k, 1, bias=False)
        self.value_conv = nn.Conv3d(channels_in, n_head * d_v, 1, bias=False)
        self.output_Linear = nn.Conv3d(n_head * d_v, channels_in, 1, bias=False)
        # Optimization: self-adapting layer normalization
        self.bn = nn.LayerNorm([int(self.channels_in/self.n_head), bn_out[0], bn_out[1]])

    def forward(self, first, x):
        dilation, radius = self.dilation, self.radius
        x_ = x.permute(0, 2, 1, 3, 4).contiguous()
        first_ = first.permute(0, 2, 1, 3, 4).contiguous()
        query = self.query_conv(first_).permute(0, 2, 1, 3, 4)
        query_chunk = query.chunk(self.n_head, 2)
        key = self.key_conv(x_).permute(0, 2, 1, 3, 4)
        key_chunk = key.chunk(self.n_head, 2)
        value = self.value_conv(x_).permute(0, 2, 1, 3, 4)
        value_chunk = value.chunk(self.n_head, 2)

        M_T, M_A = [], []

        for i in range(self.n_head):
            query_i = query_chunk[i].contiguous()
            query_i = self.bn(query_i)
            key_i = key_chunk[i].contiguous()
            value_i = value_chunk[i].contiguous()
            # Optimization: self-adapting scaling factor
            M_A_i = relevance_measuring(query_i, key_i, radius[i], dilation[i]) / sqrt(self.channels_in/self.n_head)
            M_A.append(F.softmax(M_A_i, dim=2))
            M_T.append(spatial_temporal_aggregation(M_A_i, value_i, radius[i], dilation[i]))

        M_S, _ = torch.max(torch.cat(M_A, dim=2), dim=2)
        M_T = torch.cat(M_T, dim=2).permute(0, 2, 1, 3, 4)
        out_cat = self.output_Linear(M_T) * M_S.unsqueeze(2).permute(0, 2, 1, 3, 4)

        return out_cat.permute(0, 2, 1, 3, 4)


__all__ = ["NS_Block", "relevance_measuring", "spatial_temporal_aggregation"]
