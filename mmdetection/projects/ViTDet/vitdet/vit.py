# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.model import BaseModule, ModuleList

from mmdet.registry import MODELS

from mmpretrain.models import (VisionLLaMA, TransformerEncoderLayerRoPE, MultiheadAttentionRoPE, precompute_freqs_cis_2d, \
                               apply_rotary_emb, precompute_freqs_cis_2d_general, apply_rotary_emb_single)


class MultiheadAttentionRoPEDET(MultiheadAttentionRoPE):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 use_layer_scale=False,
                 layer_scale_init_value=0.,
                 init_cfg=None,
                 input_size=None):
        super().__init__(embed_dims, num_heads, input_dims, attn_drop, proj_drop, dropout_layer, qkv_bias, qk_scale,
                         proj_bias, v_shortcut, use_layer_scale, layer_scale_init_value, init_cfg)
        self.input_size = input_size
        self.freqs_cis = None

    def forward(self, x, freqs_cis=None):
        B, N, _ = x.shape
        if self.freqs_cis is None:
            num_patches = self.input_size[0] * self.input_size[1]
            self.freqs_cis = precompute_freqs_cis_2d(self.head_dims, num_patches).to(x.device)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 1, 3, 4) # B, N, 3, H, Hc -> 3, B, N, H, Hc
        q, k, v = qkv[0], qkv[1], qkv[2] #  B N H Hc
        q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis) # B, N, H, Hc

        attn_drop = self.attn_drop if self.training else 0.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class MultiheadAttentionRoPEDETv1(MultiheadAttentionRoPEDET):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 use_layer_scale=False,
                 layer_scale_init_value=0.,
                 init_cfg=None,
                 input_size=None,
                 sr_ratio=0,
                 avg=False,
                 ):
        super().__init__(embed_dims, num_heads, input_dims, attn_drop, proj_drop, dropout_layer, qkv_bias, qk_scale,
                         proj_bias, v_shortcut, use_layer_scale, layer_scale_init_value, init_cfg, input_size)
        self.sr_ratio = sr_ratio
        self.avg = avg
        if self.sr_ratio > 1:
            if not avg:
                self.sr = nn.Conv2d(embed_dims, embed_dims, kernel_size=sr_ratio, stride=sr_ratio, groups=embed_dims)
                self.norm = nn.LayerNorm(embed_dims)
            else:
                self.sr = nn.AvgPool2d(sr_ratio, sr_ratio)
                self.norm = None

        self.q_freqs_cis = None
        self.k_freqs_cis = None

    def forward(self, x, freqs_cis=None):
        if self.sr_ratio == 0:
            return super().forward(x, freqs_cis)
        else:
            B, N, C = x.shape
            H, W = self.input_size
            N_reduce = int(H * W // self.sr_ratio // self.sr_ratio)
            if self.q_freqs_cis is None or N != self.q_freqs_cis.shape[0]:
                self.q_freqs_cis = precompute_freqs_cis_2d_general(self.head_dims, N, W=W, use_cls=False).to(x.device)

            if self.k_freqs_cis is None or N // self.sr_ratio ** 2 != self.k_freqs_cis.shape[0]:
                self.k_freqs_cis = precompute_freqs_cis_2d_general(self.head_dims, N_reduce, W=W, step=self.sr_ratio,
                                                                   bias=1.0 - 1.0 / self.sr_ratio,
                                                                   use_cls=False).to(x.device)
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                      self.head_dims).permute(2, 0, 1, 3, 4)  # B, N, 3, H, Hc -> 3, B, N, H, Hc
            q = qkv[0]  # B N H Hc

            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # B, M, C
                if self.norm is not None:
                    x_ = self.norm(x_)
                qkv = self.qkv(x_).reshape(B, -1, 3, self.num_heads, self.head_dims).permute(2, 0, 1, 3, 4)  # B,M,2C -> B,M, 2, H, Hc -> 2, B, M,  H, Hc
            else:
                qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.head_dims).permute(2, 0, 1, 3, 4)
            k, v = qkv[1], qkv[2]  # B,  M, H, Hc

            q = apply_rotary_emb_single(q, freqs_cis=self.q_freqs_cis)
            k = apply_rotary_emb_single(k, freqs_cis=self.k_freqs_cis)

            attn_drop = self.attn_drop if self.training else 0.
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
            x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

            x = self.proj(x)
            x = self.out_drop(self.gamma1(self.proj_drop(x)))

            if self.v_shortcut:
                x = v.squeeze(1) + x
            return x


class TransformerEncoderLayerRoPEDET(TransformerEncoderLayerRoPE):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 layer_scale_init_value=0.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 ffn_type='llama_mlp',
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 window_size=0,
                 input_size=None
                 ):
        super().__init__(embed_dims, num_heads, feedforward_channels, layer_scale_init_value, drop_rate, attn_drop_rate,
                         drop_path_rate, num_fcs, qkv_bias, ffn_type, act_cfg, norm_cfg, init_cfg)
        self.window_size = window_size
        del self.attn
        self.attn = MultiheadAttentionRoPEDET(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value,
            input_size=input_size
        )

    def forward(self, x, freqs_cis=None):
        B, N, C = x.shape
        if self.window_size == 0:
            x = x + self.attn(self.ln1(x), freqs_cis)
            x = self.ffn(self.ln2(x), identity=x)
            return x
        else:
            assert self.window_size > 0
            H = int(N ** 0.5)
            W = N // H
            shortcut = x
            x = self.ln1(x)
            x = x.reshape(B, H, W, C)
            x, pad_hw = window_partition(x, self.window_size)
            x = x.reshape(-1, self.window_size*self.window_size, C)
            x = self.attn(x, freqs_cis)
            x = x.reshape(-1, self.window_size, self.window_size, C)
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
            x = x.reshape(B, -1, C)
            x = shortcut + x
            x = self.ffn(self.ln2(x), identity=x)
            return x


class TransformerEncoderLayerRoPEDETv1(TransformerEncoderLayerRoPEDET):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 layer_scale_init_value=0.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 ffn_type='llama_mlp',
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 window_size=0,
                 input_size=None,
                 sr_ratio=0,
                 avg=False

    ):
        super().__init__(embed_dims, num_heads, feedforward_channels, layer_scale_init_value, drop_rate, attn_drop_rate,
                         drop_path_rate, num_fcs, qkv_bias, ffn_type, act_cfg, norm_cfg, init_cfg, window_size,
                         input_size)
        del self.attn
        self.attn = MultiheadAttentionRoPEDETv1(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value,
            input_size=input_size,
            sr_ratio=sr_ratio,
            avg = avg
        )



@MODELS.register_module()
class LLaMADET(VisionLLaMA):
    def __init__(self,
                 arch='base',
                 img_size=1024,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 out_type='cls_token',
                 with_cls_token=False,
                 frozen_stages=-1,
                 interpolate_mode='bicubic',
                 layer_scale_init_value=0.,
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 pre_norm=False,
                 init_cfg=None,
                 window_size=0,
                 window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
                 ):
        super().__init__(arch, img_size, patch_size, in_channels, out_indices, drop_rate, drop_path_rate,
                         qkv_bias, norm_cfg, final_norm, out_type, with_cls_token, frozen_stages, interpolate_mode,
                         layer_scale_init_value, patch_cfg, layer_cfgs, pre_norm, init_cfg)

        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers

        for i in range(self.num_layers):
            if i in window_block_indexes:
                token_size = window_size
            else:
                token_size = img_size // patch_size

            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                layer_scale_init_value=layer_scale_init_value,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(token_size, token_size))
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayerRoPEDET(**_layer_cfg))



@MODELS.register_module()
class LLaMADETv1(LLaMADET):
    def __init__(self,
                 arch='base',
                 img_size=1024,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 out_type='cls_token',
                 with_cls_token=False,
                 frozen_stages=-1,
                 interpolate_mode='bicubic',
                 layer_scale_init_value=0.,
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 pre_norm=False,
                 init_cfg=None,
                 window_size=0,
                 window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
                 avg = False
                 ):
        super().__init__(arch, img_size, patch_size, in_channels, out_indices, drop_rate, drop_path_rate, qkv_bias,
                         norm_cfg, final_norm, out_type, with_cls_token, frozen_stages, interpolate_mode,
                         layer_scale_init_value, patch_cfg, layer_cfgs, pre_norm, init_cfg, window_size, window_block_indexes)

        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers

        for i in range(self.num_layers):
            if i in window_block_indexes:
                token_size = window_size
            else:
                token_size = img_size // patch_size

            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                layer_scale_init_value=layer_scale_init_value,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(token_size, token_size),
                sr_ratio=0 if i in window_block_indexes else 4,
                avg = avg
            )
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayerRoPEDETv1(**_layer_cfg))

@MODELS.register_module()
class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_abs_pos(abs_pos, has_cls_token, hw):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode='bicubic',
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions
    of query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode='linear',
        )
        rel_pos_resized = rel_pos_resized.reshape(-1,
                                                  max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords -
                       k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Args:
        attn (Tensor): attention map.
        q (Tensor):
            query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor):
            relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor):
            relative position embeddings (Lw, C) for width axis.
        q_size (Tuple):
            spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple):
            spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', r_q, Rh)
    rel_w = torch.einsum('bhwc,wkc->bhwk', r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] +
            rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)

    return attn


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size,
               window_size, C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=True,
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h,
                                          self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W,
                            -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_cfg=dict(type='GELU'),
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = build_activation_layer(act_cfg)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None,
    ):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_cfg=act_cfg)

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 kernel_size=(16, 16),
                 stride=(16, 16),
                 padding=(0, 0),
                 in_chans=3,
                 embed_dim=768):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


@MODELS.register_module()
class ViT(BaseModule):
    """Vision Transformer with support for patch or hybrid CNN input stage."""

    def __init__(self,
                 img_size=1024,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_path_rate=0.0,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 use_abs_pos=True,
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 window_size=0,
                 window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
                 pretrain_img_size=224,
                 pretrain_use_cls_token=True,
                 init_cfg=None):

        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.init_cfg = init_cfg

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim)

        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size)
            num_positions = (num_patches +
                             1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size))
            for i in range(depth)
        ])

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            self.apply(self._init_weights)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'model' in ckpt:
                _state_dict = ckpt['model']
            self.load_state_dict(_state_dict, False)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token,
                                (x.shape[1], x.shape[2]))

        for blk in self.blocks:
            x = blk(x)

        x = x.permute(0, 3, 1, 2)

        return x
