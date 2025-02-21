#!/usr/bin/env python3
"""
vit-moco-v3 with prompt
"""
import math
import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F

from functools import partial, reduce
from operator import mul
from torch.nn import Conv2d, Dropout
from timm.models.vision_transformer import _cfg

from ..vit_backbones.vit_moco import VisionTransformerMoCo
from ...utils import logging

logger = logging.get_logger("visual_prompt")


class PromptedVisionTransformerMoCo(VisionTransformerMoCo):
    def __init__(self, prompt_config, **kwargs):
        super().__init__(**kwargs)
        self.prompt_config = prompt_config

        if self.prompt_config.DEEP and self.prompt_config.LOCATION not in [
            "prepend",
        ]:
            raise ValueError(
                "Deep-{} is not supported".format(self.prompt_config.LOCATION)
            )

        num_tokens = self.prompt_config.NUM_TOKENS

        self.num_tokens = num_tokens
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        if self.prompt_config.DEEP:
            self.prompt_depth = len(self.blocks)
        else:
            self.prompt_depth = 1

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(
                6.0
                / float(
                    3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim
                )
            )  # noqa

            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, num_tokens, self.embed_dim)
            )
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            if self.prompt_config.DEEP:
                self.deep_prompt_embeddings = nn.Parameter(
                    torch.zeros(len(self.blocks) - 1, num_tokens, self.embed_dim)
                )
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        if self.prompt_config.LOCATION == "prepend":
            # after CLS token, all before image patches
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            x = torch.cat(
                (
                    x[:, :1, :],
                    self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1)),
                    x[:, 1:, :],
                ),
                dim=1,
            )
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        else:
            raise ValueError("Other prompt locations are not supported")

        return x

    def embeddings(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat(
                (cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1
            )
        x = self.pos_drop(x + self.pos_embed)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.blocks.eval()
            self.patch_embed.eval()
            self.pos_drop.eval()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x):
        x = self.incorporate_prompt(x)

        # deep
        if self.prompt_config.DEEP:
            B = x.shape[0]
            num_layers = len(self.blocks)

            for i in range(num_layers):
                if i == 0:
                    x = self.blocks[i](x)
                else:
                    # prepend
                    x = torch.cat(
                        (
                            x[:, :1, :],
                            self.prompt_dropout(
                                self.deep_prompt_embeddings[i - 1].expand(B, -1, -1)
                            ),
                            x[:, (1 + self.num_tokens) :, :],
                        ),
                        dim=1,
                    )
                    x = self.blocks[i](x)
        else:
            # not deep:
            x = self.blocks(x)

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]


class PromptedAdaptiveVisionTransformerMoCo(VisionTransformerMoCo):
    def __init__(self, prompt_config, **kwargs):
        super().__init__(**kwargs)
        self.prompt_config = prompt_config

        if self.prompt_config.DEEP and self.prompt_config.LOCATION not in [
            "prepend",
        ]:
            raise ValueError(
                "Deep-{} is not supported".format(self.prompt_config.LOCATION)
            )

        num_tokens = self.prompt_config.NUM_TOKENS

        self.num_tokens = num_tokens
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        if self.prompt_config.DEEP:
            self.prompt_depth = len(self.blocks)
        else:
            self.prompt_depth = 1

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(
                6.0
                / float(
                    3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim
                )
            )  # noqa

            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, num_tokens, self.embed_dim)
            )

            # xavier_uniform initialization

            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config.DEEP:
                total_d_layer = len(self.blocks) - self.prompt_config.VPT_DEPTH

                if self.prompt_config.VPT_DEPTH > 1:
                    self.deep_prompt_embeddings = nn.Parameter(
                        torch.zeros(len(self.blocks) - 1, num_tokens, self.embed_dim)
                    )

                    # xavier_uniform initialization

                    nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

                self.deep_prompt_embeddings_mlp = PromptMLP(
                    in_features=self.embed_dim,
                    out_features=self.embed_dim,
                    hidden_features=self.prompt_config.HIDDEN_DIM,
                    dropout=self.prompt_config.DROPOUT_MLP,
                    length=num_tokens,
                    activation=self.prompt_config.ACTIVATION,
                    learnable_scale=self.prompt_config.LEARNABLE_SCALE,
                )

                self.deep_prompt_norm = nn.ModuleList(
                    [nn.LayerNorm(self.embed_dim) for _ in range(total_d_layer)]
                )

                if self.prompt_config.CONV:
                    self.deep_prompt_downsample = nn.ModuleList(
                        [
                            PromptDownSample(
                                height=14,
                                width=14,
                                num_tokens=num_tokens,
                                prompt_dim=self.embed_dim,
                                kernel=self.prompt_config.KERNEL,
                                padding=self.prompt_config.PADDING,
                                channelwise=self.prompt_config.CHANNELWISE,
                            )
                            for _ in range(total_d_layer)
                        ]
                    )

                else:
                    self.deep_prompt_downsample = nn.ModuleList(
                        [nn.Linear(196, num_tokens) for _ in range(total_d_layer)]
                    )

                    for i in range(total_d_layer):
                        nn.init.zeros_(self.deep_prompt_downsample[i].weight)

                        nn.init.zeros_(self.deep_prompt_downsample[i].bias)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        if self.prompt_config.LOCATION == "prepend":
            # after CLS token, all before image patches
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            x = torch.cat(
                (
                    x[:, :1, :],
                    self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1)),
                    x[:, 1:, :],
                ),
                dim=1,
            )
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        else:
            raise ValueError("Other prompt locations are not supported")

        return x

    def embeddings(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat(
                (cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1
            )
        x = self.pos_drop(x + self.pos_embed)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.blocks.eval()
            self.patch_embed.eval()
            self.pos_drop.eval()
            self.prompt_dropout.train()
            self.deep_prompt_embeddings_mlp.train()

            for layer in self.deep_prompt_norm:
                layer.train()

            for layer in self.deep_prompt_downsample:
                layer.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x):
        x = self.incorporate_prompt(x)

        # deep
        if self.prompt_config.DEEP:
            B = x.shape[0]
            num_layers = len(self.blocks)

            for i in range(num_layers):
                if i == 0:
                    x = self.blocks[i](x)
                else:
                    # prepend
                    if i < self.prompt_config.VPT_DEPTH:
                        deep_prompt_emb = self.prompt_dropout(
                            self.deep_prompt_embeddings[i - 1].expand(B, -1, -1)
                        )
                    else:
                        x_states = torch.cat(
                            (x[:, :1, :], x[:, (1 + self.num_tokens) :, :]), dim=1
                        )
                        x_states = self.deep_prompt_norm[
                            i - self.prompt_config.VPT_DEPTH
                        ](x_states)[
                            :, 1:, :
                        ]  # (B, num_tokens, hidden_dim)
                        if self.prompt_config.CONV:
                            x_states = x_states.permute(0, 2, 1).reshape(
                                B, self.embed_dim, 14, 14
                            )
                        else:
                            x_states = x_states.permute(0, 2, 1)

                        x_states = self.deep_prompt_downsample[
                            i - self.prompt_config.VPT_DEPTH
                        ](x_states).permute(
                            0, 2, 1
                        )  # (B, num_tokens, hidden_dim)
                        deep_prompt_emb = self.deep_prompt_embeddings_mlp(x_states)

                    x = torch.cat(
                        (
                            x[:, :1, :],
                            deep_prompt_emb,
                            x[:, (1 + self.num_tokens) :, :],
                        ),
                        dim=1,
                    )
                    x = self.blocks[i](x)
        else:
            # not deep:
            x = self.blocks(x)

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]


class PromptMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 16,
        bias: bool = False,
        dropout: float = 0.0,
        length: int = 5,
        activation: str = "relu",
        learnable_scale: bool = False,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.len = length

        non_linearity = nn.Identity()
        if activation == "relu":
            non_linearity = nn.ReLU(inplace=True)
        if activation == "sigmoid":
            non_linearity = nn.Sigmoid()
        elif activation == "attention":
            non_linearity = nn.Softmax(dim=-1)
        elif activation == "gelu":
            non_linearity = nn.GELU()

        self.block = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features, bias=bias),
            non_linearity,
            nn.Linear(self.hidden_features, self.out_features, bias=bias),
        )

        if dropout > 0.0:
            self.block[1].register_forward_hook(
                lambda m, inp, out: F.dropout(out, p=dropout, training=m.training)
            )

        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = 1.0

    def forward(self, x: torch.Tensor):
        bsz = x.size(0)

        out = x
        prompt_out = self.block(out)
        prompt_out = prompt_out * self.scale
        prompt_out = prompt_out + out

        return prompt_out


class ChannelWiseConv2d(nn.Module):
    def __init__(self, kernel_size, padding=0, channel=768, channelwise=True):
        super(ChannelWiseConv2d, self).__init__()

        # The kernel size should be a single 2D convolution kernel shared across all input channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.channelwise = channelwise

        # Define the kernel as a learnable parameter

        if channelwise:
            self.kernel = nn.Parameter(
                torch.randn(1, 1, kernel_size, kernel_size)
            )  # (1, 1, h, w)
        else:
            self.kernel = nn.Parameter(
                torch.randn(channel, 1, kernel_size, kernel_size)
            )  # (C, 1, h, w)

    def forward(self, x):
        # Ensure that the kernel is applied across all input channels
        # We need to expand the kernel to match the number of input channels

        if self.channelwise:
            kernel_expanded = self.kernel.expand(
                x.size(1), 1, self.kernel_size, self.kernel_size
            )  # (C, 1, h, w)
        else:
            kernel_expanded = self.kernel

        # Perform the convolution: using groups=x.size(1) ensures each input channel gets the same kernel
        # The kernel is broadcast across all input channels, and each channel uses the same filter.
        return F.conv2d(x, kernel_expanded, groups=x.size(1), padding=self.padding)


class PromptDownSample(nn.Module):
    def __init__(
        self,
        height,
        width,
        num_tokens,
        prompt_dim,
        kernel=2,
        padding=0,
        channelwise=True,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_tokens = num_tokens
        self.kernel = kernel
        self.padding = padding

        self.conv = ChannelWiseConv2d(
            kernel, padding=padding, channel=prompt_dim, channelwise=channelwise
        )

        # compute H_new * W_new
        input_feat = (height + 2 * padding - kernel + 1) * (
            width + 2 * padding - kernel + 1
        )
        self.linear = nn.Linear(input_feat, num_tokens)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()

        x = self.conv(x)  # (B, C, H_new, W_new)
        x = self.dropout(x)
        x = x.reshape(B, C, -1)  # (B, C, H_new * W_new)
        x = self.linear(x)

        return x


def vit_base(prompt_cfg, **kwargs):
    if prompt_cfg.ADAPTIVE:
        logger.info("Using VAPT")
        model = PromptedAdaptiveVisionTransformerMoCo(
            prompt_cfg,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs
        )
    else:
        logger.info("Using VPT")
        model = PromptedVisionTransformerMoCo(
            prompt_cfg,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs
        )

    model.default_cfg = _cfg()
    return model