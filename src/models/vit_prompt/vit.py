#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage

from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from ...utils import logging

logger = logging.get_logger("visual_prompt")


class PromptedTransformer(Transformer):
    def __init__(self, prompt_config, config, img_size, vis):
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED
        super(PromptedTransformer, self).__init__(config, img_size, vis)

        self.prompt_config = prompt_config
        self.vit_config = config

        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode="fan_out")
        else:
            prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(
                6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim)
            )  # noqa

            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, num_tokens, prompt_dim)
            )
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config.DEEP:  # noqa

                total_d_layer = config.transformer["num_layers"] - 1
                self.deep_prompt_embeddings = nn.Parameter(
                    torch.zeros(total_d_layer, num_tokens, prompt_dim)
                )
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat(
            (
                x[:, :1, :],
                self.prompt_dropout(
                    self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)
                ),
                x[:, 1:, :],
            ),
            dim=1,
        )
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(
                        self.prompt_proj(self.deep_prompt_embeddings[i - 1]).expand(
                            B, -1, -1
                        )
                    )

                    hidden_states = torch.cat(
                        (
                            hidden_states[:, :1, :],
                            deep_prompt_emb,
                            hidden_states[:, (1 + self.num_tokens) :, :],
                        ),
                        dim=1,
                    )

                hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x):
        # this is the default version:
        embedding_output = self.incorporate_prompt(x)

        if self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(embedding_output)
        else:
            encoded, attn_weights = self.encoder(embedding_output)

        return encoded, attn_weights


class PromptedTransformerAdaptive(Transformer):
    def __init__(self, prompt_config, config, img_size, vis):
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED
        super(PromptedTransformerAdaptive, self).__init__(config, img_size, vis)

        self.prompt_config = prompt_config
        self.vit_config = config

        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        if patch_size[0] == 14:
            self.__size = 16
        else:
            self.__size = 14

        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode="fan_out")
        else:
            prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()

        if self.prompt_config.DEEP:
            self.prompt_depth = config.transformer["num_layers"]
        else:
            self.prompt_depth = 1

        self.prompt_dim = prompt_dim

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(
                6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim)
            )  # noqa

            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, num_tokens, prompt_dim)
            )
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config.DEEP:  # noqa

                total_d_layer = config.transformer["num_layers"] - self.prompt_config.VPT_DEPTH

                if self.prompt_config.VPT_DEPTH > 1:
                    self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                        self.prompt_config.VPT_DEPTH - 1, num_tokens, prompt_dim))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

                if self.prompt_config.FEAT_PROJECTOR:
                    if self.prompt_config.SHARE_PROJECTOR:
                        self.deep_prompt_embeddings_mlp = PromptMLP(
                            in_features=prompt_dim,
                            out_features=prompt_dim,
                            hidden_features=self.prompt_config.HIDDEN_DIM,
                            dropout=self.prompt_config.DROPOUT_MLP,
                            length=num_tokens,
                            activation=self.prompt_config.ACTIVATION,
                            learnable_scale=self.prompt_config.LEARNABLE_SCALE,
                        )
                    else:
                        self.deep_prompt_embeddings_mlp = nn.ModuleList(
                            [
                                PromptMLP(
                                    in_features=prompt_dim,
                                    out_features=prompt_dim,
                                    hidden_features=self.prompt_config.HIDDEN_DIM,
                                    dropout=self.prompt_config.DROPOUT_MLP,
                                    length=num_tokens,
                                    activation=self.prompt_config.ACTIVATION,
                                    learnable_scale=self.prompt_config.LEARNABLE_SCALE,
                                )
                                for _ in range(total_d_layer)
                            ]
                        )
                else:
                    self.deep_prompt_embeddings_mlp = nn.Identity()
                    

                self.deep_prompt_norm = nn.ModuleList(
                    [nn.LayerNorm(prompt_dim) for _ in range(total_d_layer)]
                )

                if self.prompt_config.CONV:
                    self.deep_prompt_downsample = nn.ModuleList(
                        [
                            PromptDownSample(
                                height=self.__size,
                                width=self.__size,
                                num_tokens=num_tokens,
                                prompt_dim=prompt_dim,
                                kernel=self.prompt_config.KERNEL,
                                padding=self.prompt_config.PADDING,
                                channelwise=self.prompt_config.CHANNELWISE,
                            )
                            for _ in range(total_d_layer)
                        ]
                    )
                else:
                    self.deep_prompt_downsample = nn.ModuleList(
                        [nn.Linear(self.__size * self.__size, num_tokens) for _ in range(total_d_layer)]
                    )

                    for i in range(total_d_layer):
                        nn.init.zeros_(self.deep_prompt_downsample[i].weight)
                        nn.init.zeros_(self.deep_prompt_downsample[i].bias)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)

        x = torch.cat(
            (
                x[:, :1, :],
                self.prompt_dropout(
                    self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)
                ),
                x[:, 1:, :],
            ),
            dim=1,
        )
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
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

    def forward_deep_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                if i < self.prompt_depth:

                    if i < self.prompt_config.VPT_DEPTH:
                        deep_prompt_emb = self.prompt_dropout(
                            self.prompt_proj(self.deep_prompt_embeddings[i - 1]).expand(
                                B, -1, -1))
                        hidden_states = torch.cat(
                            (
                                hidden_states[:, :1, :],
                                deep_prompt_emb,
                                hidden_states[:, (1 + self.num_tokens) :, :],
                            ),
                            dim=1,
                        )
                    else:       
                        x_states = torch.cat(
                            (
                                hidden_states[:, :1, :],
                                hidden_states[:, (1 + self.num_tokens) :, :],
                            ),
                            dim=1,
                        )
                        x_states = self.deep_prompt_norm[i - self.prompt_config.VPT_DEPTH](x_states)[:, 1:, :]  # (B, num_tokens, hidden_dim)
                        if self.prompt_config.CONV:
                            x_states = x_states.permute(0, 2, 1).reshape(
                                B, self.prompt_dim, self.__size, self.__size
                            )
                        else:
                            x_states = x_states.permute(0, 2, 1)

                        x_states = self.deep_prompt_downsample[i - self.prompt_config.VPT_DEPTH](x_states).permute(
                            0, 2, 1
                        )  # (B, num_tokens, hidden_dim)

                        if self.prompt_config.SHARE_PROJECTOR:
                            deep_prompt_emb = self.deep_prompt_embeddings_mlp(x_states)
                        else:
                            deep_prompt_emb = self.deep_prompt_embeddings_mlp[i - self.prompt_config.VPT_DEPTH](x_states)

                        hidden_states = torch.cat(
                            (
                                hidden_states[:, :1, :],
                                deep_prompt_emb,
                                hidden_states[:, (1 + self.num_tokens) :, :],
                            ),
                            dim=1,
                        )

                hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x):
        # this is the default version:
        embedding_output = self.incorporate_prompt(x)

        if self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(embedding_output)
        else:
            encoded, attn_weights = self.encoder(embedding_output)

        return encoded, attn_weights


class PromptedVisionTransformer(VisionTransformer):
    def __init__(
        self, prompt_cfg, model_type, img_size=224, num_classes=21843, vis=False
    ):
        assert prompt_cfg.VIT_POOL_TYPE == "original"
        super(PromptedVisionTransformer, self).__init__(
            model_type, img_size, num_classes, vis
        )
        if prompt_cfg is None:
            raise ValueError(
                "prompt_cfg cannot be None if using PromptedVisionTransformer"
            )
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type]
        if prompt_cfg.ADAPTIVE:
            self.transformer = PromptedTransformerAdaptive(
                prompt_cfg, vit_cfg, img_size, vis
            )
        else:
            self.transformer = PromptedTransformer(prompt_cfg, vit_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)

        x = x[:, 0]

        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights


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
        self, height, width, num_tokens, prompt_dim, kernel=2, padding=0, channelwise=True
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_tokens = num_tokens
        self.kernel = kernel
        self.padding = padding

        self.conv = ChannelWiseConv2d(kernel, padding=padding, channel=prompt_dim, channelwise=channelwise)

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
