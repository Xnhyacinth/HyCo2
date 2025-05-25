# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import pdb
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_


class IdentityMap(nn.Module):
    def __init__(self, hiiden, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": "identity"}


# def get_abs_pos(abs_pos, tgt_size):
#     src_size = int(math.sqrt(abs_pos.size(0)))
#     dtype = abs_pos.dtype

#     return F.interpolate(
#         abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
#         size=(tgt_size[0], tgt_size[1]),
#         mode="bicubic",
#         align_corners=False,
#     ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)


def get_abs_pos(abs_pos, tgt_length):
    """
    Interpolate absolute positional embeddings for 1D sequences (e.g., text).

    Args:
        abs_pos: Tensor of shape (src_len, embed_dim), the original positional embeddings.
        tgt_length: Target sequence length (int), the desired interpolated length.

    Returns:
        A tensor of shape (tgt_length, embed_dim), interpolated positional embeddings.
    """
    # Original source sequence length and embedding dimension
    src_len, embed_dim = abs_pos.shape
    dtype = abs_pos.dtype

    # Reshape positional embeddings for interpolation (1D -> 3D for compatibility with interpolate)
    abs_pos = abs_pos.permute(1, 0).unsqueeze(0)  # Shape: (1, embed_dim, src_len)

    # Apply 1D interpolation
    interpolated_pos = F.interpolate(
        abs_pos,
        size=(tgt_length,),  # Target sequence length
        mode="linear",
        align_corners=False,
    )

    # Reshape back to original format
    interpolated_pos = interpolated_pos.squeeze(0).permute(
        1, 0
    )  # Shape: (tgt_length, embed_dim)

    return interpolated_pos.to(dtype=dtype)


def get_1d_sincos_pos_embed(embed_dim, seq_length):
    """
    seq_length: int of the sequence length
    return:
    pos_embed: [seq_length, embed_dim]
    """
    pos = np.arange(seq_length, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_length(embed_dim, pos)


def get_1d_sincos_pos_embed_from_length(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class TextResamplerWithAttention(nn.Module):
    """
    A perceiver-resampler network for NLP tasks with one cross-attention layer.
    This module processes text sequences using learnable queries and supports
    concatenation of textual embeddings for cross-attention.

    Inputs:
        - `x`: Input tensor of shape (batch_size, seq_len, kv_dim), textual embeddings.
        - `text`: Additional optional tensor of shape (batch_size, seq_len, hidden_dim).
        - `attn_mask`: Optional attention mask of shape (batch_size, seq_len).
    Outputs:
        - A tensor of shape (batch_size, seq_len, embed_dim).
    """

    def __init__(
        self,
        num_queries,  # Number of learnable queries
        embed_dim,  # Embedding dimension of the model
        num_heads,  # Number of attention heads
        kv_dim=None,  # Key-value input dimension, if different from embed_dim
        llm_hidden_size=4096,  # Optional hidden size for LLMs
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # Normalization layer
        use_post_proj=False,  # Whether to include a post-projection layer
        use_pos=False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_pos = use_pos

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_1d_sincos_pos_embed(embed_dim, num_queries)).to(
                torch.bfloat16
            )
        ).requires_grad_(False)

        # Learnable query embeddings
        self.query = nn.Parameter(
            torch.zeros(num_queries, embed_dim), requires_grad=True
        )
        trunc_normal_(self.query, std=0.02)

        # Optional key-value projection for the text input
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        # Multi-head self-attention and cross-attention layers
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

        # Normalization layers
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.ln_post = norm_layer(embed_dim)

        # Optional post-projection to LLM hidden size
        if use_post_proj:
            self.proj = nn.Linear(embed_dim, llm_hidden_size)
        else:
            self.proj = nn.Identity()

    def forward(self, x, attn_mask=None, text=None, text_attn_mask=None):
        """
        Forward pass for TextResamplerWithAttention.

        Args:
            x: (batch_size, seq_len, kv_dim), input token embeddings.
            text: Optional, tensor of shape (batch_size, seq_len, kv_dim) representing external embeddings.
            attn_mask: Optional, attention mask (batch_size, seq_len).

        Returns:
            A tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Project text embeddings to `embed_dim` if required
        if text is not None:
            text = self.kv_proj(text)
            text = self.ln_kv(text)
        else:
            # If no text is provided, treat as an empty tensor
            text = torch.zeros(
                (batch_size, 0, self.embed_dim), device=x.device, dtype=x.dtype
            )

        # Combine learnable queries with textual embeddings
        query = self._repeat(
            self.query, batch_size
        )  # Shape: (num_queries, batch_size, embed_dim)
        combined_input = torch.cat(
            [query, text.transpose(0, 1)], dim=0
        )  # (num_queries + seq_len, batch_size, embed_dim)

        # Create combined attention mask
        if text_attn_mask is not None:
            # Expand attention mask for queries and concatenate with text mask
            query_mask = torch.ones(
                (batch_size, self.num_queries),
                device=text_attn_mask.device,
                dtype=text_attn_mask.dtype,
            )
            combined_mask = torch.cat(
                [query_mask, text_attn_mask], dim=1
            ).bool()  # (batch_size, num_queries + seq_len)
            # contate_attn_mask = torch.cat([torch.zeros((N, self.num_queries), dtype=attn_mask.dtype, device=attn_mask.device) , ~attn_mask], dim=-1).bool()
        else:
            combined_mask = None

        # Apply self-attention to combined queries and text embeddings
        combined_output = self.self_attn(
            combined_input,
            combined_input,
            combined_input,
            key_padding_mask=~combined_mask if combined_mask is not None else None,
        )[
            0
        ]  # Shape: (num_queries + seq_len, batch_size, embed_dim)

        # Extract queries after self-attention
        query = combined_output[
            : self.num_queries
        ]  # Shape: (num_queries, batch_size, embed_dim)
        query = self.ln_q(query)
        # breakpoint()

        # breakpoint()
        # Apply cross-attention between queries and x (input embeddings)
        x = x.permute(1, 0, 2)  # Transpose to (seq_len, batch_size, embed_dim)
        if self.use_pos:
            pos_embed = get_abs_pos(self.pos_embed.detach(), seq_len).detach()
            out = self.attn(
                query
                + self.pos_embed.unsqueeze(
                    1
                ).detach(),  # Query: (num_queries, batch_size, embed_dim)
                x + pos_embed.unsqueeze(1),  # Key: (seq_len, batch_size, embed_dim)
                x,  # Value: (seq_len, batch_size, embed_dim)
                key_padding_mask=~attn_mask.bool(),
            )[
                0
            ]  # Output: (num_queries, batch_size, embed_dim)
        else:
            out = self.attn(
                query,  # Query: (num_queries, batch_size, embed_dim)
                x,  # Key: (seq_len, batch_size, embed_dim)
                x,  # Value: (seq_len, batch_size, embed_dim)
                key_padding_mask=~attn_mask.bool(),
            )[0]

        # Transpose back and normalize output
        out = out.permute(1, 0, 2)  # (batch_size, num_queries, embed_dim)
        out = self.ln_post(out)
        # Apply optional post-projection layer
        out = self.proj(out)
        return out

    def _repeat(self, query, batch_size):
        """
        Repeat the learnable queries for each batch.
        """
        return query.unsqueeze(1).repeat(
            1, batch_size, 1
        )  # (num_queries, batch_size, embed_dim)


class TextResampler(nn.Module):
    """
    A text perceiver - resampler network with one cross - attention layer by
        (num_queries) learnable queries and 1d sincos pos_emb
    Outputs:
        A tensor with the shape of (num_queries, embed_dim)
    """

    def __init__(
        self,
        num_queries,
        embed_dim,
        num_heads,
        kv_dim=None,
        llm_hidden_size=4096,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_post_proj=False,
        use_pos=False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_pos = use_pos

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_1d_sincos_pos_embed(embed_dim, num_queries)).to(
                torch.bfloat16
            )
        ).requires_grad_(False)

        self.query = nn.Parameter(
            torch.zeros(self.num_queries, embed_dim), requires_grad=True
        )
        trunc_normal_(self.query, std=0.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)

        if use_post_proj:
            self.proj = nn.Linear(embed_dim, llm_hidden_size)
        else:
            self.proj = nn.Identity()

    def forward(self, x, attn_mask=None, text=None, text_attn_mask=None):
        if len(x.shape) <= 2:
            x = x.unsqueeze(0)
            mark = True
        else:
            mark = False
        batch_size, seq_len, _ = x.shape
        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)

        query = self.ln_q(self._repeat(self.query, batch_size))
        # breakpoint()
        if not self.use_pos:
            out = self.attn(query, x, x, key_padding_mask=~attn_mask.bool())[0]
        else:
            pos_embed = get_abs_pos(self.pos_embed.detach(), seq_len)
            out = self.attn(
                query
                + self.pos_embed.unsqueeze(
                    1
                ).detach(),  # Query: (num_queries, batch_size, embed_dim)
                x
                + pos_embed.unsqueeze(
                    1
                ).detach(),  # Key: (seq_len, batch_size, embed_dim)
                x,  # Value: (seq_len, batch_size, embed_dim)
                key_padding_mask=~attn_mask.bool(),
            )[0]
        # breakpoint()
        x = out.permute(1, 0, 2)

        x = self.ln_post(x)
        x = self.proj(x)
        return x if not mark else x.squeeze()

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
        self,
        grid_size,
        embed_dim,
        num_heads,
        kv_dim=None,
        llm_hidden_size=4096,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_post_proj=False,
    ):
        super().__init__()
        self.num_queries = grid_size**2
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).to(
                torch.float16
            )
        ).requires_grad_(False)

        self.query = nn.Parameter(
            torch.zeros(self.num_queries, embed_dim), requires_grad=True
        )
        trunc_normal_(self.query, std=0.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)

        if use_post_proj:
            self.proj = nn.Linear(embed_dim, llm_hidden_size)
        else:
            self.proj = nn.Identity()

    def forward(self, x, tgt_size=(24, 24), text=None, attn_mask=None):
        if len(x.shape) <= 2:
            x = x.unsqueeze(0)
            mark = True
        else:
            mark = False
        if x.shape[1] != tgt_size[0] * tgt_size[1]:
            tgt_size = (int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])))

        pos_embed = get_abs_pos(self.pos_embed.detach(), tgt_size).detach()
        if torch.isnan(self.pos_embed).any():
            # some init error
            self.pos_embed = nn.Parameter(
                torch.from_numpy(
                    get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
                )
                .to(torch.float16)
                .to(x.device)
            ).requires_grad_(False)
        pos_embed = get_abs_pos(self.pos_embed.detach(), tgt_size).detach()

        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)

        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1).detach(),
            x + pos_embed.unsqueeze(1),
            x,
        )[0]
        x = out.permute(1, 0, 2)

        x = self.ln_post(x)
        x = self.proj(x)
        return x if not mark else x.squeeze()

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class ResamplerWithText(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
        self,
        grid_size,
        embed_dim,
        num_heads,
        kv_dim=None,
        llm_hidden_size=4096,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_post_proj=False,
    ):
        super().__init__()
        self.num_queries = grid_size**2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size))
            .float()
            .to(torch.float16)
        ).requires_grad_(False)

        self.query = nn.Parameter(
            torch.zeros(self.num_queries, embed_dim), requires_grad=True
        )
        trunc_normal_(self.query, std=0.02)

        if llm_hidden_size is not None and llm_hidden_size != embed_dim:
            self.kv_proj = nn.Linear(llm_hidden_size, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)

        if use_post_proj:
            self.proj = nn.Linear(embed_dim, llm_hidden_size)
        else:
            self.proj = nn.Identity()

    def forward(self, x, tgt_size=(24, 24), text=None, attn_mask=None):
        if len(x.shape) <= 2:
            x = x.unsqueeze(0)
        if len(text.shape) <= 2:
            text = text.unsqueeze(0)
            attn_mask = attn_mask.unsqueeze(0)
        if x.shape[1] != tgt_size[0] * tgt_size[1]:
            tgt_size = (int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])))

        pos_embed = get_abs_pos(self.pos_embed.detach(), tgt_size).detach()

        N = x.shape[0]

        text = self.kv_proj(text)
        text = self.ln_kv(text)

        text, x = text.permute(1, 0, 2), x.permute(1, 0, 2)

        query = self._repeat(self.query, N)

        contate_query_text = torch.cat([query, text], dim=0)
        contate_attn_mask = torch.cat(
            [
                torch.zeros(
                    (N, self.num_queries),
                    dtype=attn_mask.dtype,
                    device=attn_mask.device,
                ),
                ~attn_mask,
            ],
            dim=-1,
        ).bool()
        contate_query_text = self.self_attn(
            contate_query_text,
            contate_query_text,
            contate_query_text,
            key_padding_mask=contate_attn_mask,
        )[0]

        query = contate_query_text[: self.query.shape[0]]
        query = self.ln_q(query)

        out = self.attn(
            query + self.pos_embed.unsqueeze(1).detach(),
            x + pos_embed.unsqueeze(1).detach(),
            x,
        )[0]
        x = out.permute(1, 0, 2)

        x = self.ln_post(x)
        x = self.proj(x)
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)
