import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.init import trunc_normal_

from src.model.multimodal_projector.moe import SparseDispatcher
from src.model.multimodal_resampler.sampler import (
    Resampler,
    ResamplerWithText,
    TextResampler,
    TextResamplerWithAttention,
)


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class GatedBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        if "attn" in config.projector_type:
            self.attn = TextResamplerWithAttention(  # TextResamplerWithAttention
                num_queries=config.num_query_tokens,
                embed_dim=config.retriever_hidden_size,
                num_heads=config.num_attention_heads,
                kv_dim=config.retriever_hidden_size,
                llm_hidden_size=config.hidden_size,
                use_post_proj=False,
                use_pos="pos" in config.projector_type,
            )
        elif "adapool" not in config.projector_type and "adagen" not in config.projector_type:
            self.attn = TextResampler(  # TextResamplerWithAttention
                num_queries=config.num_query_tokens,
                embed_dim=config.retriever_hidden_size,
                num_heads=config.num_attention_heads,
                kv_dim=config.retriever_hidden_size,
                llm_hidden_size=config.hidden_size,
                use_post_proj=False,
                use_pos="pos" in config.projector_type,
            )
        self.num_queries = config.num_query_tokens
        self.config = config

        # Feed-forward projection module
        modules = [nn.Linear(config.retriever_hidden_size, config.hidden_size)]
        for _ in range(1, 2):  # 可以根据需要修改层数
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.projection = nn.Sequential(*modules)

        self.sampler = nn.AdaptiveAvgPool2d(
            (config.num_query_tokens, config.hidden_size)
        )

        if "proj" in config.projector_type:
            modules = [nn.Linear(config.retriever_hidden_size, config.hidden_size)]
            for _ in range(1, 2):  # 可以根据需要修改层数
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))

            if 'fproj' in config.projector_type:
                self.fprojection = nn.Sequential(*modules)
            else:
                self.projector = nn.Sequential(*modules)

        print(f"Method: {config.projector_type}")
        # Mixture of Experts
        if "adapool" in config.projector_type or 'adagen' in config.projector_type:
            self.expert_ffn = [self.projection]
            self.num_experts = len(self.expert_ffn)
        else:
            self.expert_ffn = [self.projection, self.attn]  # 包含投影层和 TextResampler
            self.num_experts = len(self.expert_ffn)
            self.w_gate = nn.Parameter(
                torch.zeros(
                    config.retriever_hidden_size, self.num_experts, dtype=torch.bfloat16
                ),
                requires_grad=True,
            )
            self.w_noise = nn.Parameter(
                torch.zeros(
                    config.retriever_hidden_size, self.num_experts, dtype=torch.bfloat16
                ),
                requires_grad=True,
            )

            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(1)

            self.register_buffer("mean", torch.tensor([0.0], dtype=torch.bfloat16))
            self.register_buffer("std", torch.tensor([1.0], dtype=torch.bfloat16))

            self.learnable_gated = config.learnable_gated  # 控制 gated 模式
            self.k = 2

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """
        Helper function to calculate probability of being in the top-k values, for noisy gates.
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )

        normal = Normal(
            self.mean.to(clean_values.device), self.std.to(clean_values.device)
        )
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def _gates_to_load(self, gates):
        """
        Compute true load per expert, given the gates.
        """
        return (gates > 0).sum(0)

    def cv_squared(self, x):
        """
        Squared coefficient of variation for load balancing loss.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def noisy_top_k_gating(self, x, noise_epsilon=1e-2):
        """
        Noisy top-k gating.
        """
        clean_logits = x @ self.w_gate.to(x.dtype)
        if self.training:
            raw_noise_stddev = x @ self.w_gate
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        # Calculate top-k gating
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.k < self.num_experts and self.training:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, attn_mask=None, text_embedding=None, text_attn_mask=None):
        """
        Forward method for GatedBlock.
        """
        # if x.shape[1] != self.target_sequence_length:
        #     return self.projection(x)

        if len(x.shape) <= 2:
            x = x.unsqueeze(0)
            mark = True
        else:
            mark = False

        N, C, D = x.shape

        # Collect expert outputs
        if "adagen" in self.config.projector_type:
            # x_sampler = self.attn(x, attn_mask, text_embedding, text_attn_mask)
            chunks = x.split(100, dim=1)
            final_representations = []
            if 'avg' in self.config.projector_type:
                sampler = nn.AdaptiveAvgPool2d(
                    (4, x.shape[-1])
                )
                for chunk in chunks:
                    avg_tokens = sampler(chunk)  # 提取最后两个 tokens（从 chunk 的第二轴）
                    final_representations.append(avg_tokens)
            else:
                for chunk in chunks:
                    last_two_tokens = chunk[:, -4:, :]  # 提取最后两个 tokens（从 chunk 的第二轴）
                    final_representations.append(last_two_tokens) 

            # 拼接所有 chunks 的表示
            x_sampler = torch.cat(final_representations, dim=1)
            # pass
        else:
            x_sampler = self.sampler(x)
        if self.num_experts == 1:
            return self.projection(x_sampler).view(-1, D)

        expert_outputs = [self.projection(x_sampler)]
        for i in range(1, self.num_experts):
            expert_output = self.expert_ffn[i](
                x, attn_mask, text_embedding, text_attn_mask
            )  # , attn_mask, text_embedding, text_attn_mask
            if "fproj" in self.config.projector_type:
                expert_output = self.fprojection(expert_output)
            elif "proj" in self.config.projector_type:
                expert_output = self.projector(expert_output)
            else:
                expert_output = self.projection(expert_output)
            expert_outputs.append(expert_output)

        if self.learnable_gated >= 0:
            # Use a specific expert based on learnable gating
            if mark:
                return expert_outputs[self.learnable_gated].squeeze(0)
            return expert_outputs[self.learnable_gated]

        if "cat" in self.config.projector_type:
            output = torch.cat(expert_outputs, dim=1)
        else:
            gates, load = self.noisy_top_k_gating(
                x_sampler.reshape(N * self.num_queries, D)
            )
            gates = gates.reshape(N, self.num_queries, -1)
            output = torch.stack(expert_outputs, dim=-1)
            # breakpoint()
            output = torch.matmul(output, gates.unsqueeze(-1)).squeeze(-1)
        if mark:
            output = output.squeeze(0)
        return output.view(-1, D)


class GatedBlock0(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.target_sequence_length = 576
        grid_size = int(math.sqrt(self.target_sequence_length))
        self.attn = Resampler(
            grid_size=grid_size,
            embed_dim=config.mm_hidden_size,
            num_heads=config.mm_hidden_size // 128,
            kv_dim=config.mm_hidden_size,
            llm_hidden_size=config.hidden_size,
            use_post_proj=False,
        )

        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.projection = nn.Sequential(*modules)

        # Mixture of Experts
        self.expert_ffn = [self.projection, self.attn]  #
        self.num_experts = len(self.expert_ffn)

        self.w_gate = nn.Parameter(
            torch.zeros(config.mm_hidden_size, self.num_experts, dtype=torch.bfloat16),
            requires_grad=True,
        )
        self.w_noise = nn.Parameter(
            torch.zeros(config.mm_hidden_size, self.num_experts, dtype=torch.bfloat16),
            requires_grad=True,
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        self.register_buffer("mean", torch.tensor([0.0], dtype=torch.bfloat16))
        self.register_buffer("std", torch.tensor([1.0], dtype=torch.bfloat16))

        self.learnable_gated = config.mm_learnable_gated
        self.k = 2
        # self.apply(self._init_weights)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(
            self.mean.to(clean_values.device), self.std.to(clean_values.device)
        )
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def noisy_top_k_gating(self, x, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input Tensor with shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a Tensor with shape [batch_size, num_experts]
          load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate.to(x.dtype)
        if self.training:
            raw_noise_stddev = x @ self.w_gate
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (
            top_k_logits.sum(1, keepdim=True) + 1e-6
        )  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.k < self.num_experts and self.training:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, text_embedding=None, attn_mask=None):
        if (
            x.shape[0] != self.target_sequence_length
            and x.shape[1] != self.target_sequence_length
        ):
            return self.projection(x)

        if len(x.shape) <= 2:
            x = x.unsqueeze(0)
            mark = True
        else:
            mark = False

        N, C, D = x.shape

        expert_outputs = [self.projection(x)]

        for i in range(1, self.num_experts):
            expert_output = self.expert_ffn[i](x)
            expert_output = self.projection(expert_output)
            expert_outputs.append(expert_output)

        if self.learnable_gated >= 0:
            if mark:
                return expert_outputs[self.learnable_gated].squeeze(0)
            return expert_outputs[self.learnable_gated]

        gates, load = self.noisy_top_k_gating(x.reshape(N * C, D))
        gates = gates.reshape(N, C, -1)
        output = torch.stack(expert_outputs, dim=-1)
        output = torch.matmul(output, gates.unsqueeze(-1)).squeeze(-1)
        if mark:
            output = output.squeeze(0)
        return output


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "projector_type", "linear")

    if "linear" in projector_type:
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif "qformer" in projector_type:
        target_sequence_length = 576
        grid_size = int(math.sqrt(target_sequence_length))
        resampler = Resampler(
            grid_size=grid_size,
            embed_dim=config.mm_hidden_size,
            num_heads=config.mm_hidden_size // 128,
            kv_dim=config.mm_hidden_size,
            llm_hidden_size=config.hidden_size,
        )
        return resampler
    elif "qformer_text" in projector_type:
        target_sequence_length = 576
        grid_size = int(math.sqrt(target_sequence_length))
        resampler = ResamplerWithText(
            grid_size=grid_size,
            embed_dim=config.mm_hidden_size,
            num_heads=config.mm_hidden_size // 128,
            kv_dim=config.mm_hidden_size,
            llm_hidden_size=config.hidden_size,
        )
        return resampler
    elif "gated" in projector_type:
        return GatedBlock(config)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
