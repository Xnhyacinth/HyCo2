import re
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import MixtralConfig, MixtralForCausalLM


class XMixtralConfig(MixtralConfig):
    def __init__(
        self,
        projector_type="mlp2x_gelu",
        retriever_hidden_size=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.retriever_hidden_size = retriever_hidden_size


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        projector_type = config.projector_type
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.retriever_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.projector = nn.Sequential(*modules)

    def forward(self, context_embedding):
        return self.projector(context_embedding)


## compatible with normal Mixtral model
class XMixtralForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        if (
            hasattr(config, "retriever_hidden_size")
            and config.retriever_hidden_size > 0
        ):
            self.projector = Projector(config)
            self.retriever_hidden_size = config.retriever_hidden_size
        self.post_init()

    def set_xrag_token_id(self, token_id):
        self.xrag_token_id = token_id

    def prepare_inputs_embeds(self, input_ids, retrieval_embeds):
        inputs_embeds = self.model.embed_tokens(input_ids)
        retrieval_embeds = retrieval_embeds.view(-1, self.retriever_hidden_size)

        ## sanity check
        num_xrag_tokens = torch.sum(input_ids == self.xrag_token_id).item()
        num_retrieval_embeds = retrieval_embeds.shape[0]
        assert num_xrag_tokens == num_retrieval_embeds, (
            num_xrag_tokens,
            num_retrieval_embeds,
        )

        retrieval_embeds = self.projector(retrieval_embeds.to(inputs_embeds.dtype)).to(
            retrieval_embeds.device
        )
        inputs_embeds[input_ids == self.xrag_token_id] = retrieval_embeds

        return inputs_embeds

    def forward(
        self,
        input_ids=None,
        retrieval_embeds=None,  ## [-1,retrieval_hidden_size]
        attention_mask=None,
        **kwargs,
    ):
        ## when inputs_embeds is passed, it means the model is doing generation
        ## and only the first round of generation would pass inputs_embeds
        ## https://github.com/huggingface/transformers/blob/79132d4cfe42eca5812e8c45ea1b075f04f907b6/src/transformers/models/llama/modeling_llama.py#L1250
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        at_the_beginning_of_generation = False
        if inputs_embeds is not None:
            assert not self.training
            assert retrieval_embeds is None
            at_the_beginning_of_generation = True

        if not at_the_beginning_of_generation:
            ## a single forward
            if retrieval_embeds is not None:
                inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
                input_ids = None
                if attention_mask is not None:
                    assert inputs_embeds.shape[1] == attention_mask.shape[1], (
                        inputs_embeds.shape,
                        attention_mask.shape,
                    )

        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        retrieval_embeds=None,
        **kwargs,
    ):
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported for generate")

        inputs_embeds = None
        if retrieval_embeds is not None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
            input_ids = None
            if attention_mask is not None:
                assert inputs_embeds.shape[1] == attention_mask.shape[1], (
                    inputs_embeds.shape,
                    attention_mask.shape,
                )
            return super().generate(
                attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
            )

        else:
            return super().generate(
                attention_mask=attention_mask, input_ids=input_ids, **kwargs
            )
