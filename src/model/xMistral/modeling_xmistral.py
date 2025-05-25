import re
from typing import Optional, Union

import torch
import torch.nn as nn
from peft import (LoraConfig, PromptTuningConfig, PromptTuningInit, TaskType,
                  get_peft_config, get_peft_model)
from transformers import (AutoConfig, AutoModel, AutoTokenizer, MistralConfig,
                          MistralForCausalLM)
from transformers.models.roberta.configuration_roberta import RobertaConfig

from src.model.multimodal_projector.builder import build_vision_projector
from src.model.multimodal_resampler.builder import build_vision_sampler

from .Qformer import BertLMHeadModel
from .Qformer_roberta import RobertaForCausalLM, RobertaforConfig

AutoConfig.register("robertafor", RobertaforConfig)
AutoModel.register(RobertaforConfig, RobertaForCausalLM)


class XMistralConfig(MistralConfig):
    def __init__(
        self,
        projector_type="mlp2x_gelu",
        retriever_hidden_size=0,
        num_query_tokens=0,
        cformer_model_name_or_path=None,
        cformer_model_finetuning_type=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.retriever_hidden_size = retriever_hidden_size
        self.num_query_tokens = num_query_tokens
        self.cformer_model_name_or_path = cformer_model_name_or_path
        self.cformer_model_finetuning_type = cformer_model_finetuning_type


class CFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.args = config
        self.num_query_tokens = config.num_query_tokens
        self.roberta_config = AutoConfig.from_pretrained(
            config.cformer_model_name_or_path
        )
        if hasattr(config, "projector_vocab_size"):
            print("Resizeing the vocab of projector!")
            self.roberta_config.vocab_size = config.projector_vocab_size
        self.cformer_text_input = True
        # self.encoder_tokenizer = AutoTokenizer.from_pretrained(config.cformer_model_name_or_path, use_fast=False)
        # if "inter" in self.strategy:
        #     self.query_embeds_generator = QueryEmbedsGenerator(self.num_query_tokens)

        # if self.strategy[:2] == "v3":
        self.roberta_config.add_cross_attention = True
        self.roberta_config.query_length = self.num_query_tokens
        self.roberta_config.encoder_width = config.hidden_size
        self.roberta_config.cross_attn_start_layer = (
            self.roberta_config.num_hidden_layers
            - self.roberta_config.num_hidden_layers
        )
        self.roberta_config.is_decoder = True
        # self.roberta_config.model_type='robertafor'
        # self.roberta_config.cross_attention_freq = 2
        # self.model = AutoModel.from_config(self.roberta_config)
        self.model = RobertaForCausalLM(self.roberta_config)
        # self.model = RobertaForCausalLM.from_pretrained(
        # # self.model = BertLMHeadModel.from_pretrained(
        #     config.cformer_model_name_or_path, config=self.roberta_config,
        # )

        if self.args.cformer_model_finetuning_type == "full":
            self.model.requires_grad_(True)
        elif self.args.cformer_model_finetuning_type == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=config.cformer_target_modules.split(","),
                r=config.cformer_r,
                lora_alpha=2 * config.cformer_r,
                lora_dropout=0,
            )
            self.model = get_peft_model(self.model, peft_config)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.query_token_embeds = nn.Parameter(
            torch.zeros(self.num_query_tokens, self.roberta_config.hidden_size)
        )
        self.query_token_embeds.data.normal_(
            mean=0.0, std=self.roberta_config.initializer_range
        )
        # self.itm_head = nn.Linear(self.roberta_config.hidden_size, 2)

        if self.num_query_tokens > 0:
            self.projector = nn.Linear(
                self.roberta_config.hidden_size,
                config.hidden_size,
                dtype=self.query_token_embeds.dtype,
            )
            self.ln_norm = nn.LayerNorm(
                config.hidden_size, dtype=self.query_token_embeds.dtype
            )

    def init_tokenizer_and_embeds(self, encoder_tokenizer):
        # self.encoder_tokenizer.add_tokens(["<xRAG>"], special_tokens=True)
        # self.encoder_tokenizer.gen_token_id = self.encoder_tokenizer.convert_tokens_to_ids("<xRAG>")

        self.model.resize_token_embeddings(len(encoder_tokenizer))

    @property
    def device(self):
        return self.decoder.device

    def gen_query_embeds_pt(self, qformer_inputs):
        input_ids = qformer_inputs["input_ids"]
        batch_size = input_ids.shape[0]
        query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        return query_embeds

    def gen_query_embeds(self, text_ids, text_atts, doc_embeds, doc_attention_mask):
        # text_ids, text_atts = qformer_inputs["input_ids"], qformer_inputs["attention_mask"]
        bs = text_ids.shape[0]
        query_tokens = self.query_token_embeds.unsqueeze(0).expand(bs, -1, -1)
        # doc_embeds = self.model.get_input_embeddings()(doc_ids)
        # doc_atts = torch.ones(doc_embeds.size()[:-1], dtype=query_tokens.dtype).to(query_tokens.device)

        query_atts = torch.ones(query_tokens.shape[:-1]).to(query_tokens.device)
        attention_mask = torch.cat([query_atts, text_atts], dim=1)

        if self.cformer_text_input:
            query_output = self.model.roberta(
                text_ids,
                attention_mask=attention_mask,
                query_embeds=query_tokens,
                encoder_hidden_states=doc_embeds,
                encoder_attention_mask=doc_attention_mask,
                return_dict=True,
                use_cache=True,
                # position_ids = torch.arange(text_ids.shape[1]).unsqueeze(0).expand_as(text_ids).to(text_ids.device),
                # token_type_ids = torch.zeros_like(text_ids).to(text_ids.device)
            )
        else:
            query_output = self.model(
                query_embeds=query_tokens,
                encoder_hidden_states=doc_embeds,
                encoder_attention_mask=doc_attention_mask,
                return_dict=True,
                use_cache=True,
            )
        query_embeds = query_output.last_hidden_state[:, : self.num_query_tokens, :]
        # query_embeds = self.projector(query_tokens)

        return query_embeds

    def construct_inputs_embeds(
        self,
        input_ids,
        attention_mask,
        retrieval_embeds,
        inputs_embeds,
        doc_attention_mask,
    ):
        query_embeds = self.gen_query_embeds(
            input_ids, attention_mask, retrieval_embeds, doc_attention_mask
        )

        if type(query_embeds) is list:
            # llaga or g-retrieve setting
            if self.num_query_tokens == 1:
                query_embeds = [
                    query_embed.mean(dim=0, keepdim=True)
                    for query_embed in query_embeds
                ]
            res_embeds = [
                self.ln_norm(self.projector(query_embed.to(inputs_embeds.dtype)))
                for query_embed in query_embeds
            ]
        else:
            query_embeds = query_embeds.to(inputs_embeds.dtype)
            if query_embeds.shape[-1] != inputs_embeds.shape[-1]:
                res_embeds = self.projector(query_embeds)
                res_embeds = self.ln_norm(res_embeds)
            else:
                # prompt tunning
                res_embeds = query_embeds

        return res_embeds

    def forward(self, qformer_inputs, **kwargs):
        return {}


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        projector_type = config.projector_type
        mlp_gelu_match = re.search(r"mlp(\d+)x_gelu", projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            print(f"******* MLP {mlp_depth}x_gelu  *******")
            modules = [
                nn.Linear(
                    (
                        config.retriever_hidden_size
                        if "prompt" not in config.projector_type
                        else config.hidden_size
                    ),
                    config.hidden_size,
                )
            ]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.projector = nn.Sequential(*modules)

    def forward(self, context_embedding):
        return self.projector(context_embedding)


## compatible with normal Mistral model
class XMistralForCausalLM(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        if "gated" in config.projector_type:
            # self.moe = Moe(config)
            config.retriever_hidden_size = config.hidden_size
            config.learnable_gated = getattr(config, "learnable_gated", -1)
            self.projector = build_vision_projector(config)
        elif "cformer" in config.projector_type:
            # print('******* cformer  *******')
            self.projector = CFormer(config)
        elif "prompt" in config.projector_type:
            self.learnable_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
            self.retriever_hidden_size = config.hidden_size
            self.projector = Projector(config)
        elif (
            hasattr(config, "retriever_hidden_size")
            and config.retriever_hidden_size > 0
        ):
            self.projector = Projector(config)
            self.retriever_hidden_size = config.retriever_hidden_size

        if "selectp" in config.projector_type:
            self.token_weights = nn.Linear(self.config.hidden_size, 1)
            
        if "ptuning" in config.projector_type:
            self.learnable_token = nn.Parameter(torch.randn(1, 4, config.hidden_size))
            self.retriever_hidden_size = config.hidden_size
            self.pproj = Projector(config)

        self.post_init()

    def init_cformer(self, encoder_tokenizer):
        print(
            f"loading from {self.config.cformer_model_name_or_path} for init the projector!"
        )
        self.projector.model.from_pretrained(self.config.cformer_model_name_or_path)
        self.projector.init_tokenizer_and_embeds(encoder_tokenizer)
        self.config.projector_vocab_size = len(encoder_tokenizer)

    def set_xrag_token_id(self, token_id, tokenizer=None, pad_id=32001):
        self.xrag_token_id = token_id
        self.pad_token_id = pad_id
        self.tokenizer = tokenizer

    def prepare_inputs_embeds(
        self,
        input_ids,
        retrieval_embeds,
        cformer_input_ids,
        cformer_attention_mask,
        doc_attention_mask,
    ):
        inputs_embeds = self.model.embed_tokens(input_ids)
        if 'onlyp' in self.config.projector_type:
            return inputs_embeds
        batch_size = inputs_embeds.shape[0]

        if "gated" in self.config.projector_type:
            resampler_embeds = self.model.embed_tokens(cformer_input_ids)
            # breakpoint()
            retrieval_embeds = self.projector(
                retrieval_embeds.to(inputs_embeds.dtype),
                doc_attention_mask.to(inputs_embeds.dtype),
                resampler_embeds.to(inputs_embeds.dtype),
                cformer_attention_mask.to(inputs_embeds.dtype),
            )
            if 'ptuning' in self.config.projector_type:
                retrieval_embeds = retrieval_embeds.view(batch_size, -1, inputs_embeds.shape[-1])
                learnable_tokens = self.learnable_token.expand(batch_size, -1, -1)
                retrieval_embeds = torch.cat([retrieval_embeds, self.pproj(learnable_tokens.to(inputs_embeds.dtype))], dim=1).view(-1, inputs_embeds.shape[-1])
            # breakpoint()
        elif "cformer" in self.config.projector_type:
            # retrieval_embeds = self.model.embed_tokens(retrieval_input_ids)
            # retrieval_embeds = self.projector.construct_inputs_embeds(input_ids, attention_mask, retrieval_embeds, inputs_embeds).to(inputs_embeds.dtype)
            retrieval_embeds = (
                self.projector.construct_inputs_embeds(
                    cformer_input_ids,
                    cformer_attention_mask,
                    retrieval_embeds,
                    inputs_embeds,
                    doc_attention_mask,
                )
                .to(inputs_embeds.dtype)
                .view(-1, inputs_embeds.size(-1))
            )
        elif "prompt" in self.config.projector_type:
            retrieval_embeds = self.learnable_token.expand(batch_size, -1, -1).view(
                -1, self.retriever_hidden_size
            )
            retrieval_embeds = self.projector(retrieval_embeds.to(inputs_embeds.dtype))
            # breakpoint()
        else:
            retrieval_embeds = retrieval_embeds.view(-1, self.retriever_hidden_size)

            ## sanity check
            num_xrag_tokens = torch.sum(input_ids == self.xrag_token_id).item()
            num_retrieval_embeds = retrieval_embeds.shape[0]
            assert num_xrag_tokens == num_retrieval_embeds, (
                num_xrag_tokens,
                num_retrieval_embeds,
            )

            retrieval_embeds = self.projector(retrieval_embeds.to(inputs_embeds.dtype))

        if "adagen" in self.config.projector_type or 'ptuning' in self.config.projector_type:
            retrieval_embeds = retrieval_embeds.view(batch_size, -1, inputs_embeds.shape[-1])
            xrag_position = (
                (input_ids == self.xrag_token_id)
                .nonzero(as_tuple=True)[1]
                .view(batch_size, -1)[:, 0]
            )
            xrag_end_position = (
                (input_ids == self.xrag_token_id)
                .nonzero(as_tuple=True)[1]
                .view(batch_size, -1)[:, -1]
            )
            # updated_input_embeds = []
            # for i in range(batch_size):
            #     inputs_embeds_sample = inputs_embeds[i]
            #     insert_position = xrag_position[i].item()
            inputs_embeds = torch.stack([torch.cat(
                    (
                        inputs_embeds[i][:xrag_position[i].item()],
                        retrieval_embeds[i],
                        inputs_embeds[i][xrag_end_position[i].item() + 1:],
                    )
                ) for i in range(batch_size)], dim=0)

        else:
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
        p = None
        retriever_input_ids = kwargs.pop("retriever_input_ids", None)
        if not at_the_beginning_of_generation:
            doc_attention_mask = kwargs.pop("doc_attention_mask", None)

            if retriever_input_ids is not None:
                retrieval_outputs = super().forward(
                    input_ids=retriever_input_ids,
                    attention_mask=doc_attention_mask,
                    output_hidden_states=True,
                )
                last_hidden = retrieval_outputs["hidden_states"][-1].detach()
                retrieval_embeds = last_hidden
                if "selectp" in self.config.projector_type and "stage1" not in self.config.projector_type:
                    p = torch.sigmoid(
                        self.token_weights(last_hidden).view(last_hidden.shape[:2])
                    )
                    topk = torch.topk(p, int(0.15 * retriever_input_ids.shape[1]))
                    # breakpoint()
                    if "chunk" in self.config.projector_type:
                        batch_size = input_ids.shape[0]
                        retriever_topk_ids = []
                        for i in range(batch_size):
                            topk_indices = topk.indices[i]
                            topk_retriever_ids = retriever_input_ids[i, topk_indices]
                            retriever_topk_ids.append(topk_retriever_ids)
                        retriever_topk_ids = torch.stack(
                            retriever_topk_ids, dim=0
                        )  # (batch_size, topk_count)
                        retrieval_attention_mask = torch.ones(
                            retriever_topk_ids.shape
                        ).to(p.device)

                        xrag_position = (
                            (input_ids == self.xrag_token_id)
                            .nonzero(as_tuple=True)[1]
                            .view(batch_size, -1)[:, 0]
                        )
                        xrag_end_position = (
                            (input_ids == self.xrag_token_id)
                            .nonzero(as_tuple=True)[1]
                            .view(batch_size, -1)[:, -1]
                        )
                        updated_input_ids = []
                        for i in range(batch_size):
                            input_ids_sample = input_ids[i]
                            insert_position = xrag_position[i].item()
                            if 'wop' in self.config.projector_type:
                                topk_ids_sample = retriever_topk_ids[i]
                            else:
                                topk_ids_sample = torch.cat(
                                    [
                                        torch.tensor(
                                            self.tokenizer(
                                                "Local hard compressed document: "
                                            ).input_ids[1:]
                                        ).to(p.device),
                                        retriever_topk_ids[i],
                                        torch.tensor(
                                            self.tokenizer(
                                                "\nGlobal soft compressed document: "
                                            ).input_ids[1:]
                                        ).to(p.device),
                                    ],
                                    dim=0,
                                )

                            if 'onlyp' in self.config.projector_type:
                                new_input_ids_sample = torch.cat(
                                    (
                                        input_ids_sample[:insert_position],
                                        topk_ids_sample,
                                        input_ids_sample[xrag_end_position[i].item() + 1:],
                                    )
                                )
                            else:
                                new_input_ids_sample = torch.cat(
                                    (
                                        input_ids_sample[:insert_position],
                                        topk_ids_sample,
                                        input_ids_sample[insert_position:],
                                    )
                                )
                            updated_input_ids.append(new_input_ids_sample)
                        input_ids = torch.stack(updated_input_ids, dim=0)
                        attention_mask = (
                            (input_ids != self.pad_token_id).long().to(p.device)
                        )
                        # breakpoint()
                    else:
                        retrieval_attention_mask = torch.zeros(p.shape).to(p.device)
                        for pdx, pos in enumerate(topk.indices):
                            retrieval_attention_mask[pdx][pos] = 1
                        input_ids = torch.cat((retriever_input_ids, input_ids), dim=1)
                        attention_mask = torch.cat(
                            (retrieval_attention_mask, attention_mask), dim=1
                        )
            ## a single forward
            # breakpoint()
            if retrieval_embeds is not None:
                cformer_input_ids = kwargs.pop("cformer_input_ids", None)
                cformer_attention_mask = kwargs.pop("cformer_attention_mask", None)
                cformer_labels = kwargs.pop("cformer_labels", None)
                # doc_attention_mask = kwargs.pop("doc_attention_mask",None)
                inputs_embeds = self.prepare_inputs_embeds(
                    input_ids,
                    retrieval_embeds,
                    cformer_input_ids,
                    cformer_attention_mask,
                    doc_attention_mask,
                )
                input_ids = None
                if "adagen" in self.config.projector_type or 'ptuning' in self.config.projector_type:
                    if inputs_embeds.shape[1] <= attention_mask.shape[1]:
                        attention_mask = attention_mask[:, :inputs_embeds.shape[1]]
                    else:
                        padding_length = inputs_embeds.shape[1] - attention_mask.shape[1]
                        attention_mask = torch.nn.functional.pad(
                            attention_mask,
                            (0, padding_length, 0, 0),
                            mode="constant",
                            value=1,
                        )
                if attention_mask is not None:
                    assert inputs_embeds.shape[1] == attention_mask.shape[1], (
                        inputs_embeds.shape,
                        attention_mask.shape,
                    )
            # else:
            # assert self.xrag_token_id not in input_ids, input_ids
        output = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        if "selectp" in self.config.projector_type and retriever_input_ids is not None and "stage1" not in self.config.projector_type:
            output["p"] = p
        # breakpoint()
        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        retrieval_embeds=None,  ## [-1,retrieval_hidden_size]
        **kwargs,
    ):
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported for generate")

        inputs_embeds = None
        retriever_input_ids = kwargs.pop("retriever_input_ids", None)
        doc_attention_mask = kwargs.pop("doc_attention_mask", None)
        if retriever_input_ids is not None:
            retrieval_outputs = super().forward(
                input_ids=retriever_input_ids,
                attention_mask=doc_attention_mask,
                output_hidden_states=True,
            )
            last_hidden = retrieval_outputs["hidden_states"][-1].detach()
            retrieval_embeds = last_hidden
            if "selectp" in self.config.projector_type and "stage1" not in self.config.projector_type:
                p = torch.sigmoid(
                    self.token_weights(last_hidden).view(last_hidden.shape[:2])
                )
                topk = torch.topk(p, int(0.15 * retriever_input_ids.shape[1]))
                # breakpoint()
                if "chunk" in self.config.projector_type:
                    batch_size = input_ids.shape[0]
                    retriever_topk_ids = []
                    for i in range(batch_size):
                        topk_indices = topk.indices[i]
                        topk_retriever_ids = retriever_input_ids[i, topk_indices]
                        retriever_topk_ids.append(topk_retriever_ids)
                    retriever_topk_ids = torch.stack(
                        retriever_topk_ids, dim=0
                    )  # (batch_size, topk_count)
                    retrieval_attention_mask = torch.ones(
                        retriever_topk_ids.shape
                    ).to(p.device)

                    xrag_position = (
                        (input_ids == self.xrag_token_id)
                        .nonzero(as_tuple=True)[1]
                        .view(batch_size, -1)[:, 0]
                    )
                    xrag_end_position = (
                        (input_ids == self.xrag_token_id)
                        .nonzero(as_tuple=True)[1]
                        .view(batch_size, -1)[:, -1]
                    )
                    updated_input_ids = []
                    for i in range(batch_size):
                        input_ids_sample = input_ids[i]
                        insert_position = xrag_position[i].item()
                        if 'wop' in self.config.projector_type:
                            topk_ids_sample = retriever_topk_ids[i]
                        else:
                            topk_ids_sample = torch.cat(
                                [
                                    torch.tensor(
                                        self.tokenizer(
                                            "Local hard compressed document: "
                                        ).input_ids[1:]
                                    ).to(p.device),
                                    retriever_topk_ids[i],
                                    torch.tensor(
                                        self.tokenizer(
                                            "\nGlobal soft compressed document: "
                                        ).input_ids[1:]
                                    ).to(p.device),
                                ],
                                dim=0,
                            )

                        if 'onlyp' in self.config.projector_type:
                            new_input_ids_sample = torch.cat(
                                (
                                    input_ids_sample[:insert_position],
                                    topk_ids_sample,
                                    input_ids_sample[xrag_end_position[i].item() + 1:],
                                )
                            )
                        else:
                            new_input_ids_sample = torch.cat(
                                (
                                    input_ids_sample[:insert_position],
                                    topk_ids_sample,
                                    input_ids_sample[insert_position:],
                                )
                            )
                        updated_input_ids.append(new_input_ids_sample)
                    input_ids = torch.stack(updated_input_ids, dim=0)
                    attention_mask = (
                        (input_ids != self.pad_token_id).long().to(p.device)
                    )
                    # breakpoint()
                else:
                    retrieval_attention_mask = torch.zeros(p.shape).to(p.device)
                    for pdx, pos in enumerate(topk.indices):
                        retrieval_attention_mask[pdx][pos] = 1
                    input_ids = torch.cat((retriever_input_ids, input_ids), dim=1)
                    attention_mask = torch.cat(
                        (retrieval_attention_mask, attention_mask), dim=1)

        if retrieval_embeds is not None:
            cformer_input_ids = kwargs.pop("cformer_input_ids", None)
            cformer_attention_mask = kwargs.pop("cformer_attention_mask", None)
            # doc_attention_mask = kwargs.pop("doc_attention_mask",None)
            # breakpoint()
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids,
                retrieval_embeds,
                cformer_input_ids,
                cformer_attention_mask,
                doc_attention_mask,
            )
            input_ids = None
            if "adagen" in self.config.projector_type or 'ptuning' in self.config.projector_type:
                if inputs_embeds.shape[1] <= attention_mask.shape[1]:
                    attention_mask = attention_mask[:, :inputs_embeds.shape[1]]
                else:
                    padding_length = inputs_embeds.shape[1] - attention_mask.shape[1]
                    attention_mask = torch.nn.functional.pad(
                        attention_mask,
                        (0, padding_length, 0, 0),
                        mode="constant",
                        value=1,
                    )
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
