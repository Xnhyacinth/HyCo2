## built-in
import argparse
import json
import logging
import math
import os
import pickle
import random
import types

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"] = "*.pth"  ## not upload ckpt to wandb cloud

import copy
from datetime import *
from functools import partial

## third-party
import datasets
import deepspeed
import torch
import torch.distributed as dist
import transformers
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from datasets import load_dataset, concatenate_datasets, load_from_disk
from tokenizers import AddedToken
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
    Qwen2TokenizerFast,
    SchedulerType,
    get_scheduler,
)
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from src.language_modeling.preprocessing import (
    encode_with_chat_format_finetune,
    encode_with_chat_format_pretrain,
)
from src.language_modeling.utils import (
    XRAG_TOKEN,
    get_kl_loss,
    get_nll_loss,
    get_retrieval_embeds,
    save_with_accelerate,
)

## own
from src.model import (
    SFR,
    XMistralConfig,
    XMistralForCausalLM,
    XMixtralConfig,
    XMixtralForCausalLM,
    XLlamaConfig, XLlamaForCausalLM,
    XQwen2Config, XQwen2ForCausalLM
)
from src.utils import get_yaml_file

logger = get_logger(__name__)


# torch.autograd.set_detect_anomaly(True)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude_dataset_type",
        help="task type to exclude when doing finetuning",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--distill_topk",
        type=int,
        help="topk token to distill in the self-distillation part",
    )
    parser.add_argument("--base_model", help="base LLM load")
    parser.add_argument(
        "--use_fast_tokenizer",
        type=eval,
    )
    parser.add_argument(
        "--use_rag_tuning",
        type=eval,
        help="whether to use retrieval-augmented instruction tuning",
    )
    parser.add_argument("--with_xrag", type=eval, help="whether to use xrag token")
    parser.add_argument(
        "--chat_format", choices=["mistral", "tulu", "mixtral", "qwen", "yi", "gemma"]
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
    )
    parser.add_argument(
        "--update_projector_only",
        type=eval,
    )
    parser.add_argument(
        "--workdir",
        type=str,
    )
    parser.add_argument(
        "--config", type=str, required=True, help="config file to launch the training"
    )
    parser.add_argument("--task_type", type=str, help="pretrain or finetune")
    parser.add_argument(
        "--retrieval_context_length",
        type=int,
        help="max token number for document encoder in dense retrieval",
    )
    parser.add_argument(
        "--alpha_nll",
        type=float,
        help="coefficient for multi-task learning",
    )
    parser.add_argument(
        "--alpha_kl",
        type=float,
        help="coefficient for multi-task learning",
    )
    parser.add_argument(
        "--alpha_p",
        type=float,
        help="coefficient for multi-task learning",
    )
    parser.add_argument(
        "--kl_temperature",
        type=float,
        help="Temperature coefficient for calculation KL-Divergency loss",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default=None,
        help="A csv or a json file containing the dev data.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--retriever_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--use_flash_attn",
        type=eval,
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        help="Ratio of total training steps used for warmup.",
    )
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--exp_note", type=str, default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=eval,
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=eval,
        help=("Turn on gradient checkpointing. Saves memory but slows training."),
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        help="Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).",
    )
    parser.add_argument(
        "--cformer_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument("--num_query_tokens", type=int, default=0)
    parser.add_argument("--cformer_r", type=int, default=16)
    parser.add_argument(
        "--cformer_target_modules",
        type=str,
    )
    parser.add_argument("--projector_type", type=str, default="mlp2x_gelu")
    parser.add_argument("--cformer_model_finetuning_type", type=str, default="full")
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_target", type=str, default="all")
    parser.add_argument("--lora_dropout", type=float, default=0.0) 
    parser.add_argument("--additional_target", type=str, default=None)

    args = parser.parse_args()
    yaml_config = get_yaml_file(args.config)

    ## priority: CLI > YAML (with all default value set to None in argument parser)
    for k, v in yaml_config.items():
        assert hasattr(args, k), f"{k} not in parsed arguments"
        # if getattr(args,k) is None:
        setattr(args, k, v)

    args.train_file = os.path.join(args.workdir, args.train_file)
    if args.dev_file is not None:
        args.dev_file = os.path.join(args.workdir, args.dev_file)
    if args.retriever_name_or_path is not None and os.path.isdir(
        args.retriever_name_or_path
    ):
        args.retriever_name_or_path = os.path.join(
            args.workdir, args.retriever_name_or_path
        )
    if os.path.isdir(os.path.join(args.workdir, args.model_name_or_path)):
        args.model_name_or_path = os.path.join(args.workdir, args.model_name_or_path)

    return args


def collator(
    samples,
    llm_tokenizer,
    retriever_tokenizer=None,
    retrieval_context_length=180,
):
    """
    collate tokenized input_ids and labels with left and right side padding supported

    Args:
        samples (dict): a dict contains input_ids, labels and maybe retrieval_text
        llm_tokenizer: tokenizer for llm
        retriever_tokenizer: tokenizer for retriever
        retrieval_context_length: max length for the retrieved passages

    Returns:
        xrag_input_ids: input_ids with xrag_token_id (xrag_labels,xrag_attention_mask)
        input_ids: input_ids for llm without xrag_token_id, vanilla rag (labels,attention_mask)
        retriever_input_ids: input_ids for retriever (retriever_attention_mask)

    """

    def padding(input_ids, labels=None, padding_side="right"):
        """
        batch padding
        """

        def _padding(ids, padding_value, padding_side="right"):
            if padding_side == "right":
                return torch.nn.utils.rnn.pad_sequence(
                    ids, batch_first=True, padding_value=padding_value
                )
            elif padding_side == "left":
                flipped_ids = [torch.flip(x, dims=[0]) for x in ids]
                return torch.flip(
                    torch.nn.utils.rnn.pad_sequence(
                        flipped_ids, batch_first=True, padding_value=padding_value
                    ),
                    dims=[1],
                )

        input_ids = _padding(
            input_ids,
            padding_value=llm_tokenizer.pad_token_id,
            padding_side=padding_side,
        )
        attention_mask = (input_ids != llm_tokenizer.pad_token_id).long()
        if labels is not None:
            labels = _padding(labels, padding_value=-100, padding_side=padding_side)
        return input_ids, attention_mask, labels

    xrag_input_ids, xrag_attention_mask, xrag_labels = padding(
        input_ids=[x["xrag_input_ids"] for x in samples],
        labels=(
            [x["xrag_labels"] for x in samples]
            if "xrag_labels" in samples[0].keys()
            else None
        ),
        padding_side=llm_tokenizer.padding_side,
    )

    ## add some noise to pretraining task TODO

    ret = {
        "xrag_input_ids": xrag_input_ids,
        "xrag_attention_mask": xrag_attention_mask,
        "xrag_labels": xrag_labels,
    }

    if "retriever_input_text" in samples[0].keys():
        retriever_input_text = [x["retriever_input_text"] for x in samples]
        # assert
        if isinstance(retriever_input_text[0], list):
            retriever_input_text = [x for y in retriever_input_text for x in y]
            tokenized_retrieval_text = retriever_tokenizer(
                retriever_input_text,
                max_length=retrieval_context_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        else:
            tokenized_retrieval_text = llm_tokenizer(
                retriever_input_text,
                max_length=retrieval_context_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        ## handling different retriever tokenization problem
        # if retriever_tokenizer.name_or_path == "intfloat/e5-large-v2":
        #     retriever_input_text = ["passage: "+x for x in retriever_input_text]
        # elif retriever_tokenizer.name_or_path == 'intfloat/e5-mistral-7b-instruct':
        #     retriever_input_text = [x + retriever_tokenizer.eos_token for x in retriever_input_text]
        # if tokenized_retrieval_text["input_ids"].shape[1] == 0:
        #     breakpoint()
        ret["retriever_input_ids"] = tokenized_retrieval_text["input_ids"]
        ret["retriever_attention_mask"] = tokenized_retrieval_text["attention_mask"]

    if "cformer_input_ids" in samples[0].keys():
        cformer_input_ids, cformer_attention_mask, cformer_labels = padding(
            input_ids=[x["cformer_input_ids"] for x in samples],
            labels=(
                [x["cformer_labels"] for x in samples]
                if "cformer_labels" in samples[0].keys()
                else None
            ),
            padding_side=retriever_tokenizer.padding_side,
        )
        ret.update(
            {
                "cformer_input_ids": cformer_input_ids,
                "cformer_attention_mask": cformer_attention_mask,
                "cformer_labels": cformer_labels,
            }
        )

    # breakpoint()
    if "input_ids" in samples[0].keys():
        input_ids = [x["input_ids"] for x in samples]
        labels = [x["labels"] for x in samples]

        input_ids, attention_mask, labels = padding(
            input_ids, labels, padding_side=llm_tokenizer.padding_side
        )

        ret["input_ids"] = input_ids
        ret["attention_mask"] = attention_mask
        ret["labels"] = labels

    return ret


@torch.no_grad()
def validate_during_pretrain(model, dataloader, accelerator, vocab_size, retriever):
    model.eval()
    total_loss = []
    for batch in dataloader:
        retrieval_embeds = get_retrieval_embeds(
            model=retriever,
            input_ids=batch["retriever_input_ids"],
            attention_mask=batch["retriever_attention_mask"],
        )
        outputs = model(
            input_ids=batch["xrag_input_ids"],
            attention_mask=batch["xrag_attention_mask"],
            retrieval_embeds=retrieval_embeds,
        )
        nll_loss = get_nll_loss(
            labels=batch["xrag_labels"],
            logits=outputs.logits,
            vocab_size=vocab_size,
        )
        total_loss.append(nll_loss.item())
    model.train()
    if accelerator.use_distributed and accelerator.num_processes > 1:
        all_ranks_objects = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(all_ranks_objects, total_loss)
        total_loss = [x for y in all_ranks_objects for x in y]
    ppl = torch.exp(torch.tensor(sum(total_loss) / len(total_loss)))
    return ppl


def count_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(
                param.quant_storage, "itemsize"
            ):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def main():
    args = parse_args()
    set_seed(args.seed)
    if "calp" in args.projector_type:
        args.alpha_p = 1.0
    ## we need to load retriever before accelerator init
    retriever = None
    retriever_hidden_size = -1
    retrieval_embed_length = 0  ## deprecated since ColBERT is not concluded
    retriever_tokenizer = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
    )

    if args.retriever_name_or_path is not None:
        if "sfr-embedding-mistral" in args.retriever_name_or_path.lower():
            retriever = SFR.from_pretrained(
                args.retriever_name_or_path, torch_dtype=torch.bfloat16
            )
            retriever_tokenizer = AutoTokenizer.from_pretrained(
                args.retriever_name_or_path
            )
        retrieval_embed_length = retriever.get_embed_length()
        retriever_hidden_size = retriever.get_embed_dim()
        retriever.eval()
    elif "cformer" in args.projector_type:
        retriever_tokenizer = AutoTokenizer.from_pretrained(
            args.cformer_model_name_or_path, use_fast=False
        )
        retriever_tokenizer.eos_token = retriever_tokenizer.sep_token
        retrieval_embed_length = args.num_query_tokens
        retriever_tokenizer.add_tokens(["<xRAG>"], special_tokens=True)
    elif "gated" in args.projector_type or "selectp" in args.projector_type:
        retriever_tokenizer = tokenizer
        retrieval_embed_length = args.num_query_tokens
        retriever_tokenizer.add_tokens(["<xRAG>"], special_tokens=True)

    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="wandb")
    # accelerator.init_trackers(
    #     project_name=args.project_name,
    #     config=args,
    #     init_kwargs={
    #         "wandb": {
    #             "dir": args.workdir,
    #             "name": args.exp_name if args.exp_name is not None else None,
    #             "notes": args.exp_note if args.exp_note is not None else None,
    #             "save_code": True,
    #         },
    #     }
    # )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(seconds=10 * 3600)),
            DistributedDataParallelKwargs(find_unused_parameters=True),
        ],
    )
    accelerator.print(json.dumps(vars(args), indent=4))
    checkpoint_dir = [None]
    if accelerator.is_local_main_process:
        # wandb_tracker = accelerator.get_tracker("wandb")
        if args.chat_format == "mistral":
            checkpoint_dir = [
                os.path.join(
                    args.workdir, f"checkpoint/{args.task_type}/{args.projector_type}"
                )
            ]
        else:
            checkpoint_dir = [
                os.path.join(
                    args.workdir, f"checkpoint_{args.chat_format}/{args.task_type}/{args.projector_type}"
                )
            ]
        # if '_pt' in args.projector_type:
        #     projector_type = args.projector_type.replace('_pt','')
        # else:
        #     projector_type = args.projector_type
        # checkpoint_dir = [os.path.join(wandb_tracker.run.dir,'checkpoint')]
    if accelerator.use_distributed:
        dist.broadcast_object_list(checkpoint_dir, src=0)
    args.output_dir = checkpoint_dir[0]

    if retriever is not None:
        retriever = retriever.to(accelerator.device)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    data_files = {}
    dataset_args = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.dev_file is not None:
        data_files["dev"] = args.dev_file
    if 'mix' in args.projector_type and args.task_type == 'pretrain':
        args.dev_file = None
        if 'dev' in data_files:
            del data_files['dev']
        raw_datasets_red = load_from_disk('data/train/redpajama')
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )
        raw_datasets['train'] = concatenate_datasets([
                raw_datasets_red['train'],
                raw_datasets['train']
            ])
    elif 'red' in args.projector_type and args.task_type == 'pretrain':
        args.dev_file = None
        raw_datasets = load_from_disk('data/train/redpajama')
    else:
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    ## select N samples, mainly for debug
    if (
        args.max_train_samples is not None
        and len(raw_datasets["train"]) > args.max_train_samples
    ):
        selected_indices = random.sample(
            range(len(raw_datasets["train"])), args.max_train_samples
        )
        raw_datasets["train"] = raw_datasets["train"].select(selected_indices)
    # breakpoint()
    if args.exclude_dataset_type is not None:
        for d_type in args.exclude_dataset_type:
            raw_datasets["train"] = raw_datasets["train"].filter(
                lambda example: example["task_type"] != d_type
            )

    if args.chat_format == "mixtral":
        MODEL_CLASS, CONFIG_CLASS = XMixtralForCausalLM, XMixtralConfig
        tokenizer.padding_side = "left"
    elif args.chat_format == "mistral":
        MODEL_CLASS, CONFIG_CLASS = XMistralForCausalLM, XMistralConfig
        tokenizer.padding_side = "left"
    elif args.chat_format == "llama":
        MODEL_CLASS, CONFIG_CLASS = XLlamaForCausalLM, XLlamaConfig
        tokenizer.padding_side = "left"
    elif args.chat_format == "qwen":
        MODEL_CLASS, CONFIG_CLASS = XQwen2ForCausalLM, XQwen2Config
        tokenizer.padding_side = "left"
    else:
        raise ValueError("chat_format not supported")
    config = CONFIG_CLASS.from_pretrained(
        args.model_name_or_path,
        retriever_hidden_size=retriever_hidden_size,
        projector_type=args.projector_type,
        num_query_tokens=args.num_query_tokens,
        cformer_model_name_or_path=args.cformer_model_name_or_path,
        cformer_model_finetuning_type=args.cformer_model_finetuning_type,
    )
    model = MODEL_CLASS.from_pretrained(
        args.model_name_or_path,
        config=config,
        use_flash_attention_2=args.use_flash_attn,
        torch_dtype=torch.bfloat16 if accelerator.mixed_precision == "bf16" else "auto",
    )
    if "cformer" in args.projector_type and "_pt" not in args.projector_type:
        model.init_cformer(retriever_tokenizer)

    num_added_tokens = 0
    ## mistral tokenizer is also a LLamaTokenizer
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast) or args.chat_format == "llama":
        #  or isinstance(tokenizer, Qwen2TokenizerFast)
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "pad_token": "<pad>",
            }
        )
        assert num_added_tokens in [
            0,
            1,
        ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."

    ## XRAG_TOKEN simply functions as a placeholder, would not be trained
    num_added_tokens += tokenizer.add_tokens([AddedToken(XRAG_TOKEN, lstrip=False, rstrip=False)])
    xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    model.set_xrag_token_id(xrag_token_id, tokenizer, tokenizer.pad_token_id)
    if num_added_tokens > 0 and args.chat_format != "qwen":
        model.resize_token_embeddings(len(tokenizer))
        vocab_size = len(tokenizer)
    else:
        vocab_size = model.config.vocab_size
        
    if 'lora' in args.projector_type:
        from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
        from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model

        def find_all_linear_modules(model: "PreTrainedModel") -> list[str]:
            r"""Find all available modules to apply LoRA, GaLore or APOLLO."""
            model_type = getattr(model.config, "model_type", None)
            forbidden_modules = {"lm_head"}
            if model_type == "chatglm":
                forbidden_modules.add("output_layer")
            elif model_type == "internlm2":
                forbidden_modules.add("output")

            # if model_type in COMPOSITE_MODELS:
            #     forbidden_modules.add(COMPOSITE_MODELS[model_type].projector_key)

            # if freeze_vision_tower and model_type in COMPOSITE_MODELS:
            #     forbidden_modules.update(COMPOSITE_MODELS[model_type].vision_model_keys)

            module_names = set()
            for name, module in model.named_modules():
                if any(forbidden_module in name for forbidden_module in forbidden_modules):
                    continue

                if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__ and 'proj' in name.split(".")[-1]:
                    module_names.add(name.split(".")[-1])

            logger.info("Found linear modules: {}".format(",".join(module_names)))
            return list(module_names)

        if args.lora_target == "all":
            target_modules = find_all_linear_modules(model)
        else:
            target_modules = args.lora_target
        # target_modules.append("embed_tokens")
        if args.lora_alpha is None:
            args.lora_alpha = args.lora_rank * 2
        peft_kwargs = {
            "r": args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            # "use_rslora": args.use_rslora,
            # "use_dora": args.use_dora,
            # "modules_to_save": args.additional_target.split(','),
        }
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, **peft_kwargs,)
        # breakpoint()
        model = get_peft_model(model, lora_config)
        trainable_params, all_param = count_parameters(model)

        param_stats = (
            "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
                trainable_params, all_param, 100 * trainable_params / all_param
            )
        )
        logger.info(param_stats)
        # breakpoint()
    # print(f'retrieval_embed_length: {retrieval_embed_length}')
    # print(f'projector_type: {args.projector_type}')
    # Preprocessing the datasets.
    if args.task_type == "finetune":
        encode_function = partial(
            encode_with_chat_format_finetune,  # if "messages" in raw_datasets["train"].column_names else encode_with_completion_format_finetune,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            retrieval_embed_length=retrieval_embed_length,
            use_rag_tuning=args.use_rag_tuning,
            use_retriever_embed=not (retriever is None),
            projector_type=args.projector_type,
            retriever_tokenizer=retriever_tokenizer,
            chat_format=args.chat_format,
            with_xrag=args.with_xrag,
            num_query_tokens=args.num_query_tokens,
        )
    elif args.task_type == "pretrain":
        encode_function = partial(
            encode_with_chat_format_pretrain,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            retrieval_embed_length=retrieval_embed_length,
            chat_format=args.chat_format,
            retriever_tokenizer=retriever_tokenizer,
            projector_type=args.projector_type,
            with_xrag=args.with_xrag,
            num_query_tokens=args.num_query_tokens,
        )
    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            # cache_file_name=
            remove_columns=[
                name
                for name in raw_datasets["train"].column_names
                if name not in ["input_ids", "labels", "attention_mask"]
            ],
            desc=f"Tokenizing and reformatting data on rank: {accelerator.local_process_index}",
        )
        lm_datasets.set_format(type="pt")
        if args.task_type == "finetune":
            lm_datasets["train"] = lm_datasets["train"].filter(
                lambda example: (example["labels"] != -100).any()
            )
            if args.alpha_kl is not None and args.alpha_kl > 0.0:
                lm_datasets["train"] = lm_datasets["train"].filter(
                    lambda example: (example["labels"] != -100).sum()
                    == (example["xrag_labels"] != -100).sum()
                )

    train_dataset = lm_datasets["train"]
    dev_dataset = lm_datasets["dev"] if args.dev_file is not None else None

    collate_fn = partial(
        collator,
        llm_tokenizer=tokenizer,
        retriever_tokenizer=retriever_tokenizer,
        retrieval_context_length=args.retrieval_context_length,
    )

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
    )

    dev_dataloader = None
    if dev_dataset is not None:
        dev_dataloader = DataLoader(
            dev_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=args.per_device_train_batch_size,
        )

    if args.update_projector_only:
        for n, p in model.named_parameters():
            if 'lora' in args.projector_type:
                if p.requires_grad == True:
                    continue
            if "stage1" in args.projector_type:
                if "projector" not in n and "learnable_token" not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            elif "stage2" in args.projector_type:
                if "token_weights" not in n and "pproj" not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            else:
                if (
                    "projector" not in n
                    and "learnable_token" not in n
                    and "token_weights" not in n
                    and "pproj" not in n
                ):
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
        )
    else:
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate
        )

    trainable_params, all_param = count_parameters(model)

    param_stats = (
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )
    logger.info(param_stats)
    # breakpoint()
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # # https://github.com/microsoft/DeepSpeed/pull/4966
    # if args.chat_format == 'mixtral':
    #     deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    # Prepare everything with `accelerator`.
    if dev_dataset is not None:
        model, optimizer, train_dataloader, lr_scheduler, dev_dataloader = (
            accelerator.prepare(
                model, optimizer, train_dataloader, lr_scheduler, dev_dataloader
            )
        )

    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Max Sequence Length = {args.max_seq_length}")
    logger.info(
        f"  Trainable Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)/(10**6):.2f} M"
    )  ## not applicable for deepspeed

    completed_steps = 0
    starting_epoch = 0

    # logging_interval_grad_norm = 0
    logging_interval_loss = 0
    logging_interval_kl_loss = 0
    logging_interval_nll_loss = 0

    total_loss = 0
    total_kl_loss = 0
    total_nll_loss = 0

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    # progress_bar = tqdm(range(args.max_train_steps), disable=True)

    # update the progress_bar if load from checkpoint
    save_one_sample = True

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        active_dataloader = train_dataloader

        for batch in active_dataloader:
            if save_one_sample:
                if accelerator.is_local_main_process:
                    pickle.dump(
                        batch,
                        open(
                            os.path.join(
                                os.path.dirname(args.output_dir), "sample_data.pkl"
                            ),
                            "wb",
                        ),
                    )
                accelerator.print("**" * 20, "show one example", "**" * 20)
                accelerator.print(batch.keys())
                accelerator.print(tokenizer.decode(batch["xrag_input_ids"][0]))
                accelerator.print(batch["xrag_input_ids"][0])
                if "retriever_input_text" in batch:
                    accelerator.print(batch["retriever_input_text"][0])
                if "input_ids" in batch:
                    for input_id, label_id, attention_mask in zip(
                        batch["input_ids"][0],
                        batch["labels"][0],
                        batch["attention_mask"][0],
                    ):
                        accelerator.print(
                            f"{tokenizer.convert_ids_to_tokens([input_id])[0]}({label_id.item()})({attention_mask})",
                            end=" ",
                        )
                accelerator.print()
                for input_id, label_id, attention_mask in zip(
                    batch["xrag_input_ids"][0],
                    batch["xrag_labels"][0],
                    batch["xrag_attention_mask"][0],
                ):
                    accelerator.print(
                        f"{tokenizer.convert_ids_to_tokens([input_id])[0]}({label_id.item()})({attention_mask})",
                        end=" ",
                    )
                accelerator.print("\n" + "**" * 20, "show one example", "**" * 20)
                save_one_sample = False

            with accelerator.accumulate(model):
                ## forward with retrieval embeds
                retrieval_kwargs = {}
                if retriever is not None:
                    retrieval_kwargs["retrieval_embeds"] = get_retrieval_embeds(
                        model=retriever,
                        input_ids=batch["retriever_input_ids"],
                        attention_mask=batch["retriever_attention_mask"],
                    )
                elif "cformer" in args.projector_type or "gated" in args.projector_type or "selectp" in args.projector_type:
                    # with torch.no_grad():
                    #     model.eval()
                    #     retrieval_outputs = model(
                    #         input_ids = batch['retriever_input_ids'],
                    #         attention_mask = batch['retriever_attention_mask'],
                    #         output_hidden_states = True
                    #     )
                    #     last_hidden = retrieval_outputs.hidden_states[-1]
                    #     retrieval_kwargs['retrieval_embeds'] = last_hidden
                    #     retrieval_kwargs['doc_attention_mask'] = batch['retriever_attention_mask']

                    #     p = torch.sigmoid(model.token_weights(last_hidden).view(last_hidden.shape[:2]))
                    #     topk = torch.topk(p,int(0.1*batch['retriever_input_ids'].shape[1]))
                    #     attention_mask = torch.zeros(p.shape).to(p.device)
                    #     for pdx, pos in enumerate(topk.indices):
                    #         attention_mask[pdx][pos]=1
                    #     model.train()

                    # retrieval_kwargs['retrieval_input_ids'] = batch['retriever_input_ids']
                    retrieval_kwargs["retriever_input_ids"] = batch[
                        "retriever_input_ids"
                    ]
                    retrieval_kwargs["doc_attention_mask"] = batch[
                        "retriever_attention_mask"
                    ]
                    retrieval_kwargs.update(
                        {
                            "cformer_input_ids": batch["cformer_input_ids"],
                            "cformer_attention_mask": batch["cformer_attention_mask"],
                            "cformer_labels": batch["cformer_labels"],
                        }
                    )
                    # breakpoint()
                else:
                    retrieval_kwargs["retrieval_embeds"] = []

                outputs = model(
                    input_ids=batch["xrag_input_ids"],
                    attention_mask=batch["xrag_attention_mask"],
                    **retrieval_kwargs,
                )
                loss = None
                # labels = batch['labels']
                xrag_labels = batch["xrag_labels"]
                if xrag_labels.shape[1] != outputs.logits.shape[1]:
                    target_length = outputs.logits.size(1)
                    padding_length = target_length - xrag_labels.size(1)
                    if padding_length > 0:
                        xrag_labels = torch.nn.functional.pad(
                            xrag_labels,
                            (padding_length, 0, 0, 0),
                            mode="constant",
                            value=-100,
                        )
                    else:
                        xrag_labels = xrag_labels[:, -padding_length:]
                        print(-padding_length)
                # breakpoint()
                if args.alpha_nll is not None and args.alpha_nll > 0.0:

                    nll_loss = get_nll_loss(
                        labels=xrag_labels,
                        logits=outputs.logits,
                        vocab_size=vocab_size,
                    )

                    logging_interval_nll_loss += nll_loss.detach().float()

                    loss = args.alpha_nll * nll_loss

                if args.alpha_kl is not None and args.alpha_kl > 0.0:

                    ## forward with retrieval tokens
                    with torch.no_grad():
                        model.eval()
                        teacher_outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                        model.train()

                    kl_loss = get_kl_loss(
                        teacher_logits=teacher_outputs.logits,
                        teacher_labels=batch["labels"],
                        student_logits=outputs.logits,
                        student_labels=xrag_labels,
                        temperature=args.kl_temperature,
                        distill_topk=args.distill_topk,
                    )
                    logging_interval_kl_loss += kl_loss.detach().float()
                    if loss is not None:
                        loss += args.alpha_kl * kl_loss
                    else:
                        loss = args.alpha_kl * kl_loss

                def compute_Lloss(output, p=1):
                    loss = torch.norm(output.p, p=p).to(torch.float32)
                    return 100 * loss / output.p.shape[1]

                if args.alpha_p is not None and args.alpha_p > 0.0 and 'stage1' not in args.projector_type:
                    l1_loss = compute_Lloss(outputs)
                    alpha = 1.0 / (l1_loss + 1e-8)
                    loss = (alpha * loss) + l1_loss

                logging_interval_loss += loss.detach().float()
                # with torch.autograd.detect_anomaly():
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = (
                        accelerator.gather(logging_interval_loss).mean().item()
                        / args.gradient_accumulation_steps
                        / args.logging_steps
                    )
                    total_loss += (
                        accelerator.gather(logging_interval_loss).mean().item()
                        / args.gradient_accumulation_steps
                    )

                    to_be_logged = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "train_loss": avg_loss,
                        "rolling_loss": total_loss / completed_steps,
                    }
                    if args.alpha_nll is not None and args.alpha_nll > 0.0:
                        total_nll_loss += (
                            accelerator.gather(logging_interval_nll_loss).mean().item()
                            / args.gradient_accumulation_steps
                        )
                        to_be_logged["rolling_nll_loss"] = (
                            total_nll_loss / completed_steps
                        )

                    if args.alpha_kl is not None and args.alpha_kl > 0.0:
                        total_kl_loss += (
                            accelerator.gather(logging_interval_kl_loss).mean().item()
                            / args.gradient_accumulation_steps
                        )
                        to_be_logged["rolling_kl_loss"] = (
                            total_kl_loss / completed_steps
                        )

                    accelerator.log(to_be_logged, step=completed_steps)

                    # logging_interval_grad_norm = 0
                    logging_interval_loss = 0
                    logging_interval_kl_loss = 0
                    logging_interval_nll_loss = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = os.path.join(
                            args.output_dir, f"step_{completed_steps}"
                        )
                        save_with_accelerate(
                            accelerator,
                            model,
                            tokenizer,
                            output_dir,
                            save_projector_only=args.update_projector_only,
                        )

                        if dev_dataloader is not None:
                            if args.task_type == "pretrain":
                                ppl = validate_during_pretrain(
                                    model,
                                    dev_dataloader,
                                    accelerator,
                                    vocab_size,
                                    retriever,
                                )
                                accelerator.log({"dev_ppl": ppl}, step=completed_steps)

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            save_with_accelerate(
                accelerator,
                model,
                tokenizer,
                output_dir,
                save_projector_only=args.update_projector_only,
            )

    accelerator.end_training()

    ## save the last one
    output_dir = os.path.join(args.output_dir, "last")
    save_with_accelerate(
        accelerator, model, tokenizer, output_dir, save_projector_only=False
    )


if __name__ == "__main__":
    main()
