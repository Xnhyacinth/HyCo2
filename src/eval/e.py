## built-in
import argparse
import json
import os
import time

import datasets
import pandas as pd
import torch
from tqdm import tqdm
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile
from torch.profiler import record_function
from transformers import AutoTokenizer
## third party
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    MistralForCausalLM,
    MixtralForCausalLM,
    Qwen2ForCausalLM,
    LlamaForCausalLM
)

from src.eval.utils import (
    eval_fact_checking,
    eval_truthfulqa,
    get_substring_match_score,
    keyword_extraction_with_tfidf,
    stop_sequences_criteria,
    calculate_information_loss,
    calculate_rouge_scores,
    evaluate_bertscore
)
from src.language_modeling.utils import XRAG_TOKEN, get_retrieval_embeds
import random
## own
from src.model import SFR, XMistralForCausalLM, XMixtralForCausalLM, XLlamaForCausalLM, XQwen2ForCausalLM
from src.utils import get_jsonl

re_prompts = [
    "These two expressions are equivalent in essence:(1) {token} (2)",
    "In other words, background: {token} is just another way of saying:",
    "Background: {token} means the same as",
    "{token} After unpacking the ideas in the background information above, we got:",
    "{token} Please offer a restatement of the background sentences I’ve just read."
]

def create_prompt_with_mistral_chat_format(messages, tokenizer, *args, **kwargs):
    # return tokenizer.apply_chat_template(messages,tokenize=False,add_special_tokens=False)
    formatted_text = ""
    for message in messages:
        if message["role"] == "user":
            formatted_text += "[INST] " + message["content"] + " [/INST]"
        elif message["role"] == "assistant":
            formatted_text += message["content"] + tokenizer.eos_token
        else:
            raise ValueError(
                "Mistral chat template only supports 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    # formatted_text += " The answer is:"
    return formatted_text

def create_prompt_with_llama_chat_format(messages, tokenizer, *args, **kwargs):
    # return tokenizer.apply_chat_template(messages,tokenize=False,add_special_tokens=False)
    return tokenizer.apply_chat_template(
                messages, tokenize=False
            )

def create_prompt_with_qwen_chat_format(messages, tokenizer, *args, **kwargs):
    # return tokenizer.apply_chat_template(messages,tokenize=False,add_special_tokens=False)
    return tokenizer.apply_chat_template(
                messages, tokenize=False
            )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_prefix", default="colbertv2")
    parser.add_argument("--test", default="rag")
    parser.add_argument(
        "--tf_idf_topk",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--base_model",
    )
    parser.add_argument(
        "--use_rag",
        action="store_true",
    )
    parser.add_argument(
        "--cformer",
        action="store_true",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
    )
    parser.add_argument("--projector_type", type=str, default="mlp2x_gelu")
    parser.add_argument(
        "--cformer_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="models/FacebookAI__roberta-base",
    )
    parser.add_argument(
        "--enable_progress_bar",
        type=eval,
        default=True,
    )
    parser.add_argument(
        "--data",
    )
    parser.add_argument(
        "--model_name_or_path",
    )
    parser.add_argument(
        "--eval_metrics",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--retriever_name_or_path",
    )
    parser.add_argument("--num_query_tokens", type=int)
    parser.add_argument(
        "--retrieval_topk",
        type=int,
        default=1,
        # nargs='+',
    )
    parser.add_argument(
        "--retrieval_embed_length",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        help="for debug",
    )
    parser.add_argument(
        "--save_dir",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--chat_format",
        default="mistral",
    )
    args = parser.parse_args()

    ## post-process
    if args.data in [
        "nq_open",
        "hotpotqa",
        "triviaqa",
        "webqa",
        "tqa",
        "nq",
        "wq",
        "2wikimqa",
        "popqa",
        "cwq"
    ]:
        args.task_type = "open_qa"
        args.eval_metrics = "substring_match"
    elif args.data in ["truthfulqa"]:
        args.task_type = "open_qa"
        args.eval_metrics = "truthfulqa_f1_rl"
    elif args.data in ["factkg"]:
        args.task_type = "fact_checking"
        args.eval_metrics = "fact_checking_acc"

    # args.retrieval_topk = [x-1 for x in args.retrieval_topk] ## rank starts from 1

    if args.chat_format is not None:
        args.chat_format = eval(f"create_prompt_with_{args.chat_format}_chat_format")

    if args.retriever_name_or_path is not None:
        args.use_rag = True

    return args


QA_PROMPT = "Question: {question}?\n"
FECT_CHECKING_PROPMT = "Claim: {question}\n"
BACKGROUND_PROMPT_TEMPLATE = "Background: {background}\n\n"

PROMPT_TEMPLATES = {
    "open_qa": QA_PROMPT,
    "fact_checking": FECT_CHECKING_PROPMT,
}


def get_start_prompt(task_type, use_rag, sample=None):
    if task_type == "open_qa":
        return {
            True: "Refer to the background document and answer the questions:",
            False: "Answer the questions:",
        }[use_rag]
    elif task_type == "fact_checking":
        return {
            True: 'Refer to the background document and verify the following claims with "True" or "False":',
            False: 'Verify the following claims with "True" or "False":',
        }[use_rag]


@torch.no_grad()
def prepare_retrieval_embeds(
    backgrounds, retriever, tokenizer, model=None, batch_size=16
):
    backgrounds = [
        backgrounds[idx : idx + batch_size]
        for idx in range(0, len(backgrounds), batch_size)
    ]
    device = model.device
    ret = []
    attn_masks = []

    for background in backgrounds:
        tokenized_retrieval_text = tokenizer(
            background,
            max_length=4096,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        ## return a torch tensor of shape [batch_size,d_model]
        if retriever is not None:
            embeds = get_retrieval_embeds(
                model=retriever,
                input_ids=tokenized_retrieval_text["input_ids"].to(device),
                attention_mask=tokenized_retrieval_text["attention_mask"].to(device),
            ).cpu()
            embeds = [embeds[idx] for idx in range(embeds.shape[0])]
        else:
            # embeds = model(input_ids=tokenized_retrieval_text['input_ids'].to(device), attention_mask = tokenized_retrieval_text['attention_mask'].to(device), output_hidden_states=True).hidden_states[-1].cpu()

            attn_masks.extend(
                [
                    tokenized_retrieval_text["attention_mask"][idx]
                    for idx in range(
                        tokenized_retrieval_text["attention_mask"].shape[0]
                    )
                ]
            )
            embeds = [
                tokenized_retrieval_text["input_ids"][idx]
                for idx in range(tokenized_retrieval_text["attention_mask"].shape[0])
            ]

        ret.extend(embeds)
    return ret, attn_masks


@torch.no_grad()
def llm_for_open_generation(
    llm,
    llm_tokenizer,
    retriever_tokenizer,
    prompts,
    retrieval_embeds,
    batch_size=4,
    enable_progress_bar=True,
    projector_type=None,
    attn_masks=None,
    args=None
):
    generated_answers = []
    total_test_number = len(prompts)
    device = llm.device
    batched_prompts = [
        prompts[idx : idx + batch_size] for idx in range(0, len(prompts), batch_size)
    ]
    if retrieval_embeds is not None:
        batched_retrieval_embeds = [
            retrieval_embeds[idx : idx + batch_size]
            for idx in range(0, len(retrieval_embeds), batch_size)
        ]
        batched_attn_masks = [
            attn_masks[idx : idx + batch_size]
            for idx in range(0, len(attn_masks), batch_size)
        ]
        assert len(batched_prompts) == len(batched_retrieval_embeds)

    progress_bar = tqdm(
        range(total_test_number), ncols=60, disable=not enable_progress_bar
    )
    for batch_idx in range(len(batched_prompts)):
        prompt = batched_prompts[batch_idx]
        tokenized_propmt = llm_tokenizer(prompt, padding="longest", return_tensors="pt")
        input_ids = tokenized_propmt.input_ids.to(device)
        attention_mask = tokenized_propmt.attention_mask.to(device)
        stopping_criteria = stop_sequences_criteria(
            llm_tokenizer, input_ids.shape[1], input_ids.shape[0]
        )
        retrieval_kwargs = {}
        if retrieval_embeds is not None:
            embeds = batched_retrieval_embeds[batch_idx]

            # masks = [x for y in masks for x in y]
            if "cformer" == projector_type or "gated" in projector_type or "selectp" == projector_type:
                import re

                # pattern = r"Question: (.+?)\?\n"
                text = "\n".join(prompt)
                if args.reconstruct:
                    questions = "Please offer a restatement of the background sentences I’ve just read."
                else:
                    questions = re.findall(r"Question: (.+?)\?", text)
                # breakpoint()
                cformer_inputs = retriever_tokenizer(
                    questions, padding="longest", return_tensors="pt"
                )
                retrieval_kwargs.update(
                    {
                        "cformer_input_ids": cformer_inputs["input_ids"].to(device),
                        "cformer_attention_mask": cformer_inputs["attention_mask"].to(
                            device
                        ),
                    }
                )
                masks = batched_attn_masks[batch_idx]
                masks = torch.stack(masks).to(device)
                retrieval_kwargs["doc_attention_mask"] = masks
                retrieval_kwargs["retriever_input_ids"] = torch.stack(embeds).to(device)
                # breakpoint()
            else:
                embeds = [x for y in embeds for x in y]
                embeds = torch.stack(embeds).to(device)
                retrieval_kwargs["retrieval_embeds"] = embeds

            stopping_criteria = stop_sequences_criteria(
                llm_tokenizer, 0, input_ids.shape[0]
            )
        elif "prompt" == projector_type:
            retrieval_kwargs["retrieval_embeds"] = []
        ## actual computation
        # breakpoint()
        generated_output = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            do_sample=False,
            max_new_tokens=30,
            min_new_tokens=30,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            **retrieval_kwargs,
        )
        ## because HF generate with inputs_embeds would not return prompt
        input_length = 0 if retrieval_kwargs else input_ids.shape[1]
        results = tokenizer.batch_decode(
            generated_output[:, input_length:], skip_special_tokens=False
        )
        generated_answers.extend(results)
        progress_bar.update(batch_size)

    generated_answers = [x.strip() for x in generated_answers]
    return generated_answers


def format_one_example(
    sample,
    include_answer,
    use_rag,
    retrieval_embed_length,
    task_type,
    projector_type=None,
    args=None
):

    question = sample["question"]
    prompt_dict = dict(question=question)
    prompt = PROMPT_TEMPLATES[task_type].format_map(prompt_dict).strip()
    backgrounds = []

    if use_rag:
        backgrounds = sample["background"]  ## a list
        background_prompts = ""

        if "adagen" in projector_type:
            background_prompts += XRAG_TOKEN
        elif "cformer" == projector_type or "gated" in projector_type:
            if "cat" in projector_type:
                background_prompts += "".join([XRAG_TOKEN] * retrieval_embed_length * 2)
            else:
                background_prompts += "".join([XRAG_TOKEN] * retrieval_embed_length)
        elif "prompt" == projector_type or "selectp" == projector_type:
            background_prompts += XRAG_TOKEN
        else:
            for background in backgrounds:
                if retrieval_embed_length > 0:
                    background_prompts += (
                        " ".join([XRAG_TOKEN] * retrieval_embed_length) + " "
                    )
                else:
                    background_prompts += background + " "
        # background_prompts = background_prompts.strip()
        prompt = (
            BACKGROUND_PROMPT_TEMPLATE.format_map(dict(background=background_prompts))
            + prompt
        )
    if args.reconstruct:
        return random.choice(re_prompts).format(token=background_prompts), backgrounds
    return prompt, backgrounds


def get_n_shot_prompt(
    dev_data, n_shot, task_type, use_rag=False, retrieval_embed_length=0
):
    assert n_shot >= 0, n_shot
    n_shot_prompt = []
    n_shot_background = []
    if dev_data is not None:
        n_shot_examples = dev_data[:n_shot]
        for example in n_shot_examples:
            prompt, background = format_one_example(
                example,
                include_answer=True,
                use_rag=use_rag,
                retrieval_embed_length=retrieval_embed_length,
                task_type=task_type,
            )
            n_shot_prompt.append(prompt)
            n_shot_background.append(background)

    return n_shot_prompt, n_shot_background


def prepare_prompts(
    dev_data,
    test_data,
    task_type,
    tokenizer,
    n_shot=0,
    use_rag=False,
    retrieval_embed_length=0,
    chat_format=None,
    projector_type=None,
    args=None
):
    splitter = "\n\n"
    prompts = []
    backgrounds = []
    original_n_shot = n_shot
    for idx, sample in enumerate(test_data):
        n_shot = original_n_shot
        while True:
            prompt_start = get_start_prompt(task_type, use_rag=use_rag, sample=sample)
            prompt_end, background = format_one_example(
                sample,
                include_answer=False,
                use_rag=use_rag,
                retrieval_embed_length=retrieval_embed_length,
                task_type=task_type,
                projector_type=projector_type,
                args=args
            )
            if "subject" not in sample.keys():
                n_shot_prompt, n_shot_background = get_n_shot_prompt(
                    dev_data,
                    n_shot=n_shot,
                    use_rag=use_rag,
                    retrieval_embed_length=retrieval_embed_length,
                    task_type=task_type,
                )
            else:
                ## select n-shot within the same subjects for MMLU
                dev_data_with_same_subjects = []
                for d in dev_data:
                    if d["subject"] == sample["subject"]:
                        dev_data_with_same_subjects.append(d)
                assert len(dev_data_with_same_subjects) == 5, sample["subject"]
                n_shot_prompt, n_shot_background = get_n_shot_prompt(
                    dev_data_with_same_subjects,
                    n_shot=n_shot,
                    use_rag=use_rag,
                    retrieval_embed_length=retrieval_embed_length,
                    task_type=task_type,
                )

            if n_shot_prompt:
                prompt = (
                    prompt_start
                    + splitter
                    + splitter.join(n_shot_prompt)
                    + splitter
                    + prompt_end
                )
            elif args.reconstruct:
                prompt = prompt_end
            else:
                prompt = prompt_start + splitter + prompt_end

            if chat_format is not None:
                messages = [{"role": "user", "content": prompt}]
                if args.reconstruct:
                    prompt = chat_format(messages, tokenizer)
                else:
                    prompt = chat_format(messages, tokenizer) + " The answer is:"

            tokenized_prompt = tokenizer(
                prompt, truncation=False, add_special_tokens=False
            ).input_ids

            if len(tokenized_prompt) > 2048 and n_shot >= 1:
                n_shot -= 1
            else:
                break

        prompts.append(prompt)
        backgrounds.append(background + n_shot_background)

    print("**" * 20, "show one example", "**" * 20)
    print(prompts[0])
    print("**" * 20, "show one example", "**" * 20)

    return prompts, backgrounds


def load_dataset(data, use_rag, args):

    dev_data = None
    test_path = f"program/mydata/data30/{data}/test.jsonl"
    test_data = None
    if os.path.isfile(test_path):
        test_data = get_jsonl(test_path)

    if use_rag:
        
        # if args.longbench:
        # if args.baseline:
        #     for idx in range(len(test_data)):
        #         test_data[idx]["background"] = [test_data[idx][f"{args.test}_r{args.retrieval_topk}_t32"]]
        # else:
        if data == "triviaqa":
            test_retrieval_path = os.path.join(
                f"compress/data/eval/{data}/retrieval/{args.retrieval_prefix}",
                "test.jsonl",
            )
            test_retrieval = get_jsonl(test_retrieval_path)
            assert len(test_retrieval) == len(test_data)
            if args.retrieval_prefix == "contriever":
                for idx in range(len(test_data)):
                    test_data[idx]["background"] = [
                        test_retrieval[idx]["background"][rank]
                        for rank in range(args.retrieval_topk)
                    ]
            else:
                for idx in range(len(test_data)):
                    test_data[idx]["background"] = [
                        test_retrieval[idx]["topk"][rank]["text"]
                        for rank in range(args.retrieval_topk)
                    ]
        elif data == "hotpotqa" or data == "2wikimqa":
            for idx in range(len(test_data)):
                try:
                    # test_data[idx]['background'] = [' '.join(test_data[idx]['context'][rank][1]) for rank in range(args.retrieval_topk)]
                    test_data[idx]["background"] = [
                        "Title: "
                        + test_data[idx]["context"][rank][0]
                        + ". Context: "
                        + " ".join(test_data[idx]["context"][rank][1])
                        for rank in range(args.retrieval_topk)
                    ]
                except:
                    test_data[idx]["background"] = [" "]
        elif data == "popqa" or data == "cwq":
            for idx in range(len(test_data)):
                try:
                    # test_data[idx]['background'] = [' '.join(test_data[idx]['context'][rank][1]) for rank in range(args.retrieval_topk)]
                    test_data[idx]["background"] = [
                        test_data[idx]["passages"][rank]
                        for rank in range(args.retrieval_topk)
                    ]
                except:
                    test_data[idx]["background"] = [" "]
        else:
            for idx in range(len(test_data)):
                try:
                    test_data[idx]["background"] = [
                        test_data[idx]["background"][rank]
                        for rank in range(args.retrieval_topk)
                    ]
                except:
                    test_data[idx]["background"] = [" "]

        if args.tf_idf_topk > 0:
            assert args.use_rag
            if os.path.exists(f"compress/data/eval/{data}/keywords.json"):
                keywords = []
                with open(f"compress/data/eval/{data}/keywords.json", 'r') as f:
                    cache_keywords = json.load(f)
                    for keyword in cache_keywords:
                        keywords.append(' '.join(keyword[:args.tf_idf_topk]))
            else:
                documents = [x["background"][0] for x in test_data]
                keywords = keyword_extraction_with_tfidf(documents, topk=args.tf_idf_topk)
                
            for idx in range(len(test_data)):
                test_data[idx]["background"] = [keywords[idx]]

        if (
            args.retriever_name_or_path is not None
            and args.retriever_name_or_path.lower() == "intfloat/e5-large-v2"
        ):
            for idx in range(len(test_data)):
                test_data[idx]["background"] = [
                    "passage: " + x for x in test_data[idx]["background"]
                ]
    # if args.reconstruct:
    #     test_data = random.sample(test_data, 100)
    return dev_data, test_data


if __name__ == "__main__":

    args = parse_args()

    ## load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
        add_eos_token=False,  ## import to include this!
        use_fast=False,
    )
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ## load retriever and retriever_tokenizer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    retrieval_embed_length = 0
    retriever, retriever_tokenizer = None, None
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
        retriever = retriever.to(device)
    elif "cformer" == args.test:
        retriever_tokenizer = AutoTokenizer.from_pretrained(
            args.cformer_name_or_path, use_fast=False
        )
        retriever_tokenizer.eos_token = retriever_tokenizer.sep_token
        retrieval_embed_length = args.num_query_tokens
        retriever_tokenizer.add_tokens(["<xRAG>"], special_tokens=True)
    elif "gated" in args.test or "selectp" == args.test:
        retriever_tokenizer = tokenizer
        retrieval_embed_length = args.num_query_tokens
        retriever_tokenizer.add_tokens(["<xRAG>"], special_tokens=True)

    ## prepare prompt
    dev_data, test_data = load_dataset(
        args.data,
        args.use_rag,
        args,
    )

    if args.max_test_samples is not None:
        test_data = test_data[: args.max_test_samples]
    # test_data = test_data[:30]
    prompts, backgrounds = prepare_prompts(
        dev_data=dev_data,
        test_data=test_data,
        task_type=args.task_type,
        tokenizer=tokenizer,
        n_shot=args.n_shot,
        use_rag=args.use_rag,
        retrieval_embed_length=retrieval_embed_length,
        chat_format=args.chat_format,
        projector_type=args.test,
        args=args,
    )

    ## load llm
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    MODEL_CLASS = eval(config.architectures[0])
    # if args.cformer:
    #     model = MODEL_CLASS(config)
    #     model.projector.init_tokenizer_and_embeds(retriever_tokenizer)
    #     model = model.from_pretrained(
    #         args.model_name_or_path,
    #         torch_dtype = torch.bfloat16,
    #         # low_cpu_mem_usage = True,
    #         device_map='auto',
    #         config=config,
    #     )
    # else:

    model = MODEL_CLASS.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage = True,
        device_map="auto",
        config=config,
    )

    model.eval()
    torch.cuda.reset_peak_memory_stats(model.device)
    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True,
    ) as prof:
        with record_function("model_inference"):
          if args.test == "llmlingua2":
            from llmlingua import PromptCompressor
            compressor = PromptCompressor(
                  model_name="models/microsoft__llmlingua-2-xlm-roberta-large-meetingbank",
                  use_llmlingua2=True, # Whether to use llmlingua-2
                  device_map="auto",
              )
            print('use_llmlingua2')
            for dd in test_data[:]:
              if dd["background"] != [" "]:
                  dd[f"background"] = compressor.compress_prompt('\n'.join(dd["background"]), target_token=50, force_tokens = ['\n', '?'])['compressed_prompt']
              else:
                  dd[f"background"] = " "
          if args.test == "exit":
            from src.eval.rag import ExitRAG, Document

            # Initialize pipeline
            compressor = ExitRAG(
              retriever_model="models/google__gemma-2b-it",
              compression_model="models/doubleyyh__exit-gemma-2b",
            # reader_model="meta-llama/Llama-3.1-8B-Instruct"
            )
            print('exit')
            for dd in test_data[:]:
              if dd["background"] != [" "]:
                  dd[f"background"] = compressor.compress_documents(dd["question"], [Document(title=None,text=back) for back in dd["background"]], threshold=0.5)[0]
              else:
                  dd[f"background"] = " "
          if args.test == "longllmlingua":
            from llmlingua import PromptCompressor

            llm_lingua = PromptCompressor(model_name="models/NousResearch__Llama-2-7b-hf")
            print('use_longllmlingua')
            for dd in test_data[:]:
              if dd["background"] != [" "]:
                  compressed_prompt = llm_lingua.compress_prompt(
                    dd[f"background"],
                    question=dd['question'],
                    rate=0.55,
                    # Set the special parameter for LongLLMLingua
                    condition_in_question="after_condition",
                    reorder_context="sort",
                    dynamic_context_compression_ratio=0.3, # or 0.4
                    condition_compare=True,
                    context_budget="+100",
                    rank_method="longllmlingua",
                    )['compressed_prompt'].split('\n\n')[:-1]
                  dd[f"background"] = '\n\n'.join([x for x in compressed_prompt])
              else:
                  dd[f"background"] = " "

          retrieval_embeds, attn_masks = None, None
          if retriever_tokenizer is not None:
              # backgrounds List[List[String]]
              num_samples = len(backgrounds)
              original_orders = []
              context = [" ".join(y) for y in backgrounds]
              for idx, background in enumerate(backgrounds):
                  original_orders.extend([idx] * len(background))

              if "cformer" == args.test or "gated" in args.test or "selectp" == args.test:
                  backgrounds = [" ".join(y) for y in backgrounds]
                  # backgrounds = [x for y in backgrounds for x in y]
                  print(backgrounds[0])
                  print(len(backgrounds))

                  retrieval_embeds, attn_masks = prepare_retrieval_embeds(
                      backgrounds, retriever, tokenizer, model, batch_size=args.eval_batch_size
                  )

                  # retrieval_embeds = [[] for _ in range(num_samples)]
                  # assert len(_retrieval_embeds) == len(original_orders)
                  # for id,embeds in zip(original_orders,_retrieval_embeds):
                  #     retrieval_embeds[id].append(embeds)
              else:
                  backgrounds = [x for y in backgrounds for x in y]
                  print(f"Preparing document embedding with {args.retriever_name_or_path}...")
                  _retrieval_embeds, attn_masks = prepare_retrieval_embeds(
                      backgrounds, retriever, retriever_tokenizer, model, batch_size=args.eval_batch_size
                  )
                  retriever = retriever.to("cpu")

                  retrieval_embeds = [[] for _ in range(num_samples)]
                  assert len(_retrieval_embeds) == len(original_orders)
                  for id, embeds in zip(original_orders, _retrieval_embeds):
                      retrieval_embeds[id].append(embeds)
          # breakpoint()
          avg_prompt_length = tokenizer(prompts, return_length=True).length
          avg_prompt_length = sum(avg_prompt_length) / len(avg_prompt_length)

          # model = model.to(device)
        #   if retriever_tokenizer is not None or "prompt" in args.projector_type:
        #       assert XRAG_TOKEN in tokenizer.get_vocab()
        #       model.set_xrag_token_id(tokenizer.convert_tokens_to_ids(XRAG_TOKEN), tokenizer, tokenizer.pad_token_id)

        #   if args.task_type in ["open_qa", "fact_checking"]:
        #       generated_results = llm_for_open_generation(
        #           llm=model,
        #           llm_tokenizer=tokenizer,
        #           retriever_tokenizer=retriever_tokenizer,
        #           prompts=prompts,
        #           retrieval_embeds=retrieval_embeds,
        #           batch_size=args.eval_batch_size,
        #           enable_progress_bar=args.enable_progress_bar,
        #           projector_type=args.test,
        #           attn_masks=attn_masks,
        #           args=args
        #       )
    peak_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 2**30
    events = prof.key_averages()
    for event in events:
        if event.key == "model_inference":
            model_inference_event = event
            break
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    # breakpoint()
    total_cpu_time = model_inference_event.cpu_time_total / 1000**2 / len(test_data)
    total_cuda_time = model_inference_event.cuda_time / 1000**2 / len(test_data)
    total_gflops = sum([event.flops for event in events]) / 1e9 / len(test_data)

    result_dict = {
        # "instruction_length": instruction_length,
        # "document_length": document_length,
        # "prompt_length": input_ids.shape[1],
        "generation_length": 30,
        # "use_xrag": use_xrag,
        "cpu_time": total_cpu_time,
        "cuda_time": total_cuda_time,
        "gflops": total_gflops / 30,
        "peak_mem": peak_mem_usage,
    }
    print(json.dumps(result_dict, indent=4))
# bash src/eval/eval.sh 0 mlp2x_gelu 32 contriever 3 xrag 2wikimqa mistral 0 3
    if args.retriever_name_or_path is not None:
        result_dict["retriever"] = args.retriever_name_or_path
        prefix = f"{args.save_dir}/xrag"
    elif args.test == "cformer":
        result_dict["num_query_tokens"] = args.num_query_tokens
        prefix = f"{args.save_dir}/cformer"
    elif args.test == "prompt":
        prefix = f"{args.save_dir}/prompt"
    elif "gated" in args.test:
        prefix = f"{args.save_dir}/gated"
    elif args.test == "selectp":
        prefix = f"{args.save_dir}/selectp"
    # elif args.use_rag:
    #     prefix = f"{args.save_dir}/rag"
    elif args.baseline:
        prefix = f"{args.save_dir}/{args.test}_t32"
    else:
        prefix = f"{args.save_dir}/{args.test}"

    if args.reconstruct:
        prefix += "_reconstruct"

    # print(json.dumps(result_dict, indent=4))
    prefix += "_pro"
    os.makedirs(prefix, exist_ok=True)
    with open(f"{prefix}/results.json", "w") as f:
        json.dump(result_dict, f, indent=4)
    # with open(f"{prefix}/gen_answer.json", "w") as f:
    #     json.dump({"gen": generated_results}, f, indent=4)


