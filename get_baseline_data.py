import json
from llmlingua import PromptCompressor
import json
from compressors import LongLLMLinguaCompressor, EXITCompressor
from dataclasses import dataclass
import os
import copy

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True, # Whether to use llmlingua-2
    device_map="auto",
)
for data in ['2wikimqa', 'nq', 'tqa', 'hotpotqa', 'wq', 'cwq', 'popqa']:
    with open(f'data/{data}/test.jsonl', 'r') as f:
        test_data = [json.loads(x) for x in f.readlines()]
    if data == "cwq" or data == "popqa":
        xxx = [1,2,3]
    else:
        xxx = [1, 3, 5]
    for retrieval_topk in xxx:
        if data == "hotpotqa" or data == "2wikimqa":
            for idx in range(len(test_data)):
                try:
                    # test_data[idx]['background'] = [' '.join(test_data[idx]['context'][rank][1]) for rank in range(retrieval_topk)]
                    test_data[idx]["background00"] = [
                        "Title: "
                        + test_data[idx]["context"][rank][0]
                        + ". Context: "
                        + " ".join(test_data[idx]["context"][rank][1])
                        for rank in range(retrieval_topk)
                    ]
                except:
                    test_data[idx]["background00"] = [" "]
        elif data == "popqa" or data == "cwq":
            for idx in range(len(test_data)):
                try:
                    # test_data[idx]['background'] = [' '.join(test_data[idx]['context'][rank][1]) for rank in range(retrieval_topk)]
                    test_data[idx]["background00"] = [
                        test_data[idx]["passages"][rank]
                        for rank in range(retrieval_topk)
                    ]
                except:
                    test_data[idx]["background00"] = [" "]
        else:
            for idx in range(len(test_data)):
                try:
                    test_data[idx]["background00"] = [
                        test_data[idx]["background"][rank]
                        for rank in range(retrieval_topk)
                    ]
                except:
                    test_data[idx]["background00"] = [" "]

        for dd in test_data[:]:
            if dd["background00"] != [" "]:
                # try:
                dd[f"llmlingua2_r{retrieval_topk}_t32"] = compressor.compress_prompt('\n'.join(dd["background00"]), target_token=32, force_tokens = ['\n', '?'])['compressed_prompt']

            else:
                dd[f"llmlingua2_r{retrieval_topk}_t32"] = " "

    os.makedirs(f'data30/{data}', exist_ok=True)
    with open(f'data30/{data}/test.jsonl', 'w') as f:

        for d in test_data:
            del d['background00']
            f.write(json.dumps(d) + '\n')
