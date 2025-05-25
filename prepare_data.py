#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import random

## prepare data
import datasets

# In[2]:


templates_for_qa = [
    "Question: {question}?\nAnswer:",
    "{question}?",
    "Answer the following question:\n\n{question}",
    "Answer this question:\n\n{question}?",
    "Please answer this question: {question}",
    "Answer the question...{question}?",
    "What is the answer to this question? {question}\n\n",
    "Can you tell me the answer to {question}?",
    "Next question: {question}\n\n",
    "Q: {question} A:",
    "{question}\nWhat is the answer?",
    "Write the answer: {question}",
    "{question}???",
]

templates_for_sum = [
    "Write a short summary for the text\n\nSummary:",
    "Briefly summarize this article:\nSummary:",
    "What is a shorter version of this:\n\nSummary:",
    "Write a brief summary in a sentence or less.",
    "What is a very short summary of the above text?",
    "Summarize the aforementioned text in a single phrase.",
    "Can you generate a short summary of the above paragraph?",
    "Summarize the above articles\n\ntl;dr:",
]
template_for_fact_checking = [
    'Verify the following claims with "True" or "False":\n{question}',
]
# ipython -c "%run prepare_data.ipynb"


# In[3]:


total_data = []


# In[4]:


## commonsense_qa
data = datasets.load_dataset("commonsense_qa")
print(len(data["train"]))
for idx, sample in enumerate(data["train"]):
    question = sample["question"] + "\n\n"
    for choice, text in zip(sample["choices"]["label"], sample["choices"]["text"]):
        question += choice + ". " + text + "\n"

    question = random.choice(templates_for_qa).format_map(dict(question=question))
    answer = sample["answerKey"]

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    total_data.append(
        {
            "id": f"commonsense_qa_{idx}",
            "messages": messages,
            "task_type": "open_qa",
        }
    )


# In[5]:


## webqa
data = datasets.load_dataset("web_questions")
print(len(data["train"]))
for idx, sample in enumerate(data["train"]):
    question = sample["question"] + "\n"
    question = random.choice(templates_for_qa).format_map(dict(question=question))
    answer = sample["answers"][0]

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    total_data.append(
        {
            "id": f"web_questions_{idx}",
            "messages": messages,
            "task_type": "open_qa",
        }
    )
print(total_data[-1])


# In[6]:


## wikiqa
data = datasets.load_dataset("wiki_qa")
print(len(data["train"]))
for idx, sample in enumerate(data["train"]):
    if sample["label"] == 0:
        continue
    question = sample["question"] + "\n"
    question = random.choice(templates_for_qa).format_map(dict(question=question))
    answer = sample["answer"]

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    total_data.append(
        {
            "id": f"wiki_qa_{idx}",
            "messages": messages,
            "task_type": "open_qa",
        }
    )
print(total_data[-1])


# In[7]:


## yahoo_qa
data = datasets.load_dataset("yahoo_answers_qa")
print(len(data["train"]))
print(data["train"][0])
for idx, sample in enumerate(data["train"]):
    question = sample["question"] + "\n"
    question = random.choice(templates_for_qa).format_map(dict(question=question))
    answer = sample["answer"]

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    total_data.append(
        {
            "id": f"yahoo_answers_qa_{idx}",
            "messages": messages,
            "task_type": "open_qa",
        }
    )
print(total_data[-1])


# In[8]:


## freebase_qa
data = datasets.load_dataset("freebase_qa")
print(len(data["train"]))
print(data["train"][0])
for idx, sample in enumerate(data["train"]):
    question = sample["RawQuestion"] + "\n"
    question = random.choice(templates_for_qa).format_map(dict(question=question))
    answer = sample["Parses"]["Answers"][0]["AnswersName"][0][0]

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    total_data.append(
        {
            "id": f"freebase_qa_{idx}",
            "messages": messages,
            "task_type": "open_qa",
        }
    )
print(total_data[-1])


# In[9]:


## ms_marco
data = datasets.load_dataset("ms_marco", "v2.1")
data = list(data["train"])
# print(len(data['train']))
# print(data['train'][0])
for idx, sample in enumerate(data[:100_000]):

    question = sample["query"].lstrip(")") + "\n"
    question = random.choice(templates_for_qa).format_map(dict(question=question))
    answer = sample["answers"][0]

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    total_data.append(
        {
            "id": f"ms_marco_{idx}",
            "messages": messages,
            "task_type": "open_qa",
        }
    )
print(total_data[-1])


# In[10]:


## coqa
data = datasets.load_dataset("coqa")
print(len(data["train"]))
print(data["train"][0])
for idx, sample in enumerate(data["train"]):
    messages = []
    assert len(sample["answers"]["input_text"]) == len(sample["questions"])
    for idx, (q, a) in enumerate(
        zip(sample["questions"], sample["answers"]["input_text"])
    ):

        question = q + "\n"
        if idx == 0:
            question = random.choice(templates_for_qa).format_map(
                dict(question=question)
            )
        answer = a

        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

    total_data.append(
        {
            "id": f"coqa_{idx}",
            "messages": messages,
            "task_type": "close_qa",
            "background": sample["story"],
        }
    )
print(total_data[-1])


# In[4]:


## drop
# import datasets
data = datasets.load_dataset("drop")
print(len(data["train"]))
print(data["train"][0])
for idx, sample in enumerate(data["train"]):
    messages = []
    question = sample["question"] + "\n"
    question = random.choice(templates_for_qa).format_map(dict(question=question))
    answer = sample["answers_spans"]["spans"][0]

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    total_data.append(
        {
            "id": f"drop_{idx}",
            "messages": messages,
            "task_type": "close_qa",
            "background": sample["passage"],
        }
    )
print(total_data[-1])


# In[12]:


## narrativeqa
data = datasets.load_dataset("narrativeqa")
print(len(data["train"]))
print(data["train"][0])
for idx, sample in enumerate(data["train"]):
    messages = []
    question = sample["question"]["text"]
    answer = sample["answers"][0]["text"]
    question = random.choice(templates_for_qa).format_map(dict(question=question))

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    total_data.append(
        {
            "id": f"narrativeqa_{idx}",
            "messages": messages,
            "task_type": "close_qa",
            "background": sample["document"]["summary"]["text"],
        }
    )
print(total_data[-1])


# In[13]:


## pubmed_qa
data = datasets.load_dataset("pubmed_qa", "pqa_labeled")
print(len(data["train"]))
print(data["train"][0])
for idx, sample in enumerate(data["train"]):
    messages = []
    question = sample["question"]
    answer = (
        sample["long_answer"] + "So the final answer is: " + sample["final_decision"]
    )
    question = random.choice(templates_for_qa).format_map(dict(question=question))

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    total_data.append(
        {
            "id": f"pubmed_qa_{idx}",
            "messages": messages,
            "task_type": "close_qa",
            "background": "\n".join(sample["context"]["contexts"]),
        }
    )
print(total_data[-1])


# In[14]:


## quail
data = datasets.load_dataset("quail")
print(len(data["train"]))
print(data["train"][0])
for idx, sample in enumerate(data["train"]):
    messages = []
    question = sample["question"] + "\n"
    for answer_id, answer in enumerate(sample["answers"]):
        question += ["A. ", "B. ", "C. ", "D. "][answer_id] + answer + "\n"
    answer = ["A", "B", "C", "D"][sample["correct_answer_id"]]
    question = random.choice(templates_for_qa).format_map(dict(question=question))

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    total_data.append(
        {
            "id": f"quail_{idx}",
            "messages": messages,
            "task_type": "close_qa",
            "background": sample["context"],
        }
    )
total_data[-1]


# In[15]:


## squad_v2
data = datasets.load_dataset("squad_v2")
print(len(data["train"]))
print(data["train"][0])
for idx, sample in enumerate(data["train"]):
    messages = []
    question = sample["question"]
    answer = (
        sample["answers"]["text"][0]
        if len(sample["answers"]["text"]) > 0
        else "I don't know."
    )
    question = random.choice(templates_for_qa).format_map(dict(question=question))

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    total_data.append(
        {
            "id": f"squad_v2_{idx}",
            "messages": messages,
            "task_type": "close_qa",
            "background": sample["context"],
        }
    )
total_data[-1]


# In[16]:


## cnn_dm
data = datasets.load_dataset("cnn_dailymail", "3.0.0")
data = list(data["train"])
for idx, sample in enumerate(data[:10_0000]):
    messages = []
    answer = sample["highlights"]
    question = random.choice(templates_for_sum)

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    total_data.append(
        {
            "id": f"cnn_dailymail_{idx}",
            "messages": messages,
            "task_type": "summarization",
            "background": sample["article"],
        }
    )
total_data[-1]


# In[17]:


## samsum
dataset = datasets.load_dataset("samsum")
print(len(dataset["train"]))
print(dataset["train"][0])
# "samsum_6054" empty background
for idx, sample in enumerate(dataset["train"]):
    messages = []
    answer = sample["summary"]
    question = random.choice(templates_for_sum)

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    total_data.append(
        {
            "id": f"samsum_{idx}",
            "messages": messages,
            "task_type": "summarization",
            "background": sample["dialogue"].replace("\r\n", "\n"),
        }
    )
total_data[-1]


# In[18]:


## dialogsum
dataset = datasets.load_dataset("knkarthick/dialogsum")
print(len(dataset["train"]))
print(dataset["train"][0])
for idx, sample in enumerate(dataset["train"]):
    messages = []
    answer = sample["summary"]
    question = random.choice(templates_for_sum)

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    total_data.append(
        {
            "id": f"dialogsum_{idx}",
            "messages": messages,
            "task_type": "summarization",
            "background": sample["dialogue"],
        }
    )
total_data[-1]


# In[19]:


## pwc
import json

# dataset = [json.loads(x) for x in open("data/pwc/PwC_train.jsonl").readlines()]
dataset = datasets.load_dataset("sggetao/PwC")
print(len(dataset["train"]))
print(dataset["train"][0])
for idx, sample in enumerate(dataset):
    messages = []
    answer = sample["answer"]
    question = sample["prompt"]

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    total_data.append(
        {
            "id": f"pwc_{idx}",
            "messages": messages,
            "task_type": "close_qa",
            "background": sample["input"],
        }
    )
total_data[-1]


# In[20]:


## nq_open
dataset = datasets.load_dataset("nq_open")
print(len(dataset["train"]))
print(dataset["train"][0])
for idx, sample in enumerate(dataset["train"]):
    messages = []
    answer = sample["answer"][0]
    question = sample["question"]
    question = random.choice(templates_for_qa).format_map(dict(question=question))

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    total_data.append(
        {
            "id": f"nq_{idx}",
            "messages": messages,
            "task_type": "open_qa",
        }
    )
total_data[-1]


# In[21]:


## fm2
fm2 = [json.loads(x) for x in open("data/eval/fm2/fm2-train.jsonl").readlines()]
print(len(fm2))
for idx, sample in enumerate(fm2):
    question = sample["question"]
    messages = [
        {
            "role": "user",
            "content": template_for_fact_checking[0].format_map(
                dict(question=question)
            ),
        },
        {
            "role": "assistant",
            "content": "True" if "supports" in sample["answer"] else "False",
        },
    ]
    total_data.append(
        {
            "id": f"fm2_{idx}",
            "task_type": "fact_checking",
            "messages": messages,
        }
    )


# In[22]:


## triviaqa
tqa = [json.loads(x) for x in open("data/eval/triviaqa/tqa-train.jsonl").readlines()]
print(len(tqa))
for idx, sample in enumerate(tqa):
    messages = []
    answer = sample["answer"][0]
    question = sample["question"]
    question = random.choice(templates_for_qa).format_map(dict(question=question))

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    total_data.append(
        {
            "id": f"triviaqa_{idx}",
            "messages": messages,
            "task_type": "open_qa",
        }
    )


# In[ ]:


import json
import os

os.makedirs("data/instruction_tuning/processed", exist_ok=True)
with open(
    "data/instruction_tuning/processed/context_aware_instrution_tuning_data.jsonl", "w"
) as f:
    json.dump(total_data, f, indent=4)
