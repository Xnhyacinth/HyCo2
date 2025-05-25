gpu=${1:-"0"}
method=${2:-"cformer"}
num_query_tokens=${3:-"16"}
retrieval_prefix=${4:-"colbertv2"}
retrieval_topk=${5:-"1"}
test=${6:-"base"}
data=${7:-"triviaqa"}
model=${8:-"mistral"}
rec=${9:-"0"}
max_test_samples=${10:-"100"}
extra_args=""

logfile=res/${method}/${data}/${retrieval_prefix}_k${retrieval_topk}
model_path=checkpoint/finetune/${method}/last
if [ "$model" != "mistral" ];then
        logfile=res_${model}/${method}/${data}/${retrieval_prefix}_k${retrieval_topk}
        model_path=checkpoint_${model}/finetune/${method}/last
        extra_args="${extra_args} --chat_format ${model}"
fi
if [ "$test" = "xrag" ];then
        extra_args="${extra_args} --retriever_name_or_path models/Salesforce__SFR-Embedding-Mistral --use_rag"
fi
if [ "$test" = "rag" ];then
        extra_args="${extra_args} --use_rag"
        if [ "$model" = "qwen" ];then
                model_path=models/Qwen2.5-7B-Instruct
        fi
        if [ "$model" = "llama" ];then
                model_path=models/meta-llama/Meta-Llama-3.1-8B-Instruct
        fi
        if [ "$model" = "mistral" ];then
                model_path=models/mistralai__Mistral-7B-Instruct-v0.2
        fi
        logfile=res_${model}_ori/${method}/${data}/${retrieval_prefix}_k${retrieval_topk}
fi
if [ "$test" = "cformer" ];then
        extra_args="${extra_args} --cformer_name_or_path models/FacebookAI__roberta-base --num_query_tokens ${num_query_tokens} --use_rag"
        # logfile="${logfile}_n${num_query_tokens}"
fi
if [ "$test" = "gated" ];then
        extra_args="${extra_args} --num_query_tokens ${num_query_tokens} --use_rag"
        # logfile="${logfile}_n${num_query_tokens}"
fi
if [ "$test" = "gated_cat" ];then
        extra_args="${extra_args} --num_query_tokens ${num_query_tokens} --use_rag"
        # logfile="${logfile}_n${num_query_tokens}"
fi
if [ "$test" = "prompt" ];then
        extra_args="${extra_args} --use_rag"
        # logfile="${logfile}_n${num_query_tokens}"
fi
if [ "$test" = "tf_idf" ];then
        extra_args="${extra_args} --use_rag --tf_idf_topk ${retrieval_topk}"
        retrieval_topk=3
        # logfile="${logfile}_n${num_query_tokens}"
fi
if [ "$test" = "llmlingua2" ];then
        extra_args="${extra_args} --use_rag --baseline"
fi
if [ "$test" = "longllmlingua" ];then
        extra_args="${extra_args} --use_rag --baseline"
fi
if [ "$test" = "exit" ];then
        extra_args="${extra_args} --use_rag --baseline"
fi
if [ "$test" = "base" ];then
        if [ "$model" = "qwen" ];then
                model_path=models/Qwen2.5-7B-Instruct
        fi
        if [ "$model" = "llama" ];then
                model_path=models/meta-llama/Meta-Llama-3.1-8B-Instruct
        fi
        if [ "$model" = "mistral" ];then
                model_path=models/mistralai__Mistral-7B-Instruct-v0.2
        fi
        logfile=res_${model}_ori/${method}/${data}/${retrieval_prefix}_k${retrieval_topk}
fi

if [ "$rec" = "rec" ];then
        extra_args="${extra_args} --reconstruct"
fi
logfile=${logfile}_pro
mkdir -p ${logfile}


CUDA_VISIBLE_DEVICES=${gpu} python -m src.eval.eval \
        --data ${data} \
        --save_dir ${logfile} \
        --model_name_or_path ${model_path} \
        --projector_type ${method}  \
        --retrieval_prefix ${retrieval_prefix} \
        --retrieval_topk ${retrieval_topk} \
        --test ${test} \
        --max_test_samples ${max_test_samples} \
        ${extra_args} \
        > ${logfile}/log_${test}.log 2>&1 &


wait
