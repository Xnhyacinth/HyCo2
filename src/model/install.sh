pip install transformers==4.45.2
# pip install transformers==4.39.1
torch_path=$(python -c "import os; import torch; torch_dir = os.path.dirname(torch.__file__); print(torch_dir)")
echo $torch_path

cp src/model/functional.py $torch_path/nn
# cp src/model/linear.py $torch_path/nn/modules

transformers_path=$(python -c "import os; import transformers; transformers_dir = os.path.dirname(transformers.__file__); print(transformers_dir)")
echo $transformers_path

cp src/model/modeling_outputs.py $transformers_path
cp src/model/generic.py $transformers_path/utils