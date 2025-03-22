conda activate sum_env
conda deactivate
python summarization_DS.py

v100不支持flash-attention
flash-attention环境需要提前在虚拟环境内编译（如果服务器总挂就晚上再试）

GLM:
source /root/DiabetesPDiagLLM/.diabetesPDiagLLMVenv/bin/activates