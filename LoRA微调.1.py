import torch
import pdb
# 载入模型
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModel.from_pretrained('F:\git\githubProject\ChatGLM2-6B\chatglm2-6b', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('F:\git\githubProject\ChatGLM2-6B\chatglm2-6b', trust_remote_code=True)

print(" ===原始模型=== ")
print(model)

config = LoraConfig(
    peft_type="LORA",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    fan_in_fan_out=False,
    bias='lora_only',
    target_modules=["query_key_value"]
)

model = get_peft_model(model, config)
print(" ===LoRA模型=== ")
print(model)