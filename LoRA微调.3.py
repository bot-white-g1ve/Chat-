# 载入模型
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('F:\git\githubProject\ChatGLM2-6B\chatglm2-6b', trust_remote_code=True)

prompt = tokenizer.build_prompt("AI是什么？", None)
print(" ===Outcome of build_prompt=== ")
print(prompt)

print(" ===Special Tokens=== ")
print("BOS token ID:", tokenizer.get_command("<bos>"))
print("EOS token ID:", tokenizer.get_command("<eos>"))
print("PAD token ID:", tokenizer.get_command("<pad>"))
print("SOP token ID:", tokenizer.get_command("sop"))
print("EOP token ID:", tokenizer.get_command("eop"))
print("MASK token ID:", tokenizer.get_command("[MASK]"))
print("GMASK token ID:", tokenizer.get_command("[gMASK]"))

max_source_length = 100
max_target_length = 200
q_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                      max_length=max_source_length)
print(" ===Question ids using encode with special tokens=== ")
print(q_ids)
a_ids = tokenizer.encode(text="AI是人工智能(Artificial Intelligence)的缩写", add_special_tokens=False, truncation=True,
                                      max_length=max_target_length)
print(" ===Answer ids===")
print(a_ids)

# print(" ===Question ids using tokenizer() without argument=== ")
# q_ids2 = tokenizer([prompt])
# print(q_ids2)
# 结果是q_ids2的input_ids和q_ids一致，但是多了attention_mask, position_ids之类的

print(" !!!Loading Model!!! ")
import torch
model = AutoModel.from_pretrained("F:\git\githubProject\ChatGLM2-6B\chatglm2-6b", trust_remote_code=True, device='cuda')
max_source_length = 128
max_target_length = 128
epochs=5
batch_size=1
lr = 1e-4
lora_r=8
device=torch.device("cuda")

from peft import LoraConfig, get_peft_model, TaskType
peft_config=LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=lora_r,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model=model.half()
model.float()

print(" !!!Start Training!!! ")
import time
model.train()
start_time = time.time()
print(" ===Query=== ")
query = "学猫叫"
print(query)
print(" ===Prompt=== ")
prompt = tokenizer.build_prompt(query, None)
print(prompt)
print(" ===Inputs=== ")
inputs = tokenizer([prompt], return_tensors="pt")
print(inputs.tolist())
inputs = inputs.to(device)
