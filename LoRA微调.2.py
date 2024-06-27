# 测试Tokenizer的特殊Token

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

print(" ===The Model=== ")
model = AutoModel.from_pretrained('F:\git\githubProject\ChatGLM2-6B\chatglm2-6b', trust_remote_code=True)
