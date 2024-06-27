from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("F:\git\githubProject\ChatGLM2-6B\chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("F:\git\githubProject\ChatGLM2-6B\chatglm2-6b", trust_remote_code=True, device='cuda')
model = model.eval()
response, history = model.chat(tokenizer, "学猫叫", history=[])
print(response)