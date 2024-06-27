from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("F:\git\githubProject\ChatGLM2-6B\chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("F:\git\githubProject\ChatGLM2-6B\chatglm2-6b", trust_remote_code=True, device='cuda')
model = model.eval()
print(" === This is the model === ")
print(model)
print(" === And the tokenizer === ")
print(tokenizer)

response, history = model.chat(tokenizer, "你好", history=[])
print(" === First response === ")
print(response)
print(history)

response, history = model.chat(tokenizer, "你觉得AI是什么", history=history)
print(" === Second response === ")
print(response)