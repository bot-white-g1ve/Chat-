# 载入模型
from transformers import AutoTokenizer, AutoModel

checkpoint = "THUDM/chatglm-6b"
revision = "096f3de6b4959ce38bef7bb05f3129c931a3084e"
model = AutoModel.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)

# 配置LoRA
from peft import LoraConfig, get_peft_model, TaskType

def load_lora_config(model):
	config = LoraConfig(
	    task_type=TaskType.CAUSAL_LM, 
	    inference_mode=False,
	    r=8, 
	    lora_alpha=32, 
	    lora_dropout=0.1,
	    target_modules=["query_key_value"]
	)
	return get_peft_model(model, config)

model = load_lora_config(model)

print(" ===可训练参数量=== ")
model.print_trainable_parameters()

# 构建数据集
# 几个特殊Token
bos = tokenizer.bos_token_id
eop = tokenizer.eop_token_id
pad = tokenizer.pad_token_id
mask = tokenizer.mask_token_id
gmask = tokenizer.sp_tokenizer[tokenizer.gMASK_token]
# Setting
device = "cuda"
max_src_length = 200 #最大输入
max_dst_length = 500 #最大输出

#