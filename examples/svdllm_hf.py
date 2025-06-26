from svdllm.base import SVDModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./models/Qwen2-0.5B"
save_dir = "./models/Qwen2-0.5B-svd-v1"
device = "cuda:0"

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

SVDModel.compress(model, tokenizer, ratio=0.6, svd_version="v1", device=device)


inputs = tokenizer("Hello, this is a test.", return_tensors="pt").to(device)
outputs = model(**inputs)
print("outputs shape:", outputs.logits.shape)

SVDModel.save_model(model, save_dir)
tokenizer.save_pretrained(save_dir)


# model = SVDModel.from_compressed(save_dir, compute_dtype=torch.float16, device=device)
# inputs = tokenizer("Hello, this is a test.", return_tensors="pt").to(device)
# outputs = model(**inputs)
# print("outputs shape:", outputs.logits.shape)
