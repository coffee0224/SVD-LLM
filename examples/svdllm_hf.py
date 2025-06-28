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

# all linear layers will be compressed at ratio
# ratio = 0.6

# all model.layers.*.mlp.down/up/gate_proj will be compressed at ratio
ratio = {
    "mlp.down_proj": 0.2,
    "mlp.up_proj": 0.2,
    "mlp.gate_proj": 0.2,
    "self_attn.q_proj": 0.8,
    "self_attn.k_proj": 0.8,
    "self_attn.v_proj": 0.8,
    "self_attn.o_proj": 0.8,
}

# all model.layers.*.mlp.down/up/gate_proj will be compressed at ratio 0.8 except model.layers.0
# ratio = {
#     "model.layers.0.mlp.down_proj": 0.6,
#     "model.layers.0.mlp.up_proj": 0.6,
#     "model.layers.0.mlp.gate_proj": 0.6,
#     "mlp.gate_proj": 0.2,
#     "mlp.up_proj": 0.2,
#     "mlp.down_proj": 0.2,
# }


SVDModel.compress(model, tokenizer, ratio=ratio, svd_version="v1", device=device)


inputs = tokenizer("Hello, this is a test.", return_tensors="pt").to(device)
outputs = model(**inputs)
print("outputs shape:", outputs.logits.shape)

SVDModel.save_model(model, save_dir)
tokenizer.save_pretrained(save_dir)


# model = SVDModel.from_compressed(save_dir, compute_dtype=torch.float16, device=device)
# inputs = tokenizer("Hello, this is a test.", return_tensors="pt").to(device)
# outputs = model(**inputs)
# print("outputs shape:", outputs.logits.shape)
