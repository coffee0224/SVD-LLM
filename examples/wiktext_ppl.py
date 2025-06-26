from transformers import AutoModelForCausalLM, AutoTokenizer
from svdllm.base import SVDModel
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM

import time


all_metrics = {}
device = "cuda:0"
metric = "word_perplexity"
task_name = "wikitext"
num_samples = 8

org_model_path = "./models/Qwen2-0.5B"
svd_model_path = "./models/Qwen2-0.5B-svd-v1"
tokenizer = AutoTokenizer.from_pretrained(org_model_path)

svd_model = SVDModel.from_compressed(
    svd_model_path,
    compute_dtype=torch.float16,
    device=device,
    attn_implementation="sdpa",
)

org_model = AutoModelForCausalLM.from_pretrained(
    org_model_path,
    device_map=device,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
)


def eval_ppl(model, tag):
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    task_manager = lm_eval.tasks.TaskManager()
    torch.cuda.reset_peak_memory_stats(device)
    start_load = time.time()

    results = lm_eval.simple_evaluate(  # call simple_evaluate
        model=lm,
        tasks=task_name,
        num_fewshot=0,
        task_manager=task_manager,
        log_samples=False,
        batch_size=1,
        limit=num_samples,
    )

    total_time = time.time() - start_load
    for key, value in results["results"][task_name].items():
        if key.startswith(metric + ","):
            print(f"{tag} {task_name}_{metric}: {value:.4f}")
    print(f"time: {total_time:.2f}s")


eval_ppl(org_model, org_model_path.split("/")[-1])
eval_ppl(svd_model, svd_model_path.split("/")[-1])
