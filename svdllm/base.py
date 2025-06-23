from os.path import join as pjoin
from typing import Union, Callable
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils.data_utils import get_calib_train_data
from .utils.model_utils import find_layers

# TODO add SVDLinearV2
from .core.svd import SVDLinearV1 as SVDLinear


class SVDModel:
    @classmethod
    def get_config_file(cls, save_dir: str) -> str:
        return pjoin(save_dir, "config.json")

    @classmethod
    def get_weight_file(cls, save_dir: str) -> str:
        return pjoin(save_dir, "qmodel.pt")

    @classmethod
    def compress(
        cls,
        model,
        tokenizer,
        ratio: Union[float, dict],
        calib_dataset: str = "wikitext2",
        whitening_nsamples: int = 256,
        seqlen: int = 2048,
        device: str = "cpu",
    ):
        calib_data = get_calib_train_data(
            calib_dataset, tokenizer, whitening_nsamples, seqlen, device
        )
        whitening_mat = cls.get_whitening_mat(model, calib_data, device)

        def _patch_linear(linear_layer, mat, ratio):
            if type(linear_layer) is SVDLinear:
                return linear_layer
            out_module = SVDLinear(linear_layer, mat, ratio, device)
            return out_module

        cls.patch_model(model, whitening_mat, ratio, _patch_linear)

        model.svd_compressed = True

        return model

    @classmethod
    def patch_model(cls, model, whitening_mat, ratio, patch_linear_fn: Callable):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        try:
            for param in model.model.parameters():
                param.requires_grad = False
        except Exception:
            pass

        layers = model.model.layers
        for i in tqdm(range(len(layers)), desc="Patching model"):
            layer = layers[i]
            subset = find_layers(layer)

            def find_parent(model, name: str) -> nn.Module:
                module_tree = name.split(".")[:-1]
                parent = model
                for m in module_tree:
                    parent = parent._modules[m]
                return parent

            for name, module in subset.items():
                mat = whitening_mat[i][name]

                if isinstance(module, nn.Linear):
                    setattr(
                        find_parent(layer, name),
                        name.split(".")[-1],
                        patch_linear_fn(module, mat, ratio),
                    )
        return model

    @classmethod
    def get_whitening_mat(cls, model, calib_data, device):
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1, 2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.raw_scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum
            torch.cuda.empty_cache()

        layers = model.model.layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.raw_scaling_diag_matrix = 0
                module.register_forward_hook(hook)
        for batch in tqdm(calib_data, desc="Profiling batch data"):
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module._forward_hooks.clear()
        torch.cuda.empty_cache()
        model = model.cpu()
        for i in range(len(layers)):
            subset = find_layers(layers[i])
            for name in subset:
                subset[name].raw_scaling_diag_matrix = subset[
                    name
                ].raw_scaling_diag_matrix.cpu()
        profiling_mat = {}
        for i in tqdm(range(len(layers)), desc="Whitening data"):
            layer_profile = {}
            subset = find_layers(layers[i])
            for name in subset:
                raw_scaling_diag_matrix = (
                    subset[name].raw_scaling_diag_matrix.double().to(device)
                )
                try:
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                except Exception as e:
                    # TODO use log replace print
                    print("Warning: eigen scaling_diag_matrix is not positive!")
                    eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                    raw_scaling_diag_matrix += (-eigenvalues[0] + 1e-6) * torch.eye(
                        raw_scaling_diag_matrix.shape[0]
                    ).to(device)
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                    eigenvalues = None
                    del eigenvalues
                layer_profile[name] = scaling_diag_matrix.cpu()
                scaling_diag_matrix = raw_scaling_diag_matrix = subset[
                    name
                ].raw_scaling_diag_matrix = None
                del (
                    scaling_diag_matrix,
                    raw_scaling_diag_matrix,
                    subset[name].raw_scaling_diag_matrix,
                )
                torch.cuda.empty_cache()
            profiling_mat[i] = layer_profile
        return profiling_mat
