from os.path import join as pjoin
from typing import Union, Callable
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import transformers
from accelerate import init_empty_weights


from .utils.data_utils import get_calib_train_data
from .utils.model_utils import find_layers

# TODO add SVDLinearV2
from .core.svd import SVDLinearV1 as SVDLinear

_COMPRESSED_LAYERS = [nn.Linear, SVDLinear]
_SVD_MODULES = [SVDLinear]


def is_leaf_module(module) -> bool:
    return len(module._modules) == 0


def is_svd_module(module) -> bool:
    return type(module) in _SVD_MODULES


class SVDModel:
    @classmethod
    def compress(
        cls,
        model,
        tokenizer,
        ratio: float,
        calib_dataset: str = "wikitext2",
        whitening_nsamples: int = 256,
        seqlen: int = 2048,
        compute_dtype=torch.float16,
        device: str = "cpu",
    ):
        calib_data = get_calib_train_data(
            calib_dataset, tokenizer, whitening_nsamples, seqlen, device
        )
        whitening_mat = cls.get_whitening_mat(model, calib_data, device)

        def _patch_linear(linear_layer, matrix=None):
            if type(linear_layer) is SVDLinear:
                return linear_layer
            out_module = SVDLinear(
                linear_layer,
                matrix,
                ratio,
                compute_dtype=compute_dtype,
                device=device,
            )
            return out_module

        def _patch_other(layer):
            return layer.to(device=device, dtype=compute_dtype)

        cls.patch_model(model, whitening_mat, _patch_linear, _patch_other)

        model.svd_compressed = True

        model.base_class = cls

        return model

    @classmethod
    def patch_model(
        cls, model, mat_dict, patch_linear_fn: Callable, patch_other_fn: Callable
    ):
        def find_parent(model, name: str) -> nn.Module:
            module_tree = name.split(".")[:-1]
            parent = model
            for m in module_tree:
                parent = parent._modules[m]
            return parent

        model.eval()
        cls.autoname_modules(model)
        for param in model.parameters():
            param.requires_grad = False
        try:
            for param in model.model.parameters():
                param.requires_grad = False
        except Exception:
            pass

        layers = model.model.layers
        for i in tqdm(range(len(layers)), desc="Patching linear model"):
            layer = layers[i]
            subset = find_layers(layer)

            for name, module in subset.items():
                mat = mat_dict[i][name] if mat_dict else None
                if isinstance(module, nn.Linear):
                    setattr(
                        find_parent(layer, name),
                        name.split(".")[-1],
                        patch_linear_fn(module, mat),
                    )

        ignore_tags = cls.get_ignore_layers(model)
        tmp_mapping = {}
        for name, module in model.named_modules():
            if (name not in ignore_tags) and (
                name == "lm_head" or (type(module) not in _COMPRESSED_LAYERS)
            ):
                tmp_mapping[name] = module

        for name in tqdm(tmp_mapping, desc="Patching other modules"):
            setattr(
                find_parent(model, name),
                name.split(".")[-1],
                patch_other_fn(tmp_mapping[name]),
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
                layer_profile[name] = scaling_diag_matrix
            profiling_mat[i] = layer_profile
        return profiling_mat

    @classmethod
    def save_model(cls, model, save_dir: str):
        # Save config
        cls.cache_model(model, save_dir)
        # Serialization
        weights = cls.serialize_weights(model)
        # Save
        cls.save_weights(weights, save_dir)

    @classmethod
    def cache_model(cls, model, save_dir):
        # Update model architecture in the config
        model.config.architectures = [model.__class__.__name__]
        # Save config
        model.config.save_pretrained(save_dir)

    # Create empty model from config
    @classmethod
    def create_model(cls, save_dir):
        config = transformers.AutoConfig.from_pretrained(cls.get_config_file(save_dir))

        auto_class = transformers.AutoModel

        # Todo: add support for other auto models
        archs = config.architectures
        if len(archs) == 1:
            if "CausalLM" in archs[0]:
                auto_class = transformers.AutoModelForCausalLM
            elif "SequenceClassification" in archs[0]:
                auto_class = transformers.AutoModelForSequenceClassification

        with init_empty_weights():
            model = auto_class.from_config(config)

        return model

    # Prepares model weights by iterating through modules. It might some parameters that are NOT modules like model.param1
    @classmethod
    def serialize_weights(cls, model, verbose: bool = False) -> dict:
        weights = {}
        ignore_keys = cls.get_ignore_layers(model)
        for name, module in model.named_modules():
            if name in ignore_keys:
                continue
            try:
                # disable state_dict encoding for safetensors
                module.encoded_state_dict = False
                state_dict = module.state_dict()

                if len(state_dict) > 0:
                    weights[name] = dict(state_dict)
            except Exception:
                if verbose:
                    print("Skipping", name)

        return weights

    @classmethod
    def get_ignore_layers(cls, model) -> list:
        layers = {""}
        for name, module in model.named_modules():
            if not is_leaf_module(module) and not is_svd_module(module):
                layers.add(name)
        return list(layers)

    # Save weights to disk
    @classmethod
    def save_weights(cls, weights: dict, save_dir: str) -> None:
        torch.save(weights, cls.get_weight_file(save_dir))

    @classmethod
    def get_config_file(cls, save_dir: str) -> str:
        return pjoin(save_dir, "config.json")

    @classmethod
    def get_weight_file(cls, save_dir: str) -> str:
        return pjoin(save_dir, "model.pt")

    @classmethod
    def from_compressed(
        cls, save_dir: str, compute_dtype=torch.float16, device: str = "cuda"
    ):
        # Check
        if not os.path.exists(cls.get_weight_file(save_dir)):
            raise Exception("Weight file missing. Check your cache directory.")
        if not os.path.exists(cls.get_config_file(save_dir)):
            raise Exception("Config file missing. Check your cache directory.")

        # Load model from config
        model = cls.create_model(save_dir)

        # Track save directory
        model.save_dir = save_dir

        # Load weights
        try:
            weights = cls.load_weights(save_dir)
        except Exception:
            print("Failed to load the weights")
            raise FileNotFoundError

        # load_state_dict() doesn't work with modules initialized with init_empty_weights(), so we need to do this manually
        @torch.no_grad()
        def _load_module(module, mat=None):
            if module.name not in weights:
                return module.to(device=device, non_blocking=True)

            state_dict = weights[module.name]
            if "ratio" in state_dict:
                module = SVDLinear(
                    linear_layer=None,
                    scaling_diag_matrix=None,
                    ratio=None,
                    device=device,
                    compute_dtype=compute_dtype,
                    initialize=False,
                )
                module.load_state_dict(state_dict)
            else:
                for key in state_dict:
                    setattr(
                        module,
                        key,
                        nn.Parameter(
                            state_dict[key].to(
                                device=device, dtype=compute_dtype, non_blocking=True
                            ),
                            requires_grad=False,
                        ),
                    )

            return module

        cls.patch_model(model, None, _load_module, _load_module)

        model.svd_compressed = True

        # Set base class
        model.base_class = cls

        return model

    # Load weights from disk
    @classmethod
    def load_weights(cls, save_dir: str, map_location=None):
        return torch.load(
            cls.get_weight_file(save_dir), map_location=map_location, weights_only=True
        )

    # Autmatically name modules. This is very important to save/load the weights
    @classmethod
    def autoname_modules(cls, model) -> None:
        for name, module in model.named_modules():
            module.name = name
