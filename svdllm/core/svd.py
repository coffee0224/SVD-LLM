from torch import nn
import torch


class SVDLinearV1(nn.Module):
    def __init__(
        self,
        linear_layer: nn.Module,
        scaling_diag_matrix: torch.Tensor,
        ratio: float,
        compute_dtype=torch.float16,
        device: str = "cpu",
        initialize: bool = True,
    ):
        super().__init__()
        self.linear_layer = linear_layer
        self.scaling_diag_matrix = scaling_diag_matrix
        self.ratio = ratio
        self.device = device
        self.compute_dtype = compute_dtype
        self.u_proj = None
        self.v_proj = None
        if initialize:
            self.initialize()

    def is_initialized(self):
        return False if (None in [self.u_proj, self.v_proj]) else True

    def initialize(self):
        if self.is_initialized():
            return

        W = self.linear_layer.weight.data.float().to(self.device)
        dtype = self.compute_dtype
        scaling_diag_matrix = self.scaling_diag_matrix.to(self.device)
        try:
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
        except Exception as e:
            print("Warning: scaling_diag_matrix is not full rank!")
            scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(
                self.device
            )
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)

        scaling_diag_matrix = scaling_diag_matrix.float()
        scaling_matrix_inv = scaling_matrix_inv.float()
        W_scale = torch.matmul(W, scaling_diag_matrix)
        U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
        num_s_after_trunc = int(
            W.shape[0] * W.shape[1] * self.ratio / (W.shape[0] + W.shape[1])
        )
        truc_s = S[:num_s_after_trunc]
        truc_u = U[:, :num_s_after_trunc]
        truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
        truc_sigma = torch.diag(truc_s)
        #### Replace Attn, MLP ####
        sqrtSigma = torch.sqrt(truc_sigma)
        svd_u = torch.matmul(truc_u, sqrtSigma).to(dtype)
        svd_v = torch.matmul(sqrtSigma, truc_v).to(dtype)
        self.u_proj = svd_u
        self.v_proj = svd_v

    def forward(self, x: torch.Tensor):
        return (x @ self.u_proj) @ self.v_proj

    def state_dict_keys(self):
        return set(
            [
                "u_proj",
                "v_proj",
                "ratio",
            ]
        )

    def state_dict(self):
        if not self.is_initialized():
            return {k: None for k in self.state_dict_keys()}

        state = {
            "u_proj": self.u_proj,
            "v_proj": self.v_proj,
            "ratio": self.ratio,
        }
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.u_proj = state_dict.pop("u_proj")
        self.v_proj = state_dict.pop("v_proj")
        self.ratio = state_dict.pop("ratio")
        self.compute_dtype = state_dict.pop("compute_dtype")
        self.u_proj = self.u_proj.to(self.compute_dtype)
        self.v_proj = self.v_proj.to(self.compute_dtype)
        self.u_proj.to(self.device)
        self.v_proj.to(self.device)
