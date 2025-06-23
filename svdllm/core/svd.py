from torch import nn
import torch


class SVDLinearV1(nn.Module):
    def __init__(
        self,
        linear_layer: nn.Module,
        scaling_diag_matrix: torch.Tensor,
        radio: float = 0.6,
        device: str = "cpu",
        initialize: bool = True,
    ):
        super().__init__()
        self.linear_layer = linear_layer
        self.scaling_diag_matrix = scaling_diag_matrix
        self.ratio = radio
        self.device = device
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
        dtype = W.dtype
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
        svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
        svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
        self.u_proj = svd_u
        self.v_proj = svd_v

        W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT = truc_s = (
            truc_u
        ) = truc_v = sqrtSigma = None
        del (
            W,
            W_scale,
            scaling_matrix_inv,
            scaling_diag_matrix,
            U,
            S,
            VT,
            truc_s,
            truc_u,
            truc_v,
            sqrtSigma,
            self.linear_layer,
            self.scaling_diag_matrix,
        )

    def forward(self, x: torch.Tensor):
        if not self.is_initialized():
            raise RuntimeError(
                "SVDLinearV1 is not initialized. Call initialize() first."
            )
        x.to(self.device)
        u_proj = self.u_proj.to(self.device)
        v_proj = self.v_proj.to(self.device)
        return u_proj @ (v_proj @ x)
