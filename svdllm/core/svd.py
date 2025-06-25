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
        self.svd_u_proj = None
        self.svd_v_proj = None
        self.low_rank = None
        self.in_size = None
        self.out_size = None
        if initialize:
            self.initialize()

    def is_initialized(self):
        return False if (None in [self.svd_u_proj, self.svd_v_proj]) else True

    def initialize(self):
        if self.is_initialized():
            return

        W = self.linear_layer.weight.data.float().to(self.device)
        dtype = self.compute_dtype
        self.in_size = W.shape[0]
        self.out_size = W.shape[1]
        num_s_after_trunc = int(
            W.shape[0] * W.shape[1] * self.ratio / (W.shape[0] + W.shape[1])
        )
        self.low_rank = num_s_after_trunc

        scaling_diag_matrix = self.scaling_diag_matrix.to(self.device)
        try:
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
        except Exception:
            # TODO use verbose or log replace print
            print("Warning: scaling_diag_matrix is not full rank!")
            scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(
                self.device
            )
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)

        scaling_diag_matrix = scaling_diag_matrix.float()
        scaling_matrix_inv = scaling_matrix_inv.float()
        W_scale = torch.matmul(W, scaling_diag_matrix)
        U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
        truc_s = S[:num_s_after_trunc]
        truc_u = U[:, :num_s_after_trunc]
        truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
        truc_sigma = torch.diag(truc_s)
        sqrtSigma = torch.sqrt(truc_sigma)
        svd_u = torch.matmul(truc_u, sqrtSigma).to(dtype)
        svd_v = torch.matmul(sqrtSigma, truc_v).to(dtype)

        self.svd_u_proj = nn.Linear(
            self.low_rank, self.out_size, bias=False, dtype=dtype
        )
        self.svd_v_proj = nn.Linear(
            self.in_size, self.low_rank, bias=False, dtype=dtype
        )
        self.svd_u_proj.weight.data = svd_u
        self.svd_v_proj.weight.data = svd_v

        del self.linear_layer, scaling_diag_matrix

    def forward(self, x: torch.Tensor):
        return self.svd_u_proj(self.svd_v_proj(x))

    def state_dict_keys(self):
        return set(
            [
                "svd_u_proj",
                "svd_v_proj",
                "ratio",
                "in_size",
                "out_size",
                "low_rank",
            ]
        )

    def state_dict(self):
        if not self.is_initialized():
            return {k: None for k in self.state_dict_keys()}

        state = {
            "svd_u_proj": self.svd_u_proj.weight.data,
            "svd_v_proj": self.svd_v_proj.weight.data,
            "ratio": self.ratio,
            "in_size": self.in_size,
            "out_size": self.out_size,
            "low_rank": self.low_rank,
        }
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.ratio = state_dict.pop("ratio")
        self.in_size = state_dict.pop("in_size")
        self.out_size = state_dict.pop("out_size")
        self.low_rank = state_dict.pop("low_rank")

        self.svd_u_proj = nn.Linear(
            self.low_rank, self.out_size, bias=False, dtype=self.compute_dtype
        )
        self.svd_v_proj = nn.Linear(
            self.in_size, self.low_rank, bias=False, dtype=self.compute_dtype
        )
        self.svd_u_proj.weight.data = state_dict.pop("svd_u_proj")
        self.svd_v_proj.weight.data = state_dict.pop("svd_v_proj")
        self.svd_v_proj.to(device=self.device, dtype=self.compute_dtype)
        self.svd_u_proj.to(device=self.device, dtype=self.compute_dtype)


class SVDLinearV2(nn.Module):
    def __init__(
        self,
        linear_layer: nn.Module,
        calib_matrix: torch.Tensor,
        ratio: float,
        compute_dtype=torch.float16,
        device: str = "cpu",
        initialize: bool = True,
    ):
        super().__init__()
        self.linear_layer = linear_layer
        self.calib_matrix = calib_matrix
        self.ratio = ratio
        self.device = device
        self.compute_dtype = compute_dtype
        self.svd_u_proj = None
        self.svd_v_proj = None
        self.low_rank = None
        self.in_size = None
        self.out_size = None
        if initialize:
            self.initialize()

    def is_initialized(self):
        return False if (None in [self.svd_u_proj, self.svd_v_proj]) else True

    def initialize(self):
        if self.is_initialized():
            return

        W = self.linear_layer.weight.data.float().to(self.device)
        dtype = self.compute_dtype
        self.in_size = W.shape[0]
        self.out_size = W.shape[1]
        low_rank = int(W.shape[0] * W.shape[1] * self.ratio / (W.shape[0] + W.shape[1]))
        self.low_rank = low_rank

        XTX = self.calib_matrix.to(self.device)

        U_s, S_s, _ = torch.linalg.svd(XTX, full_matrices=False)
        U_ws, S_ws, _ = torch.linalg.svd(
            W @ U_s @ torch.diag(torch.sqrt(S_s)), full_matrices=False
        )
        S_s_inv = 1.0 / S_s
        try:
            U_s_inv = torch.linalg.inv(U_s.double()).float()
        except Exception:
            U_s += 1e-6 * torch.eye(U_s.shape[0]).to(self.device)
            U_s_inv = torch.linalg.inv(U_s.double()).float()

        truc_S_ws = torch.diag(S_ws[:low_rank])
        truc_U_ws = U_ws[:, :low_rank]

        truc_U_s_inv = U_s_inv[:low_rank, :]
        truc_S_s_inv = torch.diag(S_s_inv[:low_rank])

        svd_u = torch.matmul(truc_U_ws, truc_S_ws).to(dtype)
        svd_v = torch.matmul(
            truc_S_s_inv,
            truc_U_s_inv,
        ).to(dtype)
        self.svd_u_proj = nn.Linear(
            self.low_rank, self.out_size, bias=False, dtype=dtype
        )
        self.svd_v_proj = nn.Linear(
            self.in_size, self.low_rank, bias=False, dtype=dtype
        )
        self.svd_u_proj.weight.data = svd_u
        self.svd_v_proj.weight.data = svd_v

        del self.linear_layer, self.calib_matrix

    def forward(self, x: torch.Tensor):
        return self.svd_u_proj(self.svd_v_proj(x))

    def state_dict_keys(self):
        return set(
            [
                "svd_u_proj",
                "svd_v_proj",
                "ratio",
                "in_size",
                "out_size",
                "low_rank",
            ]
        )

    def state_dict(self):
        if not self.is_initialized():
            return {k: None for k in self.state_dict_keys()}

        state = {
            "svd_u_proj": self.svd_u_proj.weight.data,
            "svd_v_proj": self.svd_v_proj.weight.data,
            "ratio": self.ratio,
            "in_size": self.in_size,
            "out_size": self.out_size,
            "low_rank": self.low_rank,
        }
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.ratio = state_dict.pop("ratio")
        self.in_size = state_dict.pop("in_size")
        self.out_size = state_dict.pop("out_size")
        self.low_rank = state_dict.pop("low_rank")

        self.svd_u_proj = nn.Linear(
            self.low_rank, self.out_size, bias=False, dtype=self.compute_dtype
        )
        self.svd_v_proj = nn.Linear(
            self.in_size, self.low_rank, bias=False, dtype=self.compute_dtype
        )
        self.svd_u_proj.weight.data = state_dict.pop("svd_u_proj")
        self.svd_v_proj.weight.data = state_dict.pop("svd_v_proj")
        self.svd_v_proj.to(device=self.device, dtype=self.compute_dtype)
        self.svd_u_proj.to(device=self.device, dtype=self.compute_dtype)
