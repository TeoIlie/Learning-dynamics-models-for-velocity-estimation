import torch


class ConstDiagonalNoiseModel(torch.nn.Module):
    def __init__(self, state_dim, meas_dim, device=torch.device("cpu")):

        super(ConstDiagonalNoiseModel, self).__init__()

        self.Q_P_min_eps = 1e-7
        self.P0_val = 1e-3

        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.device = device

        self.Qdi = torch.tensor(
            [1e-3, 1e-3, 1e-2, 1e-2, 1e-2], device=device).sqrt()
        self.Rdi = torch.tensor(
            [5e1, 5e1, 1e-1, 1e-1, 1e-1], device=device).sqrt()
        self.P0di = torch.tensor(
            [1e-3, 1e-3, 1e-3, 1e-3, 1e-3], device=device).sqrt()

        self.Qd = torch.nn.Parameter(self.Qdi)
        self.Rd = torch.nn.Parameter(self.Rdi)
        self.P0d = torch.nn.Parameter(self.P0di)

    def forward(self, x_hat):
        """
            return Q, R, P0
        """
        current_batch_size = x_hat.shape[0]
        Q = torch.diag(self.Qd**2) + \
            torch.eye(self.state_dim, device=self.Qd.device) * self.Q_P_min_eps
        R = torch.diag(self.Rd**2)
        P0 = torch.diag(self.P0d**2) + \
            torch.eye(self.state_dim, device=self.P0d.device) * self.Q_P_min_eps
        Qb = Q.unsqueeze(0).repeat(current_batch_size, 1, 1)
        Rb = R.unsqueeze(0).repeat(current_batch_size, 1, 1)
        P0b = P0.unsqueeze(0).repeat(current_batch_size, 1, 1)
        return Qb, Rb, P0b
