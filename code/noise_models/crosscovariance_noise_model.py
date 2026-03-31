import torch


class CrossCovNoiseModel(torch.nn.Module):
    def __init__(self, state_dim, meas_dim, device=torch.device("cpu")):

        super(CrossCovNoiseModel, self).__init__()

        self.Q_P_min_eps = 1e-7

        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.device = device

        self.diag_Q = torch.diag(torch.tensor(
            [1e-3, 1e-3, 1e-2, 1e-2, 1e-4], device=device).sqrt())
        self.diag_R = torch.diag(torch.tensor(
            [5e1, 5e1, 1e-1, 1e-1], device=device).sqrt())
        self.diag_P0 = torch.diag(torch.tensor(
            [1e-3, 1e-3, 1e-3, 1e-3, 1e-4], device=device).sqrt())

        self.Qd = torch.nn.Parameter(self.diag_Q)
        self.Rd = torch.nn.Parameter(self.diag_R)
        self.P0d = torch.nn.Parameter(self.diag_P0)

    def forward(self, x_hat):
        """
            return Q, R, P0
        """
        current_batch_size = x_hat.shape[0]

        LQ = torch.tril(self.Qd)
        Q = torch.mm(LQ, LQ.T) + torch.eye(self.state_dim, device=LQ.device) * self.Q_P_min_eps

        LR = torch.tril(self.Rd)
        R = torch.mm(LR, LR.T)

        LP0 = torch.tril(self.P0d)
        P0 = torch.mm(LP0, LP0.T) + torch.eye(self.state_dim, device=LP0.device) * \
            self.Q_P_min_eps

        Qb = Q.unsqueeze(0).repeat(current_batch_size, 1, 1)
        Rb = R.unsqueeze(0).repeat(current_batch_size, 1, 1)
        P0b = P0.unsqueeze(0).repeat(current_batch_size, 1, 1)
        return Qb, Rb, P0b
