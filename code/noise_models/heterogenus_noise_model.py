import torch


class HeteroCovNoiseModel(torch.nn.Module):
    def __init__(self, state_dim, meas_dim, device=torch.device("cpu")):

        super(HeteroCovNoiseModel, self).__init__()

        self.Q_P_min_eps = 1e-7

        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.device = device

        self.diag_Q = torch.diag(torch.tensor(
            [1e-3, 1e-3, 1e-2, 1e-2, 1e-5], device=device).sqrt())
        self.diag_R = torch.diag(torch.tensor(
            [5e1, 5e1, 1e-1, 1e-1], device=device).sqrt())
        self.diag_P0 = torch.diag(torch.tensor(
            [1e-3, 1e-3, 1e-3, 1e-3, 1e-5], device=device).sqrt())

        # const
        self.Qd = torch.nn.Parameter(self.diag_Q)
        self.Rd = torch.nn.Parameter(self.diag_R)
        self.P0d = torch.nn.Parameter(self.diag_P0)

        self.state_to_Q = torch.nn.Linear(state_dim, state_dim**2)
        for name, param in self.state_to_Q.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0.0, std=1e-3)
            elif "bias" in name:
                torch.nn.init.constant_(param, 0)

        self.state_to_R = torch.nn.Linear(state_dim, meas_dim**2)
        for name, param in self.state_to_R.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0.0, std=1e-3)
            elif "bias" in name:
                torch.nn.init.constant_(param, 0)

    def forward(self, x):
        """
            x [batch, time, state_dim]
            return Q, R, P0
        """

        LQ = torch.tril(self.Qd)
        Q = torch.mm(LQ, LQ.T) + torch.eye(self.state_dim, device=LQ.device) * self.Q_P_min_eps

        LR = torch.tril(self.Rd)
        R = torch.mm(LR, LR.T)

        LP0 = torch.tril(self.P0d)
        P0 = torch.mm(LP0, LP0.T) + torch.eye(self.state_dim, device=LP0.device) * \
            self.Q_P_min_eps

        current_batch_size = x.shape[0]
        Qb_const = Q.unsqueeze(0).repeat(current_batch_size, 1, 1)
        Rb_const = R.unsqueeze(0).repeat(current_batch_size, 1, 1)
        P0b_const = P0.unsqueeze(0).repeat(current_batch_size, 1, 1)

        x_state = x[:, :self.state_dim]
        L_Q_state = torch.tril(self.state_to_Q(
            x_state).reshape(-1, self.state_dim, self.state_dim))
        L_R_state = torch.tril(self.state_to_R(
            x_state).reshape(-1, self.meas_dim, self.meas_dim))

        Qb = Qb_const + torch.matmul(L_Q_state,
                                     L_Q_state.transpose(-1, -2)) * 0.01

        Rb = Rb_const + torch.matmul(L_R_state,
                                     L_R_state.transpose(-1, -2)) * 0.01

        return Qb, Rb, P0b_const
