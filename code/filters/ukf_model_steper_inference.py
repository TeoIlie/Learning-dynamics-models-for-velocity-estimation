import torch
import torchdiffeq as ode
from robot_models.single_track_pacejka import observation
from filters.ukf import UKF
import numpy as np
from typing import Dict


class UKFModelStepperInference(torch.nn.Module):

    def __init__(self, model, noise_model, dt, solver_settngs, q_entr_lb, state_dim=5,
                 device=torch.device("cpu")) -> None:
        super().__init__()
        self.noise_model = noise_model
        self.model = model
        self.solver_settngs = solver_settngs
        self.dt = dt
        self.state_dim = state_dim
        self.device = device
        self.q_entr_lb = torch.tensor(q_entr_lb, device=device)
        self.ukf = UKF(state_dim, meas_dim=4, device=device)
        pass

    def _state_transition_func(self, x, *args):
        """
        x: [batch_size, sigma,  state_dim]
        """
        ur = args[0].unsqueeze(1).repeat(1, x.shape[1], 1)
        xu = torch.cat((x, ur), dim=-1)
        x_next = ode.odeint(self.model, xu, self.dt, **self.solver_settngs)[-1]
        x_next = x_next[..., : self.state_dim]
        return x_next

    def _observation_func(self, x, *args):
        """
        x: [batch_size, sigma,  state_dim]
        """
        ur = args[0].unsqueeze(1).repeat(1, x.shape[1], 1)
        xu = torch.cat((x, ur), dim=-1)
        return observation(self.model, xu)

    def calc_normal_entropy(self, M):
        D = M.shape[-1]
        return 0.5 * D * (1 + torch.log(torch.tensor([2 * torch.pi], device=M.device))) + 0.5 * torch.logdet(M)

    def project_entropy(self, M, lower_bound):
        M2 = M.clone()
        D = M.shape[-1]
        ent = self.calc_normal_entropy(M)
        indexes_to_project = ent < lower_bound
        proj_scaler = torch.exp((lower_bound - ent[indexes_to_project]) / D)
        proj_scaler = proj_scaler.unsqueeze(-1).unsqueeze(-1)
        M2[indexes_to_project, :, :] = proj_scaler * M[indexes_to_project, :, :]
        return M2

    def forward(self, X_hat, P, u, y, X_GT=None):
        """
        X_hat: [batch_size, state_dim]
        P: [batch_size, state_dim, state_dim]
        u: [batch_size, control_dim]
        y: [batch_size, meas_dim]
        X_gt [batch_size, state_dim]
        """

        if X_GT is not None:
            Q, R, _ = self.noise_model(X_GT)
        else:
            Q, R, _ = self.noise_model(X_hat)

        Q = self.project_entropy(Q, lower_bound=self.q_entr_lb)

        X_hat, P = self.ukf.predict(X_hat, P, Q,
                                    self._state_transition_func, u)

        X_hat, P = self.ukf.update(
            X_hat,
            P,
            y,
            R,
            self._observation_func,
            u,
        )

        return X_hat, P, self.calc_normal_entropy(Q), self.calc_normal_entropy(R)

    def P0(self, X):
        _, _, P = self.noise_model(X)
        return P
