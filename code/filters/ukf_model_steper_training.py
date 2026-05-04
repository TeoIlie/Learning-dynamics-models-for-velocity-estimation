import torch
import torchdiffeq as ode
from robot_models.single_track_pacejka import observation
from filters.ukf import UKF
import numpy as np


class UKFModelStepperTrain(torch.nn.Module):
    def __init__(
            self, ukf_model_steper: torch.nn.Module) -> None:
        super().__init__()
        self.ukf_stepper = ukf_model_steper
        pass

    def forward(self, X0, u, y, X_GT=None):
        """
        X0: [batch_size, state_dim]
        u: [batch_size, time, control_dim]
        y: [batch_size, time, meas_dim]
        P0: [batch_size, state_dim, state_dim]
        Q: [batch_size, state_dim, state_dim]
        R: [batch_size, meas_dim, meas_dim]
        """
        device = X0.device
        current_batch_size = X0.shape[0]
        time_size = u.shape[1]
        P0 = self.ukf_stepper.P0(X0)

        P = P0.clone().unsqueeze(1).repeat(1, time_size, 1, 1)
        X = X0.clone().detach().unsqueeze(1).repeat(1, time_size, 1)
        q_entr = torch.zeros(current_batch_size, time_size, device=device)
        r_entr = torch.zeros(current_batch_size, time_size, device=device)

        for i in range(1, time_size):
            X[:, i], P[:, i], q_entr[:, i], r_entr[:, i] = self.ukf_stepper(X_hat=X[:, i - 1],
                                                                            P=P[:,
                                                                                i - 1],
                                                                            u=u[:,
                                                                                i - 1],
                                                                            y=y[:, i],
                                                                            X_GT=X_GT[:, i - 1, :] if X_GT is not None else None)

        return X, P, q_entr, r_entr
