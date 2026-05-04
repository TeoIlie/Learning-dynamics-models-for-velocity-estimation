import torch
from utils.state_wrapper import STATE_DEF_LIST_SHORT, StateWrapper
import collections


class SingleTrackPacejkaModel(torch.nn.Module):

    def __init__(self, vehicle_parameters: torch.nn.Module, tire_model: torch.nn.Module) -> None:
        super(SingleTrackPacejkaModel, self).__init__()
        self.tire_model = tire_model
        self.p = vehicle_parameters

    def forward(self, t, x):
        p = self.p
        wx = StateWrapper(x)

        Fy_f, Fy_r, Fx_f, Fx_r = torch.unbind(self.tire_model(x), dim=-1)

        F_drag = p.Cd0 * torch.sign(wx.v_x) +\
            p.Cd1 * wx.v_x +\
            p.Cd2 * wx.v_x * wx.v_x

        v_x_dot = 1.0 / p.m * (Fx_r + Fx_f * torch.cos(wx.delta) -
                               Fy_f * torch.sin(wx.delta) - F_drag + p.m * wx.v_y * wx.r)

        v_y_dot = 1.0 / p.m * (Fx_f * torch.sin(wx.delta) +
                               Fy_r + Fy_f * torch.cos(wx.delta) - p.m * wx.v_x * wx.r)  #

        r_dot = 1.0 / p.I_z * \
            ((Fx_f * torch.sin(wx.delta) + Fy_f *
             torch.cos(wx.delta)) * p.lf - Fy_r * p.lr)

        omega_wheels_dot = p.R / p.I_e * (p.K_fi * wx.Iq - p.R * Fx_f - p.R * Fx_r
                                          - wx.omega_wheels * p.b1 - torch.sign(wx.omega_wheels) * p.b0)

        return torch.stack([v_x_dot, v_y_dot, r_dot, omega_wheels_dot, torch.zeros_like(wx.friction), torch.zeros_like(wx.delta), torch.zeros_like(wx.Iq)], dim=-1)


def observation(model: torch.nn.Module, x: torch.Tensor):
    x_dot = model.forward(torch.tensor(0.0, device=x.device, dtype=x.dtype), x)
    wx = StateWrapper(x)
    wx_dot = StateWrapper(x_dot)
    a_x = wx_dot.v_x - wx.r * wx.v_y
    a_y = wx_dot.v_y + wx.r * wx.v_x
    return torch.stack([a_x, a_y, wx.r, wx.omega_wheels], dim=-1)
