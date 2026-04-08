import torch
import collections


Param = collections.namedtuple(
    'param_config', ['learned', 'init_value', 'positive'], defaults=[False, 0.0, False])

DEFAULT_VEHICLE_PARAMETERS = {
    # only params marked learned=True are learned, others remain as-is
    'm': Param(learned=False, init_value=5.1),
    'g': Param(learned=False, init_value=9.81),
    'I_z': Param(learned=True, init_value=0.46),
    'L': Param(learned=False, init_value=0.33),
    'lr': Param(learned=True, init_value=0.115),
    'Cd0': Param(learned=True, init_value=0.1),
    'Cd2': Param(learned=True, init_value=0.1),
    'Cd1': Param(learned=True, init_value=0.01),
    'Cdy1': Param(learned=True, init_value=0.01),
    'mu_static': Param(learned=False, init_value=0.8),
    'I_e': Param(learned=True, init_value=0.2),
    'K_fi': Param(learned=True, init_value=0.90064745),
    'b1': Param(learned=True, init_value=0.304115174),
    'b0': Param(learned=True, init_value=0.50421894),
    'R': Param(learned=False, init_value=0.05),
    'eps': Param(learned=False, init_value=1e-6)
}


class SingleTrackParameters(torch.nn.Module):

    def __init__(self):
        super(SingleTrackParameters, self).__init__()

        self.parameters_config = DEFAULT_VEHICLE_PARAMETERS

        for key, value in self.parameters_config.items():
            if value.learned:
                self.register_parameter(f'_{key}', torch.nn.Parameter(
                    torch.tensor(value.init_value)))
            else:
                setattr(self, f'_{key}', torch.tensor(value.init_value))

    def forward(self):
        return

    def _make_positive(self, p, init_val):
        return init_val * torch.exp(p) / torch.exp(init_val)

    @property
    def m(self):
        return self._m

    @property
    def g(self):
        return self._g

    @property
    def I_z(self):
        return self._make_positive(self._I_z, torch.tensor(self.parameters_config['I_z'][1]))

    @property
    def lr(self):
        return self._lr

    @property
    def L(self):
        return self._L

    @property
    def lf(self):
        return self.L - self.lr

    @property
    def Fn_f(self):
        return self.m * self.g * self.lr / self.L

    @property
    def Fn_r(self):
        return self.m * self.g * self.lf / self.L

    @property
    def Cd0(self):
        return self._make_positive(self._Cd0, torch.tensor(self.parameters_config['Cd0'][1]))

    @property
    def Cd2(self):
        return self._make_positive(self._Cd2, torch.tensor(self.parameters_config['Cd2'][1]))

    @property
    def Cd1(self):
        return self._make_positive(self._Cd1, torch.tensor(self.parameters_config['Cd1'][1]))

    @property
    def Cdy1(self):
        return self._make_positive(self._Cdy1, torch.tensor(self.parameters_config['Cdy1'][1]))

    @property
    def mu_static(self):
        return self._mu_static

    @property
    def I_e(self):
        return self._make_positive(self._I_e, torch.tensor(self.parameters_config['I_e'][1]))

    @property
    def K_fi(self):
        return self._make_positive(self._K_fi, torch.tensor(self.parameters_config['K_fi'][1]))

    @property
    def b1(self):
        return self._make_positive(self._b1, torch.tensor(self.parameters_config['b1'][1]))

    @property
    def b0(self):
        return self._make_positive(self._b0, torch.tensor(self.parameters_config['b0'][1]))

    @property
    def R(self):
        return self._R

    @property
    def eps(self):
        return self._eps


if __name__ == "__main__":
    model = SingleTrackParameters()
    print(model.lr)
    print(model.Fn_f)
    model.lr = model.lr - 0.01
    print(model.Fn_f)
    print(model.m)
    # print all parameters
    for name, param in model.named_parameters():
        print(name, param.data)
