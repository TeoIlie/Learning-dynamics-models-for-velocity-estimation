from copy import deepcopy
from distutils import config
import os
import pickle
import torch
import numpy as np
import pandas as pd
import wandb
import time

import torchdiffeq as ode
import matplotlib.pyplot as plt
from pathlib import Path
from datasets.optitrack import OptitrackDataset
from preprocessors.no import NoPreprocessing
from preprocessors.orientation import OrientationPreprocessing
from preprocessors.imu_offset import ImuOffestAndRotation
from robot_models.residual_model import ResidualNeuralModel

from robot_models.single_track_parameters import SingleTrackParameters
from robot_models.single_track_pacejka import SingleTrackPacejkaModel, observation
from tire_models.neural_tire_model import NeuralPacejkaTireModel
from tire_models.neural_tire_model_sr_sa import NeuralPacejkaTireModelSRSA
from tire_models.neural_tire_model_fix_friction import NeuralPacejkaTireModelFixFriction
from tire_models.pacejka import PacejkaTireModel
from utils.state_wrapper import STATE_DEF_LIST
from utils.argparser import get_parser

from utils.trained_model_finder import find_model
from datasets.dataset_getter import get_loaders, get_sequence_loaders
from utils.solver_settings import get_solver_settings, get_ode_solve_method
import json


def create_base_model(args):
    device = torch.device(args.common_device)

    tvp = SingleTrackParameters()

    if args.base_tire_model == "pacejka":
        model_tire = PacejkaTireModel(vehicle_parameters=None)
    elif args.base_tire_model == "neural":
        model_tire = NeuralPacejkaTireModel(vehicle_parameters=tvp)
    elif args.base_tire_model == "neural_sr":
        model_tire = NeuralPacejkaTireModelSRSA(vehicle_parameters=tvp)
    elif args.base_tire_model == "neural_const_friction":
        model_tire = NeuralPacejkaTireModelFixFriction(vehicle_parameters=tvp)
    else:
        raise ValueError("Unknown tire model")

    model = SingleTrackPacejkaModel(
        vehicle_parameters=tvp, tire_model=model_tire)

    # model = torch.jit.script(model)

    return model, list(model.parameters())


def train_base_model(args):
    mode = "base"
    torch.manual_seed(args.base_seed)
    device = torch.device(args.common_device)

    model, fit_parameters = create_base_model(args)
    model = model.to(device)
    fit_parameters = list(model.parameters())

    ode_solve = get_ode_solve_method(args, model, fit_parameters)
    train_loader, test_loader, train_dset, test_dset = get_loaders(args, mode)

    optimizer = getattr(torch.optim, args.base_optimizer)(fit_parameters,
                                                          lr=args.base_lr,
                                                          weight_decay=args.base_weight_decay)

    loss_fn = getattr(torch.nn, args.common_loss_fn)(reduction="none")
    state_weights = torch.tensor(args.base_states_weights, device=device)

    best_model = None
    best_loss = np.inf

    wandb.init(project=f"{mode}_all_in_one", config=vars(args))
    wandb.watch(model, log_freq=10)

    def select_states(x):
        return x[:, :7]

    for i in range(args.base_epochs):
        train_loss = 0.0
        test_loss = 0.0

        for x, x_next in train_loader:

            x = select_states(x)
            x_next = select_states(x_next)

            pred = ode_solve(x, train_dset.dt)
            F2 = (model.tire_model(x)**2).sum(dim=-1).mean()

            loss = loss_fn(pred, x_next).mul(state_weights).sum()
            loss += args.base_tire_force_reg * F2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        with torch.no_grad():
            for x, x_next in test_loader:

                x = select_states(x)
                x_next = select_states(x_next)

                pred = ode_solve(x, test_dset.dt)
                loss = loss_fn(pred, x_next).mul(state_weights).sum()
                test_loss += loss.item()

        if test_loss < best_loss:
            best_loss = test_loss
            best_model = {"model": deepcopy(model)}

        train_loss = train_loss / len(train_dset)
        test_loss = test_loss / len(test_dset)
        wandb.log({"train_loss": train_loss, "test_loss": test_loss})
        print(f"{mode} Epoch {i} train loss: {train_loss} test loss: {test_loss}")
        # print tvp parameters
        for name, param in model.p.named_parameters():
            print(name, param)

    datetime = pd.Timestamp.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_path = (Path(__file__).parent.parent /
                 "trained_models" / f"base_{datetime}")
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=4)

    # save model as torchscript jit
    model = best_model["model"]
    # save state dict
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

    wandb.finish()
