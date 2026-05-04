from copy import deepcopy
from distutils import config
import os
import pickle
import torch
import numpy as np
import pandas as pd
import wandb
import time

from pathlib import Path
from datasets.optitrack import OptitrackDataset
from robot_models.residual_model import ResidualNeuralModel

from robot_models.single_track_pacejka import SingleTrackPacejkaModel, observation
from tire_models.pacejka import PacejkaTireModel

from utils.trained_model_finder import find_model
from datasets.dataset_getter import get_loaders, get_sequence_loaders
from utils.solver_settings import get_solver_settings, get_ode_solve_method
from utils.base_model_training import create_base_model
import json


def create_residual_model(args):
    device = torch.device(args.common_device)

    # create base model
    base_model, _ = create_base_model(args)

    base_model_dict = find_model(args, "base")

    base_model.load_state_dict(base_model_dict, strict=False)

    res_model = ResidualNeuralModel(base_model,
                                    res_mlp_size=args.res_mlp_size,
                                    res_layer_count=args.res_layer_count,
                                    res_activation=getattr(torch.nn, args.res_activation))

    res_model = torch.jit.script(res_model)

    fit_parameters = list(res_model.nn.parameters())

    return res_model, fit_parameters


def train_residual_model(args):
    mode = "res"
    torch.manual_seed(args.res_seed)
    device = torch.device(args.common_device)

    train_loader, test_loader, train_dset, test_dset = get_loaders(args, mode)

    model, fit_parameters = create_residual_model(args)
    model = model.to(device)
    fit_parameters = list(model.nn.parameters())

    ode_solve = get_ode_solve_method(args, model, fit_parameters)

    optimizer = getattr(torch.optim, args.res_optimizer)(
        fit_parameters, lr=args.res_lr)

    loss_fn = getattr(torch.nn, args.common_loss_fn)(reduction="none")

    state_weights = torch.tensor(args.res_states_weights, device=device)

    best_model = None
    best_loss = np.inf

    wandb.init(project=f"{mode}_all_in_one", config=vars(args))
    wandb.watch(model, log_freq=10)

    model_at_epoch = {}

    def select_states(x):
        return x[:, :7]

    def calc_loss(x, x_next):
        x_state = select_states(x)
        x_next_state = select_states(x_next)
        x_pred = ode_solve(x_state, train_dset.dt)
        return loss_fn(x_pred, x_next_state).mul(state_weights).sum()

    for i in range(args.res_epochs):
        train_loss = 0.0
        test_loss = 0.0

        for x, x_next in train_loader:

            loss = calc_loss(x, x_next)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        with torch.no_grad():
            for x, x_next in test_loader:
                loss = calc_loss(x, x_next)
                test_loss += loss.item()

        if test_loss < best_loss:
            best_loss = test_loss
            best_model = {"model": deepcopy(model)}

        if i % args.res_save_interval == 0:
            model_at_epoch[i] = {"model": deepcopy(model)}

        train_loss = train_loss / len(train_dset)
        test_loss = test_loss / len(test_dset)
        print(f"Epoch {i} train loss: {train_loss} test loss: {test_loss}")

        wandb.log({"train_loss": train_loss,
                   "test_loss": test_loss})

    datetime = pd.Timestamp.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_path = (Path(__file__).parent.parent /
                 "trained_models" / f"res_{datetime}")
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=4)

    torch.save(best_model["model"].state_dict(),
               os.path.join(save_path, "model.pt"))

    wandb.finish()
