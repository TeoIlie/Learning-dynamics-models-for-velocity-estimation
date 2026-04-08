from copy import deepcopy
from distutils import config
import os
import pickle
import torch
import numpy as np
import pandas as pd
import wandb
import time

from utils.argparser import get_parser

from utils.trained_model_finder import find_model
from datasets.dataset_getter import get_loaders, get_sequence_loaders
from utils.solver_settings import get_solver_settings, get_ode_solve_method
from utils.base_model_training import train_base_model
from utils.res_model_training import train_residual_model

from utils.base_model_training import create_base_model
from utils.res_model_training import create_residual_model
from utils.ukf_model_saver import model_saver

# ukf
from datasets.optitrack_sequential import OptitrackDatasetSequential
from noise_models.diagonal_noise_model import ConstDiagonalNoiseModel
from noise_models.crosscovariance_noise_model import CrossCovNoiseModel
from noise_models.heterogenus_noise_model import HeteroCovNoiseModel
from utils.dataframe_timeseries_save import save_as_dataframe
import json
import sys

from filters.ukf_model_steper_training import UKFModelStepperTrain
from filters.ukf_model_steper_inference import UKFModelStepperInference
from utils.create_noise_model import get_noise_model


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


if debugger_is_active():
    os.environ["WANDB_MODE"] = "disabled"
    torch.autograd.set_detect_anomaly(True)

    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def train_ukf(args, state_dim=5, control_dim=2):
    datetime = pd.Timestamp.now().strftime("%Y_%m_%d_%H_%M_%S")

    mode = "ukf"
    torch.manual_seed(args.ukf_seed)
    device = torch.device(args.common_device)
    solver_settings = get_solver_settings(args)

    if args.res_enable == 0:
        model, fit_params = create_base_model(args)
        model_dict = find_model(args, "base")
        model.load_state_dict(model_dict)
    else:
        model, fit_params = create_residual_model(args)
        res_model_dict = find_model(args, "res")
        model.load_state_dict(res_model_dict)

    noise_model = get_noise_model(args)

    train_loader, test_loader, train_dset, test_dset = get_sequence_loaders(
        args, mode)

    model = torch.jit.script(model)

    ukf_single_stepper = UKFModelStepperInference(model, noise_model,
                                                  train_dset.dt, solver_settings,
                                                  args.ukf_q_entropy_lb, device=device)

    ukf_stepper = UKFModelStepperTrain(ukf_model_steper=ukf_single_stepper)

    assert np.isclose(train_dset.Ts, test_dset.Ts).item()

    fit_params += list(noise_model.parameters())

    optimizer = getattr(torch.optim, args.ukf_optimizer)(
        fit_params, lr=args.ukf_lr)

    loss_fn = getattr(torch.nn, args.common_loss_fn)(reduction="none")
    state_weights = torch.tensor(args.ukf_states_weights, device=device)

    best_loss = np.inf
    best_model = None

    run = wandb.init(project=f"{mode}_all_in_one", config=vars(args))
    run.watch(model, log_freq=10)
    run.watch(noise_model, log_freq=10)

    def calc_loss(x, args, test=False):
        X0 = x[:, 0, :state_dim]
        u = x[:, :, state_dim: state_dim + control_dim]
        imu = x[:, :, -3:]
        wheel_speed = x[:, :, 3].unsqueeze(-1)
        y = torch.cat((imu, wheel_speed), dim=-1)

        if test:
            X_ukf, P, q_entropy, r_entropy = ukf_stepper(X0, u, y)
        else:  # train with X ground truth
            X_ukf, P, q_entropy, r_entropy = ukf_stepper(
                X0, u, y, X_GT=x[:, :, :state_dim])

        loss = loss_fn(X_ukf[:, args.ukf_start_loss:, :],
                       x[:, args.ukf_start_loss:, :state_dim]).mul(state_weights).sum()

        return loss, q_entropy, r_entropy, X_ukf, x[:, :, :state_dim], P

    for epoch in range(args.ukf_epochs):
        train_loss = 0.0
        test_loss = 0.0
        q_entropy_mean = 0.0
        q_entropy_std = 0.0
        r_entropy_mean = 0.0
        r_entropy_std = 0.0
        epoch_time = time.time()

        for x in train_loader:
            optimizer.zero_grad()
            loss, q_entropy, r_entropy, X_ukf, x, P = calc_loss(x, args)
            q_entropy_mean = q_entropy.mean().item()
            q_entropy_std = q_entropy.std().item()
            r_entropy_mean = r_entropy.mean().item()
            r_entropy_std = r_entropy.std().item()
            loss.backward()
            # clip gradient with default value 1e3 to prevent exploding gradients
            torch.nn.utils.clip_grad_value_(fit_params, args.ukf_grad_clip)
            optimizer.step()
            train_loss += loss.item()

        with torch.no_grad():
            df = None
            for x in test_loader:
                loss, q_entropy, r_entropy, X_ukf, x, P = calc_loss(
                    x, args, test=True)
                test_loss += loss.item()
                if df is None and epoch % args.ukf_send_df_interval == 0:
                    df = save_as_dataframe(X_ukf, x, P)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model = (deepcopy(model.state_dict()),
                          deepcopy(noise_model.state_dict()))

        train_loss = train_loss / \
            len(train_dset) / (args.ukf_sequence_length - args.ukf_start_loss)
        test_loss = test_loss / \
            len(test_dset) / (args.ukf_test_sequence_length - args.ukf_start_loss)

        epoch_time_in_seconds = time.time() - epoch_time

        print(f"Epoch {epoch} train loss: {train_loss} test loss: {test_loss}")
        print(f"time: {epoch_time_in_seconds}")
        print(f"noise {list(noise_model.named_parameters())}")

        wandb_log_dict = {"train_loss": train_loss,
                          "test_loss": test_loss,
                          "best_loss": best_loss,
                          "q_entropy_mean": q_entropy_mean,
                          "q_entropy_std": q_entropy_std,
                          "r_entropy_mean": r_entropy_mean,
                          "r_entropy_std": r_entropy_std,
                          "epoch_time": epoch_time_in_seconds}

        if df is not None:
            wandb_log_dict["df"] = wandb.Table(dataframe=df)
            artefact = model_saver(args, best_model, datetime, epoch)
            run.log_artifact(artefact, type="model")

        run.log(wandb_log_dict)
    run.finish()


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.common_precision == 64:
        print("Using 64 bit precision")
        torch.set_default_dtype(torch.float64)
    else:
        print("Using 32 bit precision")
        torch.set_default_dtype(torch.float32)

    # Base model
    try:
        find_model(args, "base")
    except FileNotFoundError:
        print("Training base model")
        train_base_model(args)

    # Residual model pretraining
    if args.res_enable != 0:
        try:
            find_model(args, "res")
        except FileNotFoundError:
            print("Training residual model")
            train_residual_model(args)

    # UKF fine tuning
    print("Training UKF")
    train_ukf(args)


if __name__ == "__main__":
    main()
