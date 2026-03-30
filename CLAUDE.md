# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for the paper "Learning Dynamics Models for Velocity Estimation in Autonomous Racing" ([arXiv:2408.15610](https://www.arxiv.org/abs/2408.15610)). Trains vehicle dynamics models and Unscented Kalman Filters (UKF) for velocity estimation on F1Tenth racing cars using OptiTrack motion capture data.

## Development Environment

Runs inside Docker. Build and run from the repo root:
```bash
cd code/docker && ./build.sh   # build image
cd <repo_root> && ./code/docker/run.sh   # run container (mounts repo at /learning_through_kalman_filter)
```

Dependencies: PyTorch, torchdiffeq, numpy, matplotlib, pandas, wandb, attrs, icecream.

`PYTHONPATH` is set to `/learning_through_kalman_filter` inside the container. All Python imports are relative to `code/`.

## Key Commands

Training (3-stage pipeline, runs all stages automatically):
```bash
python code/train.py [args]
```

Testing (requires a trained model wandb path set in `model_dict`):
```bash
python code/test.py
```

All training/model hyperparameters are CLI args — see `code/utils/argparser.py` for the full list. Args are prefixed by stage: `--common_*`, `--base_*`, `--res_*`, `--ukf_*`.

## Architecture

### Training Pipeline (code/train.py)

Three sequential stages, each building on the previous:

1. **Base model** (`base_*` args) — Learns a single-track vehicle dynamics model with a neural tire model. Trains via single-step ODE prediction. Saved to `code/trained_models/base_*/`.
2. **Residual model** (`res_*` args, optional via `--res_enable 1`) — Adds a residual neural network on top of the base model to capture unmodeled dynamics. Saved to `code/trained_models/res_*/`.
3. **UKF fine-tuning** (`ukf_*` args) — Loads the trained base/residual model and jointly trains it with a learnable noise model through UKF filtering on sequential data. Logs to wandb.

The pipeline auto-skips stages if a trained model already exists in `code/trained_models/`.

### Vehicle Model

- `robot_models/single_track_pacejka.py` — Single-track bicycle model. State: `[v_x, v_y, r, omega_wheels, friction, delta, Iq]` (see `utils/state_wrapper.py`).
- `tire_models/` — Tire force models (Pacejka, neural variants). Selected via `--base_tire_model`.
- `robot_models/single_track_parameters.py` — Physical vehicle parameters (mass, inertia, geometry).
- `robot_models/residual_model.py` — Optional residual NN added to base dynamics.

### UKF (Unscented Kalman Filter)

- `filters/ukf.py` — Generic batched UKF implementation (predict/update with sigma points).
- `filters/ukf_model_steper_training.py` / `ukf_model_steper_inference.py` — Wrappers that step the UKF through sequences, handling state transition via ODE integration and observation model.
- `noise_models/` — Learnable noise covariance models (diagonal, cross-covariance, heterogeneous). Selected via `--ukf_noise_model`.

### Data

- `dataset/data/` — Raw ROS2 bag files organized by tire type (A-D, different friction coefficients).
- `code/opti_test/` — Preprocessed CSV files used for training/testing.
- `datasets/optitrack.py` — Single-step dataset (for base/residual training).
- `datasets/optitrack_sequential.py` — Sequential dataset (for UKF training).

### Experiment Tracking

All training stages log to [Weights & Biases](https://wandb.ai). Debugger detection auto-disables wandb when debugging.
