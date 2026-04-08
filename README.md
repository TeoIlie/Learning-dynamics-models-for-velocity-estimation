
# Learning Dynamics Models for Velocity Estimation in Autonomous Racing

*Published on ArXiv*: [arXiv:2408.15610](https://www.arxiv.org/abs/2408.15610)

## Dataset: Aggressive Car Maneuver Data from F1Tenth Car and OptiTrack System

![Front Page](dataset/images/frontpage.jpg)

## Overview

This dataset contains high-resolution data recorded during aggressive maneuvers of an F1Tenth racing car. The vehicle operated with  sideslip angles reaching up to 40° and rear axle slip angles exceeding 20°. The distribution of rear axle slip angles can be observed in the histogram visualization provided below.

![Data Histogram](dataset/images/data_hist_4lines_crop.png)

The data was captured using both the F1Tenth equiped with onboard sensors and an OptiTrack motion capture system. The dataset includes information such as motor current, RPM, pose in the OptiTrack reference frame, control signals, and IMU measurements. The recordings are stored in ROS2 bag files and are distributed across several topics, as detailed below.

<video width="640" height="360" controls>
  <source src="dataset/video/f1tenth_iros.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

This video illustrates the aggressive maneuvers performed by the F1Tenth car during the data collection process. It highlights the extreme driving conditions, including high side slip angles and rapid changes of direction, which were essential for developing estimation algorithms for agile driving.

## Data Description

### Topics and Messages

The dataset is structured into several ROS2 topics, each containing specific sensor and control data from the vehicle and the OptiTrack system.

1. **/vesc/core**  
   Contains information about the F1Tenth car’s motor state:
   - `msg.state.current_motor`: Current on the engine (measured in Amps)
   - `msg.state.speed`: RPM of the engine (note: gear ratio needs to be added)

2. **/optitrack/rigid_body_0**  
   Provides the car’s position and orientation in the OptiTrack reference frame:
   - `msg.pose.position.x`: Position in the x-axis
   - `msg.pose.position.y`: Position in the y-axis
   - `msg.pose.orientation`: Orientation quaternion in the OptiTrack frame

3. **/vesc/servo_position_command**  
   Contains the control signal for the servo controlling the car’s steering:
   - `msg.data`: Servo position control signal

4. **/commands/motor/current**  
   Contains the control signal for the motor’s current:
   - `msg.data`: Engine current control signal

5. **/imu**  
   Provides the vehicle’s inertial measurements:
   - `msg.linear_acceleration`: Linear acceleration in the x, y, and z axes (measured in m/s²)
   - `msg.angular_velocity`: Angular velocity in the x, y, and z axes (measured in rad/s)
   - `msg.orientation`: Orientation in the IMU reference frame (quaternion)

### Data Split Based on Tire Friction Conditions

The dataset is split into four subsets corresponding to different tire conditions: **Tire A**, **Tire B**, **Tire C**, and **Tire D**. Each tire configuration was tested under varying road friction conditions, with each having a distinct coefficient of road-tire friction (μ). Specifically, **Tire A** has a friction coefficient of approximately μA ≈ 0.65, **Tire B** has a friction coefficient of approximately 0.58, **Tire C** around 0.43, and **Tire D** around 0.45. These variations in friction coefficients allow for the study of the vehicle’s dynamic behavior under different traction scenarios, making this dataset useful for analyzing the impact of tire friction on maneuverability and velocity estimation.

## Usage

The dataset is ideal for researchers and engineers looking to explore vehicle dynamics, develop autonomous racing algorithms, or study aggressive maneuvering behavior in small-scale racing platforms. The combination of motor, IMU, and OptiTrack data provides a rich source for velocity estimation, dynamic modeling, and control system design.

---

## System Identification Analysis and Adaptation Plan

The following documents an analysis of how this repository achieves system identification from data, and a plan for adapting the approach to a different vehicle model (Single Track Drift with PAC2002 tires) on a similar F1Tenth platform using Vicon motion capture.

### How the Repo Achieves System ID

This codebase implements a **reverse-estimation / virtual-sensing** approach to system identification. Instead of directly measuring tire forces (which would require expensive load cells), the system collects behavioral data about how the car moves and fits model parameters to explain that behavior.

#### Training Pipeline

The pipeline has three sequential stages (run via `code/train.py`):

1. **Base model training** (`code/utils/base_model_training.py`) — The core system ID step. Learns vehicle parameters and a tire force model by minimizing single-step prediction error. For each data pair `(x_t, x_{t+1})`, the model integrates one timestep forward via ODE and compares to the observed next state.

2. **Residual model** (optional) — Adds a residual neural network on top of the base model to capture unmodeled dynamics.

3. **UKF fine-tuning** — Jointly trains the dynamics model and a learnable noise model through Unscented Kalman Filter filtering on sequential data.

Only **stage 1** is relevant for pure parameter identification.

#### What Gets Learned

Two categories of parameters are jointly optimized:

- **Vehicle parameters** (`code/robot_models/single_track_parameters.py`): `I_z`, `lr`, `Cd0`, `Cd1`, `Cd2`, `I_e`, `K_fi`, `b0`, `b1` are registered as `torch.nn.Parameter`. Parameters that can be directly measured (`m`, `g`, `L`, `R`) are fixed.
- **Tire model**: Either a parametric Pacejka model (`code/tire_models/pacejka.py`, 9 parameters: `B_f/r`, `C_f/r`, `D_f/r`, `long_B`, `long_C`, `long_mu_tire`) or a neural tire model (`code/tire_models/neural_tire_model.py`, a 3-layer MLP mapping state + slip quantities to forces).

#### Loss Function

Weighted MSE on the first four states only:

```
loss = MSE(pred, x_next) * [0.2225, 0.5064, 0.1566, 0.1145, 0, 0, 0]
```

Only `v_x`, `v_y`, `r`, `omega_wheels` contribute. `friction`, `delta`, `Iq` are treated as known inputs (zero weight). An optional tire force regularization term `||F||^2` penalizes large forces.

#### ODE Integration

Uses `torchdiffeq` with **RK4 by default** (`--common_solver_method rk4`). Euler and Dormand-Prince adaptive (`dopri5`) are also available. The adjoint method (`odeint_adjoint`) is optionally available for memory-efficient gradient computation.

#### Prediction Horizon

**Single-step** for the base model training. Each sample is one `(x_t, x_{t+1})` pair at the data sample period (~10ms). This is a short-horizon approach — the model only needs to be accurate for one timestep at a time.

#### Handling Exploding Gradients and Parameter Constraints

- **Gradient clipping**: `clip_grad_value_` in the UKF stage (default 1e3).
- **Positivity via reparameterization**: Physical parameters that must be positive use `init_val * exp(p) / exp(init_val)`, effectively optimizing in log-space.
- **Physics-informed tire model structure**: Neural tire outputs use `softplus * tanh(100 * slip_angle)` to enforce correct sign conventions.
- **Fixed vs. learned split**: Easily-measured quantities (mass, wheelbase, wheel radius) are fixed, reducing the identifiability problem.

### Available Model Combinations

The repo supports these combinations (selected via `--base_tire_model`):

| Tire Model | Vehicle Model | Description |
|---|---|---|
| `pacejka` | `SingleTrackPacejkaModel` | Pure physics-based, 9 Pacejka params |
| `neural` | `SingleTrackPacejkaModel` | Neural MLP for tire forces, friction-scaled |
| `neural_sr` | `SingleTrackPacejkaModel` | Neural with slip ratio/angle inputs |
| `neural_const_friction` | `SingleTrackPacejkaModel` | Neural with constant friction |

All share the same single-track bicycle dynamics; only the tire force computation differs.

### Training Dataset Format

The CSV training data contains these columns:

| Column | Role | Description |
|---|---|---|
| `v_x` | State (GT) | Body-frame longitudinal velocity (from OptiTrack) |
| `v_y` | State (GT) | Body-frame lateral velocity (from OptiTrack) |
| `r` | State (GT) | Yaw rate (from OptiTrack) |
| `omega_wheels` | State | Wheel angular velocity (rad/s) |
| `friction` | State | Tire-surface friction coefficient (manually set constant) |
| `delta` | Input | Steering angle (treated as known) |
| `Iq` | Input | Motor quadrature current (from VESC) |
| `ax_imu` | Observation | IMU longitudinal acceleration (UKF stage only) |
| `ay_imu` | Observation | IMU lateral acceleration (UKF stage only) |
| `r_imu` | Observation | IMU yaw rate (UKF stage only) |
| `run_id` | Bookkeeping | Segment identifier |

### Adaptation Plan: Single Track Drift Model with PAC2002 Tires

#### Target Model

The model to be ported is `gymkhana/envs/dynamic_models/single_track_drift/single_track_drift.py` — a Single Track Drift (STD) model from the CommonRoad vehicle models library, adapted for 1/10 scale. It uses a PAC2002 tire model with combined slip (`gymkhana/envs/dynamic_models/tire_model.py`) and blends with a kinematic model at low speeds. The target parameter set is `f1tenth_std_drift_bias_params` from `gymkhana/envs/gymkhana_env.py`.

#### Key Structural Differences

| Aspect | This Repo | Target STD Model |
|---|---|---|
| State vector | `[v_x, v_y, r, omega_wheels, friction, delta, Iq]` | `[X, Y, delta, V, psi, psi_dot, beta, omega_f, omega_r]` |
| Velocities | Body-frame `(v_x, v_y)` | Speed + slip angle `(V, beta)` |
| Wheel speeds | Single `omega_wheels` | Separate front and rear `(omega_f, omega_r)` |
| Controls | `(delta, Iq)` as exogenous states | `(steering_velocity, acceleration)` as true inputs |
| Tire model | Simplified Pacejka (9 params) or neural | PAC2002 with combined slip (~30 tire params) |
| Drivetrain | Motor current drives single wheel speed | Torque split between front/rear axles |
| Low-speed | Not handled | Kinematic blending below 0.2 m/s |

#### What’s Easy

- **Training infrastructure is reusable**: The ODE integration loop, MSE loss, optimizer, wandb logging, and model saving are model-agnostic. Only `forward(self, t, x) -> dx/dt` needs to change.
- **Parameter module pattern transfers directly**: Create an `STDParameters(nn.Module)` mirroring `SingleTrackParameters`, registering chosen parameters as `nn.Parameter` with the same `_make_positive` reparameterization.
- **PAC2002 formulas are pure math**: Converting `tire_model.py` from numpy to torch is mechanical (`np.sin` -> `torch.sin`, etc.).
- **Data collection is mostly in place**: Vicon provides `v_x`, `v_y`, `r`, `psi`, `psi_dot`, which cover the ground-truth states.

#### Moderate Challenges

- **State representation conversion**: The loss should penalize only dynamic states (`V`, `psi_dot`, `beta`, `omega_f`, `omega_r`), not kinematic integrations (`X`, `Y`, `psi`). `StateWrapper` and dataset column definitions need rewriting.
- **Steering angle observability**: The current data collection setup has a servo without positional feedback. Options: (a) calibrate the servo command-to-angle mapping, (b) add a magnetic encoder (AS5600) to the steering linkage, or (c) purchase a servo with feedback. Calibration is sufficient for a first pass.
- **Numpy to differentiable PyTorch**: Hard `if/else` branches and `max(0, x)` in the STD model must be replaced with smooth differentiable approximations (`torch.relu`, sigmoid gates, `softplus`). The `@njit` decorators must be removed.
- **Low-speed kinematic blending**: The smooth `tanh` blending creates a non-smooth optimization landscape. Filter out low-speed data from training or accept that the blending region may cause gradient issues.

#### Hard Challenges

- **Parameter identifiability**: The full PAC2002 model has ~30 tire parameters plus ~10 vehicle parameters. Many are practically unobservable from typical driving data. Combined-slip parameters (`tire_r_*`) only matter during simultaneous large slip angle and slip ratio. Recommendation: fix most parameters and learn only the most sensitive ones (friction and stiffness terms).
- **Control input mismatch**: The repo uses motor current `Iq` directly from the VESC. The STD model uses acceleration as input, which gets converted to torque. Without recording actual motor current, the torque mapping itself becomes an identification target. **Recording `state.current_motor` from the VESC `/sensors/core` topic is strongly recommended.**
- **Two-wheel drivetrain**: With `T_se = 0.0` (RWD), front wheel dynamics are passive. Consider fixing front wheels to their kinematic relationship and only identifying rear drivetrain parameters.

#### Recommended Approach

1. **Start simple**: Use the repo’s existing `--base_tire_model pacejka` with your data to validate the pipeline works end-to-end.
2. **Port the STD model** with most parameters fixed. Learn only 6-10 key parameters: `lf`, `lr`, `I_z`, `tire_p_dy1`, `tire_p_ky1`, `tire_p_dx1`, `tire_p_kx1`.
3. **Record motor current** from the VESC to avoid the acceleration-to-torque identification problem.
4. **Gradually unfreeze more parameters** as more diverse driving data (especially drifting) becomes available.
5. **Validate with long-horizon rollout**: After training, feed a ground-truth control sequence through the identified model and compare the integrated trajectory against Vicon data. The repo doesn’t include this but it is straightforward to implement.
