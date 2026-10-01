"""Level-road trim for parametric 6DOF screening.

The legacy QSS helper sets body w=0 even with pitch/roll. That creates world
vertical velocity and steady damper loads. This adapter constrains world z,
roll and pitch rates to zero and solves the unchanged model equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from _0_Utils.dyn_py.models import ModelInputs, ModelOutput, _body_to_world_rotation
from _0_Utils.dyn_py.parametric import ParametricVehicle


@dataclass(frozen=True)
class LevelTrim:
    success: bool
    residual_norm: float
    state: np.ndarray
    inputs: ModelInputs
    output: ModelOutput
    acceleration_world: np.ndarray


def solve_level_trim(vehicle: ParametricVehicle, speed: float, ax: float = 0, ay: float = 0) -> LevelTrim:
    model = vehicle.model(6)
    yaw_rate = ay / speed
    radius = np.asarray(vehicle.parameters.wheel_radius_m)

    def state_inputs(values):
        beta, steer, force, z, roll, pitch = values
        state = model.initial_state(speed)
        state[2:5] = [z, roll, pitch]
        rotation = _body_to_world_rotation(state[3:6])
        state[6:9] = rotation.T @ [speed * np.cos(beta), speed * np.sin(beta), 0]
        state[9:12] = rotation.T @ [0, 0, yaw_rate]
        f = vehicle.parameters.drive_distribution_front if force >= 0 else vehicle.parameters.brake_distribution_front
        torque = force * radius * np.array([f, f, 1 - f, 1 - f]) / 2
        return state, ModelInputs(steering_rad=float(steer), wheel_torques_nm=tuple(torque)), rotation

    def residual(values):
        state, control, rotation = state_inputs(values)
        o = model.evaluate(state, control)
        acceleration = rotation @ (o.generalized_acceleration[:3] + np.cross(state[9:12], state[6:9]))
        return np.r_[(acceleration - [ax, ay, 0]) / 9.80665, o.generalized_acceleration[3:] / 10]

    guess = [0, np.arctan(vehicle.parameters.wheelbase_m * yaw_rate / speed), vehicle.parameters.mass_kg * ax, 0, 0, 0]
    result = least_squares(
        residual,
        guess,
        bounds=([-0.3, -0.5, -10000, -0.1, -0.15, -0.15], [0.3, 0.5, 10000, 0.1, 0.15, 0.15]),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=200,
        x_scale="jac",
    )
    state, control, rotation = state_inputs(result.x)
    output = model.evaluate(state, control)
    acceleration = rotation @ (output.generalized_acceleration[:3] + np.cross(state[9:12], state[6:9]))
    norm = float(np.linalg.norm(residual(result.x)))
    return LevelTrim(bool(result.success and norm < 1e-7), norm, state, control, output, acceleration)
