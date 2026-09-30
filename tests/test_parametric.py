from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from _0_Utils.dyn_py.parametric import ParametricAxle, ParametricKinematics, load_parametric_vehicle
from _0_Utils.dyn_py.parameters import G
from _0_Utils.vehicle_io import repo_root


@pytest.fixture
def vehicle():
    return load_parametric_vehicle(repo_root() / "parametric_vehicle.yml")


def test_load_requires_no_hardpoints_or_fourpost(vehicle):
    with (
        patch("_0_Utils.kin_py.kinematics.CornerKinematics.from_vehicle", side_effect=AssertionError),
        patch("_0_Utils.dyn_py.parameters._load_metrics", side_effect=AssertionError),
    ):
        loaded = load_parametric_vehicle(repo_root() / "parametric_vehicle.yml")
    assert loaded.kinematics.mode == "parametric"
    assert loaded.parameters.mass_kg == vehicle.parameters.mass_kg


def test_rc_force_path_and_virtual_work():
    axle = ParametricAxle(track_m=1.2, roll_center_height_m=0.06, force_line_height_gradient=0.2)
    kin = ParametricKinematics(axle, axle)
    zero = kin.at(np.zeros(4))
    np.testing.assert_allclose(zero.instant_links.coefficient_matrix[:, 1], [-0.1, 0.1, -0.1, 0.1])
    q = np.array([0.01, -0.02, 0.03, -0.01])
    epsilon = 1e-6
    state = kin.at(q)
    derivative = kin.at(q + epsilon).contact_patch_offsets_m - kin.at(q - epsilon).contact_patch_offsets_m
    np.testing.assert_allclose(derivative / (2 * epsilon), state.contact_patch_tangents, atol=1e-10)
    fx, fy = np.array([30, 40, 50, 60]), np.array([150, 400, 200, 500])
    fz = state.instant_links.geometric_vertical_forces(fx, fy)
    np.testing.assert_allclose(np.sum(np.column_stack((fx, fy, fz)) * state.contact_patch_tangents, axis=1), 0)


def test_motion_ratio_changes_wheel_rate_not_static_weights(vehicle):
    from _0_Utils.dyn_py.parametric import parameters_from_mapping
    from _0_Utils.vehicle_io import load_yaml

    data = load_yaml(repo_root() / "parametric_vehicle.yml")
    data["front"]["motion_ratio_spring_per_wheel"] *= 0.5
    changed = parameters_from_mapping(data, base_dir=repo_root())
    assert changed.suspension_stiffness_n_per_m[0] == pytest.approx(
        vehicle.parameters.suspension_stiffness_n_per_m[0] / 4
    )
    assert changed.suspension_damping_n_s_per_m[0] == pytest.approx(
        vehicle.parameters.suspension_damping_n_s_per_m[0] / 4
    )
    assert changed.static_wheel_loads_n == vehicle.parameters.static_wheel_loads_n


def test_camber_includes_body_roll_and_mirrors():
    axle = ParametricAxle(track_m=1.2, roll_center_height_m=0, static_camber_deg=-2, camber_gain_deg_per_m=-20)
    kin = ParametricKinematics(axle, axle)
    q = np.full(4, 0.025)
    np.testing.assert_allclose(np.rad2deg(kin.at(q).camber_rad), [-2.5, 2.5, -2.5, 2.5])
    phi = np.deg2rad(3)
    rotation = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    np.testing.assert_allclose(np.rad2deg(kin.at_pose(q, rotation, 0).camber_rad), [-5.5, -0.5, -5.5, -0.5])


def test_static_balance_and_road_camber_hook(vehicle):
    model = vehicle.model(6)
    output = model.evaluate(model.initial_state())
    assert output.normal_loads_n.sum() == pytest.approx(vehicle.parameters.mass_kg * G)
    assert np.linalg.norm(output.generalized_acceleration) < 1e-9
    state = model.initial_state()
    state[3] = 0.01
    output = model.evaluate(state)
    raw = vehicle.kinematics.at(output.jounce_m).camber_rad
    np.testing.assert_allclose(output.camber_rad, raw - 0.01, atol=1e-12)


def test_left_right_cornering_symmetry(vehicle):
    left = vehicle.steady_state(6, speed_mps=15, yaw_rate_radps=0.3)
    right = vehicle.steady_state(6, speed_mps=15, yaw_rate_radps=-0.3)
    assert left.success and right.success
    assert left.lateral_acceleration_mps2 == pytest.approx(-right.lateral_acceleration_mps2, abs=1e-6)
    np.testing.assert_allclose(left.output.normal_loads_n, right.output.normal_loads_n[[1, 0, 3, 2]], atol=1e-5)


def test_rc_changes_load_distribution(vehicle):
    from _0_Utils.dyn_py.vehicle import Vehicle

    low = vehicle.steady_state(6, speed_mps=15, yaw_rate_radps=0.4)
    kin = vehicle.kinematics
    raised = replace(kin, front=replace(kin.front, roll_center_height_m=0.10))
    high = Vehicle(replace(vehicle.parameters, kinematics=raised)).steady_state(6, speed_mps=15, yaw_rate_radps=0.4)
    assert low.success and high.success
    low_delta = low.output.normal_loads_n[1] - low.output.normal_loads_n[0]
    high_delta = high.output.normal_loads_n[1] - high.output.normal_loads_n[0]
    assert high_delta > low_delta + 10


def test_domain_errors_are_explicit(vehicle, tmp_path):
    with pytest.raises(ValueError, match="travel domain"):
        vehicle.kinematics.at([0.2, 0, 0, 0])
    with pytest.raises(ValueError, match="positive"):
        ParametricAxle(track_m=1.2, roll_center_height_m=0.03, motion_ratio_spring_per_wheel=0)
    bad = tmp_path / "bad.yml"
    bad.write_text("schema: boblib.vehicle.v1\n")
    with pytest.raises(ValueError, match="schema"):
        load_parametric_vehicle(bad)


@pytest.mark.parametrize("dof", [3, 10, 14])
def test_unvalidated_closures_rejected(vehicle, dof):
    with pytest.raises(ValueError, match="6DOF only"):
        vehicle.model(dof)


def test_study_produces_accepted_direct_outputs(tmp_path):
    import csv
    from _3_StandardSim.ParametricEval.parametric_eval import run_study

    output = tmp_path / "study"
    result = run_study(
        repo_root() / "parametric_vehicle.yml",
        output,
        grids=["front.roll_center_height_m=0.02"],
        speed=15,
        targets=[0, 4],
        step_deg=2,
        duration=3,
        dt=0.02,
    )
    assert result["all_cases_valid"]
    assert len(result["cases"]) == 1
    with (output / "steady.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert abs(float(rows[1]["ay_mps2"]) - 4) < 0.002
    assert float(rows[1]["Fz_FR"]) > float(rows[1]["Fz_FL"])
    # Archived case input must reload using only the included tire and YAML.
    archived = load_parametric_vehicle(output / "case_000/vehicle.yml")
    assert archived.parameters.mass_kg == 280
    assert (output / "response.png").is_file()


def test_camber_and_motion_ratio_grid_has_distinct_cases():
    from _3_StandardSim.ParametricEval.parametric_eval import variants
    from _0_Utils.vehicle_io import load_yaml

    data = load_yaml(repo_root() / "parametric_vehicle.yml")
    cases = variants(data, ["front.camber_gain_deg_per_m=-10,-30", "front.motion_ratio_spring_per_wheel=0.7,0.9"])
    assert len(cases) == 5
    assert data["front"]["camber_gain_deg_per_m"] == -20
    with pytest.raises(ValueError):
        variants(data, ["front.roll_center_heigth_m=0"])


def test_rise_time_interpolates_between_samples():
    from _3_StandardSim.ParametricEval.parametric_eval import threshold_time
    times = np.array([1.0, 1.1, 1.2, 1.3])
    response = np.array([0.0, 0.5, 1.0, 1.0])
    assert threshold_time(times, response, 0.1) == pytest.approx(1.02)
    assert threshold_time(times, response, 0.9) == pytest.approx(1.18)
    assert threshold_time(times, response, 2) is None
