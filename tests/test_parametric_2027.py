from copy import deepcopy

import numpy as np
import pytest

from _0_Utils.dyn_py.parameters import G
from _0_Utils.dyn_py.parametric import ParametricVehicle, parameters_from_mapping
from _0_Utils.dyn_py.parametric_projection import project_2027, with_longitudinal_anti
from _0_Utils.dyn_py.parametric_trim import solve_level_trim
from _0_Utils.vehicle_io import load_yaml, repo_root


@pytest.fixture
def data():
    source = load_yaml(repo_root() / "_3_StandardSim/ParametricEval/configs/2027_source.yml")
    return project_2027(source)[0]


def test_workbook_projection_preserves_mass_cg_and_effective_rates(data):
    p = parameters_from_mapping(data, base_dir=repo_root())
    assert p.mass_kg == pytest.approx(271.24823726)
    assert sum(p.static_wheel_loads_n[:2]) / (p.mass_kg * G) == pytest.approx(0.45)
    assert -p.corner_positions[0, 2] == pytest.approx(0.2921)
    np.testing.assert_allclose(p.suspension_stiffness_n_per_m, [11038.06890696312] * 2 + [16273.65400118526] * 2)
    assert p.antiroll_stiffness_nm_per_rad == pytest.approx((4421.950180785536, 0))
    assert p.peak_drive_force_n == pytest.approx(220 * 3.31 / 0.2045)
    assert p.brake_distribution_front == pytest.approx(0.84)


def test_reference_anti_force_paths_and_reversal(data):
    original = deepcopy(data)
    d = with_longitudinal_anti(data, 0.5, 0.75)
    p = parameters_from_mapping(d, base_dir=repo_root())
    links = p.kinematics.at(np.zeros(4)).instant_links
    m, h, wheelbase = p.mass_kg, data["vehicle"]["cg_height_m"], p.wheelbase_m
    brake_fx = -m * 8 * np.array([0.84, 0.84, 0.16, 0.16]) / 2
    geo = links.geometric_vertical_forces(brake_fx, np.zeros(4))
    assert sum(geo[:2]) / (m * 8 * h / wheelbase) == pytest.approx(0.5)
    assert -sum(geo[2:]) / (m * 8 * h / wheelbase) == pytest.approx(0.75 * 0.16)
    drive_fx = m * 5 * np.array([0, 0, 0.5, 0.5])
    assert sum(links.geometric_vertical_forces(drive_fx, np.zeros(4))[2:]) / (m * 5 * h / wheelbase) == pytest.approx(
        0.75
    )
    np.testing.assert_allclose(links.geometric_vertical_forces(-brake_fx, np.zeros(4)), -geo)
    assert data == original


@pytest.mark.parametrize("ax,ay", [(-8, 0), (5, 0), (0, 8), (-4, 5)])
def test_level_trim_has_no_fictitious_damper_velocity(data, ax, ay):
    vehicle = ParametricVehicle(parameters_from_mapping(with_longitudinal_anti(data, 0.5, 0.5), base_dir=repo_root()))
    trim = solve_level_trim(vehicle, 15, ax, ay)
    assert trim.success
    np.testing.assert_allclose(trim.acceleration_world, [ax, ay, 0], atol=1e-7)
    np.testing.assert_allclose(trim.output.jounce_speed_mps, 0, atol=1e-12)
    np.testing.assert_allclose(trim.output.derivative[2:5], 0, atol=1e-12)


def test_small_acceleration_anti_redistributes_support_without_removing_transfer(data):
    # m*a*h/L assumes FIXED geometry/CG. Suppress pitch-induced CG/patch
    # migration using a stiff limiting fixture; the regular soft car need not
    # have exactly the same load transfer after its CG moves relative to wheels.
    for axle in ("front", "rear"):
        data[axle]["spring_rate_n_per_m"] *= 10000
    observed = []
    for anti in (0, 1):
        vehicle = ParametricVehicle(
            parameters_from_mapping(with_longitudinal_anti(data, anti, 0), base_dir=repo_root())
        )
        trim = solve_level_trim(vehicle, 15, -0.02, 0)
        assert trim.success
        front_transfer = sum(trim.output.normal_loads_n[:2]) - sum(vehicle.parameters.static_wheel_loads_n[:2])
        expected = vehicle.parameters.mass_kg * 0.02 * data["vehicle"]["cg_height_m"] / vehicle.parameters.wheelbase_m
        assert front_transfer == pytest.approx(expected, rel=0.01)
        observed.append(abs(np.mean(trim.output.jounce_m[:2])))
    assert observed[1] < observed[0] * 0.02
