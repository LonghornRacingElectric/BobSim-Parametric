"""Explicit, auditable projection of the 2027 workbook block into parametric v1.

No hardpoint kinematics or cached FourPost results are consulted. Source mass
locations are used only to combine inertia; effective rates come from the
workbook block. Aero is deliberately withheld pending a consistent map/datum.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from _0_Utils.dyn_py.parameters import _combine_mass_properties, _mass_components, project_powertrain_limits


def project_2027(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    block = data["frame_torsion_study"]
    components = _mass_components(data)
    mass, cg, inertia = _combine_mass_properties(components)
    sprung_components = [c for c in components if ".unsprung" not in c[0]]
    sprung, sprung_cg, sprung_inertia = _combine_mass_properties(sprung_components)
    target_mass = float(block["total_mass_kg"])
    if not np.isclose(mass, target_mass) or not np.isclose(sprung, block["sprung_mass_kg"]):
        raise ValueError("Workbook and component masses disagree; resolve input basis first.")
    wheelbase = float(block["wheelbase_m"])
    front_fraction = float(block["total_front_weight_fraction"])
    height = float(block["total_cg_height_m"])
    # Preserve the intrinsic sprung tensor, translate its centroid to satisfy
    # the workbook total CG, then recombine with unchanged unsprung bodies.
    target_cg = np.array([-wheelbase * (1 - front_fraction), 0.0, height])
    shift = (target_cg - np.asarray(cg)) * mass / sprung
    moved = [
        (n, m, tuple(np.asarray(p) + shift) if ".unsprung" not in n else p, tensor) for n, m, p, tensor in components
    ]
    _, verified_cg, inertia = _combine_mass_properties(moved)
    if not np.allclose(verified_cg, target_cg, atol=1e-10):
        raise AssertionError("CG projection failed.")
    # Source tensors contain rounding asymmetry below 1e-6 kg m2.
    inertia = (inertia + inertia.T) / 2
    sprung_inertia = (sprung_inertia + sprung_inertia.T) / 2
    radii = [float(data[a]["wheel"]["radius_m"]) for a in ("front", "rear")]
    power = project_powertrain_limits(data, driven_wheel_radius_m=radii[1])
    output: dict[str, Any] = {
        "schema": "bobsim.parametric.v1",
        "description": "2027 workbook projection; mechanical-only screening; provisional inherited inertia",
        "tire_file": "_0_Utils/tire_templates/16x7p5_10_12psi.tir",
        "vehicle": {
            "mass_kg": mass,
            "wheelbase_m": wheelbase,
            "cg_from_front_axle_m": wheelbase * (1 - front_fraction),
            "cg_height_m": height,
            "inertia_kg_m2": inertia.tolist(),
            "sprung_inertia_kg_m2": sprung_inertia.tolist(),
        },
        "aero": {"cl_area_m2": 0.0, "cd_area_m2": 0.0, "balance_front": 0.5},
        "powertrain": {
            "peak_power_w": power.peak_power_w,
            "peak_force_n": power.peak_drive_force_n,
            "maximum_speed_mps": power.maximum_vehicle_speed_mps,
            "drive_front_fraction": 0.0,
            "brake_front_fraction": float(data["brake"]["front_bias"]),
        },
    }
    rates = []
    for i, axle in enumerate(("front", "rear")):
        source = data[axle]
        track = float(block["track_m"][i])
        k = 2 * float(block["axle_spring_roll_stiffness_nm_per_rad"][i]) / track**2
        c = 2 * float(block["axle_roll_damping_nms_per_rad"][i]) / track**2
        spring_table = source["actuation"]["shock"]["spring_table"]["table"]
        spring = (spring_table[-1][1] - spring_table[0][1]) / (spring_table[-1][0] - spring_table[0][0])
        mr = float(np.sqrt(k / spring))
        output[axle] = {
            "track_m": track,
            "roll_center_height_m": float(block["roll_center_height_m"][i]),
            "static_camber_deg": float(source["wheel"]["camber_deg"]),
            "camber_gain_deg_per_m": 0.0,
            "toe_in_deg": float(source["wheel"]["toe_deg"]),
            "longitudinal_jacking_coefficient": 0.0,
            "motion_ratio_spring_per_wheel": mr,
            "spring_rate_n_per_m": float(spring),
            "damper_n_s_per_m": c / mr**2,
            "arb_roll_stiffness_nm_per_rad": float(block["axle_arb_roll_stiffness_nm_per_rad"][i]),
            "wheel_radius_m": radii[i],
            "wheel_inertia_kg_m2": float(source["tire"]["wheel_inertia_kg_m2"]),
            "unsprung_mass_per_corner_kg": float(block["unsprung_axle_mass_kg"][i]) / 2,
            "tire_vertical_stiffness_n_per_m": float(source["tire"]["vertical_stiffness_n_per_m"]),
            "tire_vertical_damping_n_s_per_m": float(source["tire"]["vertical_damping_n_s_per_m"]),
            "travel_min_m": -0.03,
            "travel_max_m": 0.03,
        }
        rates.append({"axle": axle, "wheel_rate_n_per_m": k, "wheel_damping_ns_per_m": c, "spring_per_wheel_mr": mr})
    aero = data["aero"]
    df = np.asarray(aero["downforce_table_n"])
    moment = np.asarray(aero["my_table_nm"])
    front_x = float(data["front"]["suspension"]["wheel_center_m"][0])
    balance = 1 + (moment + (float(aero["aero_ref_m"][0]) - front_x) * df) / (df * wheelbase)
    audit = {
        "basis": "frame_torsion_study workbook effective values, not provisional hardpoint kinematics",
        "rates": rates,
        "total_mass_kg": mass,
        "sprung_mass_kg": sprung,
        "total_cg_m": verified_cg,
        "sprung_cg_m": (np.asarray(sprung_cg) + shift).tolist(),
        "source_aero_front_balance_range": [float(balance.min()), float(balance.max())],
        "aero_status": "withheld: source moment convention produces negative front aerodynamic load throughout map",
        "camber_toe_migration_status": "not identified from workbook; zero baseline and separate sensitivity required",
        "longitudinal_anti_baseline_status": (
            "not identified from workbook; zero reference is a study control, not measured car geometry"
        ),
        "motion_ratio_status": (
            "effective MR inferred from workbook roll stiffness and source spring; no extra tire compliance"
        ),
        "chassis_torsion_status": (
            "source finite torsion recorded, current 6DOF is rigid; not a chassis-stiffness target"
        ),
        "source_chassis_torsion_nm_per_rad": data["body"]["torsional_stiff_n_m_per_rad"],
        "travel_status": "provisional +/-30 mm screening envelope, not verified hardware or ground clearance",
        "powertrain_status": (
            "source peak torque/power/speed and brake bias projected; no differential or rotor dynamics in 6DOF"
        ),
    }
    return output, audit


def with_longitudinal_anti(data: dict[str, Any], anti_dive: float, anti_squat: float) -> dict[str, Any]:
    """Total-vehicle, nominal flat-road quasi-static reference fractions.

    q is positive bump and x is forward. Fz_geo = jx * Fx, hence jx is
    negative front (braking) and positive rear (traction). Translating-upright
    abstraction: one fixed axle path applies in both force directions. Rear
    braking anti-lift consequently equals AS * rear brake fraction for RWD.
    This is not a hardware brake/halfshaft validation or sprung anti convention.
    """
    if not np.all(np.isfinite([anti_dive, anti_squat])):
        raise ValueError("Anti fractions must be finite.")
    result = copy.deepcopy(data)
    ratio = data["vehicle"]["cg_height_m"] / data["vehicle"]["wheelbase_m"]
    bias = data["powertrain"]["brake_front_fraction"]
    rear_drive = 1 - data["powertrain"]["drive_front_fraction"]
    if bias <= 0 or rear_drive <= 0:
        raise ValueError("Front braking and rear drive are required for these reference definitions.")
    result["front"]["longitudinal_jacking_coefficient"] = -float(anti_dive) * ratio / bias
    result["rear"]["longitudinal_jacking_coefficient"] = float(anti_squat) * ratio / rear_drive
    return result
