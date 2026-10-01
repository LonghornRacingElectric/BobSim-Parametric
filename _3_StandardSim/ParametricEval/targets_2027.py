"""2027 mechanical anti targets: source projection, maneuver screens and evidence.

All simulations run the existing 6DOF equations. No aero, road-grip or physical
hardpoint optimum is inferred. Results retain unsupported cases and source hashes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
import yaml

from _0_Utils.dyn_py.models import ModelInputs
from _0_Utils.dyn_py.parametric import ParametricVehicle, parameters_from_mapping
from _0_Utils.dyn_py.parametric_projection import project_2027, with_longitudinal_anti
from _0_Utils.dyn_py.parametric_trim import solve_level_trim
from _0_Utils.vehicle_io import load_yaml, repo_root
from _3_StandardSim.ParametricEval.coupled_roll_center_sweep import front_lltd
from _3_StandardSim.ParametricEval.parametric_eval import write_csv
from _3_StandardSim.ParametricEval.roll_center_matrix import response_metrics


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def project(output: Path) -> dict[str, Any]:
    root = repo_root()
    source = root / "_3_StandardSim/ParametricEval/configs/2027_source.yml"
    data, audit = project_2027(load_yaml(source))
    executable = copy.deepcopy(data)
    executable["tire_file"] = "tire.tir"
    (output / "vehicle.yml").write_text(yaml.safe_dump(executable, sort_keys=False), encoding="utf-8")
    tire = root / data["tire_file"]
    audit.update(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        tire_sha256=hashlib.sha256(tire.read_bytes()).hexdigest(),
    )
    save_json(output / "projection_audit.json", audit)
    return data


def vehicle_of(data):
    return ParametricVehicle(parameters_from_mapping(data, base_dir=repo_root()))


def trim_row(data, ax, ay, speed=15.0):
    vehicle = vehicle_of(data)
    trim = solve_level_trim(vehicle, speed, ax, ay)
    o = trim.output
    tire = vehicle.parameters.tire
    valid = bool(trim.success and np.all((o.normal_loads_n >= tire.fz_min_n) & (o.normal_loads_n <= tire.fz_max_n)))
    row = {
        "ax_demand": ax,
        "ay_demand": ay,
        "ax": float(trim.acceleration_world[0]),
        "ay": float(trim.acceleration_world[1]),
        "valid": valid,
        "residual": trim.residual_norm,
        "pitch_deg": float(np.rad2deg(trim.state[4])),
        "roll_deg": float(np.rad2deg(trim.state[3])),
        "jounce_front_mm": float(np.mean(o.jounce_m[:2]) * 1000),
        "jounce_rear_mm": float(np.mean(o.jounce_m[2:]) * 1000),
        "min_fz_n": float(o.normal_loads_n.min()),
        "max_fz_n": float(o.normal_loads_n.max()),
        "front_lltd_pct": 100 * front_lltd(o.normal_loads_n) if abs(ay) > 0.1 else None,
        "front_geometric_n": float(o.geometric_vertical_forces_n[:2].sum()),
        "rear_geometric_n": float(o.geometric_vertical_forces_n[2:].sum()),
        "Fz_front_n": float(o.normal_loads_n[:2].sum()),
        "Fz_rear_n": float(o.normal_loads_n[2:].sum()),
    }
    return row


def tune_arb(data, front, rear, target, total):
    candidate = copy.deepcopy(data)
    candidate["front"]["roll_center_height_m"] = front / 1000
    candidate["rear"]["roll_center_height_m"] = rear / 1000

    def residual(rate):
        candidate["front"]["arb_roll_stiffness_nm_per_rad"] = rate
        candidate["rear"]["arb_roll_stiffness_nm_per_rad"] = total - rate
        trim = solve_level_trim(vehicle_of(candidate), 15, 0, 8)
        if not trim.success:
            raise ValueError("LLTD level trim failed")
        return front_lltd(trim.output.normal_loads_n) - target

    low, high = residual(0), residual(total)
    if low * high > 0:
        raise ValueError("Target LLTD needs ARB outside nonnegative fixed-total bracket")
    rate = brentq(residual, 0, total, xtol=1e-7)
    residual(rate)
    return candidate


def maneuver(data, speed, kind="steer", *, dt=0.01, rtol=2e-6):
    vehicle = vehicle_of(data)
    model = vehicle.model(6)
    straight = vehicle.steady_state(6, speed_mps=speed)
    if not straight.success:
        raise ValueError("Straight trim failed")
    masses, radii = vehicle.parameters.mass_kg, np.asarray(vehicle.parameters.wheel_radius_m)
    duration = 4.0 if kind == "steer" else 3.0
    brake_bias = vehicle.parameters.brake_distribution_front

    def controls(time, state):
        ramp = float(np.clip((time - 1) / 0.15, 0, 1))
        release = float(1 - np.clip((time - 1.8) / 0.15, 0, 1))
        steer = np.deg2rad(2) * ramp if kind in ("steer", "trail", "exit") else 0.0
        velocity = float(np.hypot(state[6], state[7]))
        if kind == "steer":
            force = float(np.clip(masses * 3 * (speed - velocity), -1500, 1500))
        else:
            demand = {"brake": -8.0, "drive": 5.0, "trail": -5.0, "exit": 3.0}[kind]
            force = masses * demand * ramp * release
            if force > 0:
                force = min(
                    force,
                    vehicle.parameters.peak_drive_force_n,
                    vehicle.parameters.peak_drive_power_w / max(velocity, 0.25),
                )
        front_fraction = vehicle.parameters.drive_distribution_front if force >= 0 else brake_bias
        weights = np.array([front_fraction, front_fraction, 1 - front_fraction, 1 - front_fraction]) / 2
        return ModelInputs(steering_rad=float(steer), wheel_torques_nm=tuple(force * weights * radii))

    times = np.linspace(0, duration, int(round(duration / dt)) + 1)
    result = vehicle.simulate(
        6,
        initial_state=straight.state,
        controls=controls,
        time_s=times,
        rtol=rtol,
        atol=rtol * 0.01,
        max_step_s=min(dt, 0.01),
    )
    if not result.success:
        raise ValueError(result.message)
    rows = []
    for time, state in zip(result.time_s, result.state):
        inputs = controls(time, state)
        o = model.evaluate(state, inputs)
        row = {
            "time_s": float(time),
            "ay_mps2": float(o.generalized_acceleration[1] + state[11] * state[6]),
            "ax_mps2": float(o.generalized_acceleration[0] - state[11] * state[7]),
            "speed_mps": float(np.hypot(state[6], state[7])),
            "roll_deg": float(np.rad2deg(state[3])),
            "pitch_deg": float(np.rad2deg(state[4])),
            "yaw_rate_radps": float(state[11]),
        }
        for i, c in enumerate(("FL", "FR", "RL", "RR")):
            row[f"Fz_{c}"] = float(o.normal_loads_n[i])
            row[f"jounce_{c}_mm"] = float(o.jounce_m[i] * 1000)
            row[f"geometric_{c}_n"] = float(o.geometric_vertical_forces_n[i])
        rows.append(row)
    loads = np.array([[r[f"Fz_{c}"] for c in ("FL", "FR", "RL", "RR")] for r in rows])
    travel = np.array([[r[f"jounce_{c}_mm"] for c in ("FL", "FR", "RL", "RR")] for r in rows])
    tire = vehicle.parameters.tire
    metrics = {
        "kind": kind,
        "speed": speed,
        "valid": bool(np.all((loads >= tire.fz_min_n) & (loads <= tire.fz_max_n))),
        "min_fz_n": float(loads.min()),
        "max_fz_n": float(loads.max()),
        "peak_abs_travel_mm": float(np.max(np.abs(travel))),
        "peak_front_bump_mm": float(travel[:, :2].max()),
        "peak_rear_bump_mm": float(travel[:, 2:].max()),
        "peak_pitch_deg": float(max(abs(r["pitch_deg"]) for r in rows)),
        "peak_roll_deg": float(max(abs(r["roll_deg"]) for r in rows)),
        "peak_abs_ax": float(max(abs(r["ax_mps2"]) for r in rows)),
    }
    if kind == "steer":
        metrics.update(response_metrics(rows))
        metrics["max_speed_error_mps"] = max(abs(r["speed_mps"] - speed) for r in rows)
        metrics["valid"] = metrics["valid"] and metrics["max_speed_error_mps"] < 0.2
    return metrics, rows


def execute(job):
    output, data = Path(job["output"]), job["data"]
    case = job["case"]
    folder = output / case
    folder.mkdir(parents=True, exist_ok=True)
    row = {k: v for k, v in job.items() if k not in ("data", "output")}
    try:
        if "anti_dive" in job:
            data = with_longitudinal_anti(data, job["anti_dive"], job["anti_squat"])
        if "front_rc_mm" in job:
            if job.get("tune", False):
                data = tune_arb(data, job["front_rc_mm"], job["rear_rc_mm"], job["lltd"], job["arb_total"])
            else:
                data = copy.deepcopy(data)
                data["front"]["roll_center_height_m"] = job["front_rc_mm"] / 1000
                data["rear"]["roll_center_height_m"] = job["rear_rc_mm"] / 1000
        saved = copy.deepcopy(data)
        saved["tire_file"] = "../tire.tir"
        (folder / "input.yml").write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")
        row["front_arb_nm_per_rad"] = data["front"]["arb_roll_stiffness_nm_per_rad"]
        row["rear_arb_nm_per_rad"] = data["rear"]["arb_roll_stiffness_nm_per_rad"]
        if job["kind"] == "trim":
            row.update(trim_row(data, job["ax"], job["ay"], job.get("speed", 15)))
        else:
            metrics, history = maneuver(
                data, job["speed"], job["kind"], dt=job.get("dt", 0.01), rtol=job.get("rtol", 2e-6)
            )
            row.update(metrics)
            write_csv(folder / "trace.csv", history)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        row.update(valid=False, error=str(exc))
    save_json(folder / "summary.json", row)
    print(f"{case}: valid={row['valid']} {row.get('error', '')}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("pilot", "roll", "longitudinal", "validation", "refinement"), default="pilot"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=repo_root() / "_3_StandardSim/generated_results/targets_2027_level_trim"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = project(args.output)
    source_files = [
        Path(__file__),
        repo_root() / "_0_Utils/dyn_py/parametric.py",
        repo_root() / "_0_Utils/dyn_py/parametric_projection.py",
        repo_root() / "_0_Utils/dyn_py/parametric_trim.py",
        repo_root() / "_0_Utils/dyn_py/models.py",
        repo_root() / "_0_Utils/dyn_py/parameters.py",
        repo_root() / "_0_Utils/dyn_py/transient.py",
        repo_root() / "_0_Utils/dyn_py/kinematics.py",
        repo_root() / "_3_StandardSim/ParametricEval/roll_center_matrix.py",
        repo_root() / "_3_StandardSim/ParametricEval/parametric_eval.py",
        repo_root() / "_3_StandardSim/ParametricEval/coupled_roll_center_sweep.py",
        repo_root() / "_3_StandardSim/ParametricEval/configs/2027_targets.yml",
    ]
    snapshot = args.output / "source_snapshots" / args.phase
    snapshot.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source in source_files:
        key = source.relative_to(repo_root()).as_posix()
        raw = source.read_bytes()
        (snapshot / (key.replace("/", "__") + ".txt")).write_bytes(raw)
        hashes[key] = hashlib.sha256(raw).hexdigest()
    save_json(snapshot / "hashes.json", hashes)
    (args.output / "tire.tir").write_bytes((repo_root() / data["tire_file"]).read_bytes())
    base = {"data": data, "output": args.output.as_posix(), "speed": 15.0}
    jobs = []
    if args.phase == "pilot":
        jobs.append(dict(base, case="baseline_steer15", kind="steer"))
        for ad, ass in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)):
            for ax in (-8.0, 5.0):
                jobs.append(
                    dict(
                        base,
                        case=f"pilot_AD{ad}_AS{ass}_ax{ax}",
                        kind="trim",
                        anti_dive=ad,
                        anti_squat=ass,
                        ax=ax,
                        ay=0.0,
                    )
                )
    elif args.phase == "roll":
        vehicle = vehicle_of(data)
        baseline = solve_level_trim(vehicle, 15.0, 0.0, 8.0)
        if not baseline.success:
            raise ValueError("Reference LLTD trim did not converge")
        target = front_lltd(baseline.output.normal_loads_n)
        total = sum(data[a]["arb_roll_stiffness_nm_per_rad"] for a in ("front", "rear"))
        for front in range(10, 61, 10):
            for rear in range(10, 61, 10):
                jobs.append(
                    dict(
                        base,
                        case=f"roll_F{front}_R{rear}",
                        kind="steer",
                        front_rc_mm=front,
                        rear_rc_mm=rear,
                        tune=True,
                        lltd=target,
                        arb_total=total,
                    )
                )
    elif args.phase == "longitudinal":
        for ad in np.linspace(0, 1, 5):
            for ass in np.linspace(0, 1, 5):
                for kind in ("brake", "drive", "trail", "exit"):
                    jobs.append(
                        dict(
                            base,
                            case=f"long_AD{ad:.2f}_AS{ass:.2f}_{kind}",
                            kind=kind,
                            anti_dive=float(ad),
                            anti_squat=float(ass),
                        )
                    )
    elif args.phase == "validation":
        baseline = solve_level_trim(vehicle_of(data), 15, 0, 8)
        target = front_lltd(baseline.output.normal_loads_n)
        total = sum(data[a]["arb_roll_stiffness_nm_per_rad"] for a in ("front", "rear"))
        setups = [
            ("baseline", None, None, 0, 0),
            ("fast", 60, 10, 0, 0),
            ("neighbor", 50, 10, 0, 0),
            ("middle", 40, 20, 0, 0),
            ("fast_AD25_AS25", 60, 10, 0.25, 0.25),
            ("fast_AD50_AS50", 60, 10, 0.5, 0.5),
        ]
        for name, front, rear, ad, ass in setups:
            setup = dict(base, anti_dive=ad, anti_squat=ass, dt=0.0025, rtol=1e-9)
            if front is not None:
                setup.update(front_rc_mm=front, rear_rc_mm=rear, tune=True, lltd=target, arb_total=total)
            for speed in (10, 15, 20):
                jobs.append(dict(setup, case=f"check_{name}_{speed}", kind="steer", speed=speed))
            for kind in ("brake", "drive", "trail", "exit"):
                jobs.append(dict(setup, case=f"check_{name}_{kind}", kind=kind))
        # Alignment sensitivity retains the exact same RC/ARB values after tuning.
        for gain in (-20, -10):
            altered = copy.deepcopy(data)
            altered["front"]["camber_gain_deg_per_m"] = gain
            altered["rear"]["camber_gain_deg_per_m"] = 0.75 * gain
            for name, front, rear in (("baseline", None, None), ("fast", 60, 10)):
                job = dict(base, data=altered, case=f"camber_{gain}_{name}", kind="steer", dt=0.0025, rtol=1e-9)
                if front is not None:
                    frozen = tune_arb(data, front, rear, target, total)
                    for axle in ("front", "rear"):
                        for key in ("roll_center_height_m", "arb_roll_stiffness_nm_per_rad"):
                            job["data"][axle][key] = frozen[axle][key]
                jobs.append(copy.deepcopy(job))
    else:
        baseline = solve_level_trim(vehicle_of(data), 15, 0, 8)
        target = front_lltd(baseline.output.normal_loads_n)
        total = sum(data[a]["arb_roll_stiffness_nm_per_rad"] for a in ("front", "rear"))
        for name, front, rear, ad, ass in (
            ("baseline", None, None, 0, 0),
            ("fast", 60, 10, 0, 0),
            ("nominal25", 50, 20, 0.25, 0.25),
            ("nominal50", 50, 20, 0.5, 0.5),
        ):
            setup = dict(base, anti_dive=ad, anti_squat=ass, dt=0.00125, rtol=1e-10)
            if front is not None:
                setup.update(front_rc_mm=front, rear_rc_mm=rear, tune=True, lltd=target, arb_total=total)
            for speed in (10, 15, 20):
                jobs.append(dict(setup, case=f"fine_{name}_{speed}", kind="steer", speed=speed))
            for kind in ("brake", "drive", "trail", "exit"):
                jobs.append(dict(setup, case=f"fine_{name}_{kind}", kind=kind))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(execute, jobs))
    # Heterogeneous failures keep their reason rather than being dropped.
    columns = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(args.output / f"{args.phase}_summary.csv", [{k: r.get(k) for k in columns} for r in rows])
    save_json(args.output / f"{args.phase}_summary.json", rows)
    print(json.dumps({"phase": args.phase, "cases": len(rows), "valid": sum(r["valid"] for r in rows)}), flush=True)


if __name__ == "__main__":
    main()
