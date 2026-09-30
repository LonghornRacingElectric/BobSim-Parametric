"""Compare parameter-only vehicles at matched Ay and identical step steer.

Run with ``make parametric-eval`` in Docker. All CSV loads and accelerations
are direct reduced-model evaluations; input and tire hashes are preserved.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from _0_Utils.dyn_py.models import ModelInputs
from _0_Utils.dyn_py.parametric import ParametricVehicle, parameters_from_mapping
from _0_Utils.plotting.plot_engine import PlotEngine
from _0_Utils.vehicle_io import load_yaml, repo_root


def variants(data: dict[str, Any], grids: list[str]) -> list[tuple[str, dict[str, Any]]]:
    if grids:
        parsed = []
        for item in grids:
            key, text = item.split("=", 1)
            axle, field = key.split(".")
            if axle not in ("front", "rear") or field not in data[axle]:
                raise ValueError("Grid fields must be explicit front/rear fields in the input YAML.")
            values = [float(v) for v in text.split(",")]
            if not values or not np.all(np.isfinite(values)):
                raise ValueError("Grid requires finite values.")
            parsed.append((axle, field, values))
        if len({(a, f) for a, f, _ in parsed}) != len(parsed):
            raise ValueError("Each grid field may be specified once.")
        if np.prod([len(values) for _, _, values in parsed]) > 100:
            raise ValueError("Limit one study to at most 100 grid cases.")
        changes = [
            {f"{axle}.{field}": value for (axle, field, _), value in zip(parsed, values)}
            for values in itertools.product(*(p[2] for p in parsed))
        ]
    else:
        changes = [
            {f"{axle}.roll_center_height_m": data[axle]["roll_center_height_m"] + delta}
            for axle in ("front", "rear")
            for delta in (-0.02, 0.02)
        ]
    output = [("baseline", copy.deepcopy(data))]
    seen = {json.dumps(data, sort_keys=True)}
    for change in changes:
        candidate = copy.deepcopy(data)
        for key, value in change.items():
            axle, field = key.split(".")
            candidate[axle][field] = value
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            label = ", ".join(f"{k}={v:g}" for k, v in change.items())
            output.append((label, candidate))
            seen.add(key)
    return output


def matched_ay(vehicle: ParametricVehicle, speed: float, target: float):
    yaw = target / speed
    guess = None
    for _ in range(6):
        trim = vehicle.steady_state(6, speed_mps=speed, yaw_rate_radps=yaw, initial_unknowns=guess)
        if not trim.success or abs(trim.lateral_acceleration_mps2 - target) < 0.002:
            return trim
        yaw = target / float(trim.state[6])
        guess = trim.unknowns
    return trim


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def threshold_time(times: np.ndarray, response: np.ndarray, threshold: float) -> float | None:
    """First rising crossing, interpolated between saved output samples."""
    crossings = np.flatnonzero((times >= 1.0) & (response >= threshold))
    if not crossings.size:
        return None
    i = int(crossings[0])
    if i == 0 or response[i] == response[i - 1]:
        return float(times[i])
    fraction = (threshold - response[i - 1]) / (response[i] - response[i - 1])
    return float(times[i - 1] + fraction * (times[i] - times[i - 1]))


def run_study(
    source: Path,
    output: Path,
    *,
    grids: list[str],
    speed: float,
    targets: list[float],
    step_deg: float,
    duration: float,
    dt: float = 0.02,
    rtol: float = 1e-7,
) -> dict[str, Any]:
    if speed <= 0 or duration <= 2 or dt <= 0 or not np.all(np.isfinite([speed, duration, dt, step_deg, *targets])):
        raise ValueError("Require finite inputs, positive speed/sample period, and duration > 2 seconds.")
    source = source.resolve()
    data = load_yaml(source)
    cases = variants(data, grids)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source.yml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    tire_path = (source.parent / data["tire_file"]).resolve()
    manifest: dict[str, Any] = {
        "schema": "bobsim.parametric.study.v1",
        "model_dof": 6,
        "vehicle_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tire_sha256": hashlib.sha256(tire_path.read_bytes()).hexdigest(),
        "code_sha256": {
            relative: hashlib.sha256((repo_root() / relative).read_bytes()).hexdigest()
            for relative in [
                *(f"_0_Utils/dyn_py/{name}.py" for name in
                  ("models", "parametric", "parameters", "qss", "transient", "vehicle")),
                "_3_StandardSim/ParametricEval/parametric_eval.py",
            ]
        },
        "source_vehicle": source.as_posix(),
        "tire_file": tire_path.as_posix(),
        "speed_mps": speed,
        "target_ays_mps2": targets,
        "roadwheel_step_deg": step_deg,
        "step_start_s": 1.0,
        "step_rise_s": 0.15,
        "duration_s": duration,
        "dt_s": dt,
        "rtol": rtol,
        "limitations": [
            "Illustrative parameters; not correlated to an LHRe car or full BobLib.",
            "6DOF body with algebraic uprights; no wheel hop or tire relaxation dynamics.",
            "Nominal RC labels; migration follows prescribed corner paths, not a hardpoint linkage.",
            "Simplified MF-derived tire projection; not full MF52 evaluation.",
            "No steering jacking, caster/KPI, Ackermann, compliance or bump stops.",
            "Linear wheel rates and constant motion ratios; rigid sprung body and flat road.",
            "Travel-domain exceedance rejects a case; tire fit domain is checked at saved samples.",
        ],
        "cases": [],
    }
    steady_rows: list[dict[str, Any]] = []
    transient_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    series: dict[str, dict[str, list[float]]] = {key: {} for key in ("time", "ay", "roll", "df_front", "df_rear")}
    for index, (label, candidate) in enumerate(cases):
        case_id = f"case_{index:03d}"
        print(f"{case_id}: {label}", flush=True)
        case_dir = output / case_id
        case_dir.mkdir(exist_ok=True)
        # Preserve executable candidate inputs alongside a self-contained tire copy.
        saved = copy.deepcopy(candidate)
        saved["tire_file"] = "../tire.tir"
        (case_dir / "vehicle.yml").write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")
        case_record: dict[str, Any] = {"id": case_id, "label": label, "errors": []}
        manifest["cases"].append(case_record)
        try:
            vehicle = ParametricVehicle(parameters_from_mapping(candidate, base_dir=source.parent))
            model = vehicle.model(6)
            tire = vehicle.parameters.tire
            for target in targets:
                trim = matched_ay(vehicle, speed, target)
                loads = trim.output.normal_loads_n
                valid = bool(
                    trim.success
                    and abs(trim.lateral_acceleration_mps2 - target) < 0.002
                    and np.all((loads >= tire.fz_min_n) & (loads <= tire.fz_max_n))
                )
                row = {
                    "case": case_id,
                    "label": label,
                    "target_ay_mps2": target,
                    "ay_mps2": trim.lateral_acceleration_mps2,
                    "valid": valid,
                    "solver_success": trim.success,
                    "residual_norm": trim.residual_norm,
                    "steer_deg": float(np.rad2deg(trim.inputs.steering_rad)),
                    "roll_deg": float(np.rad2deg(trim.state[3])),
                    "front_out_minus_in_n": float(np.sign(target) * (loads[1] - loads[0])),
                    "rear_out_minus_in_n": float(np.sign(target) * (loads[3] - loads[2])),
                }
                row.update({f"Fz_{corner}": float(load) for corner, load in zip(("FL", "FR", "RL", "RR"), loads)})
                steady_rows.append(row)
                if not valid:
                    case_record["errors"].append(f"Rejected steady target {target:g} m/s2: {trim.message}")
            straight = vehicle.steady_state(6, speed_mps=speed)
            if not straight.success:
                raise RuntimeError(f"Straight trim failed: {straight.message}")

            def controls(time, state):
                amplitude = np.clip((time - 1.0) / 0.15, 0.0, 1.0)
                steering = straight.inputs.steering_rad + np.deg2rad(step_deg) * amplitude
                velocity = np.hypot(state[6], state[7])
                force = np.clip(vehicle.parameters.mass_kg * 3 * (speed - velocity), -1500, 1500)
                base = np.asarray(straight.inputs.wheel_torques_nm)
                front_fraction = (
                    vehicle.parameters.drive_distribution_front
                    if force >= 0
                    else vehicle.parameters.brake_distribution_front
                )
                weights = np.array([front_fraction, front_fraction, 1 - front_fraction, 1 - front_fraction]) / 2
                torque = base + force * weights * np.asarray(vehicle.parameters.wheel_radius_m)
                return ModelInputs(steering_rad=float(steering), wheel_torques_nm=tuple(torque))

            times = np.linspace(0, duration, int(np.ceil(duration / dt)) + 1)
            result = vehicle.simulate(
                6,
                initial_state=straight.state,
                controls=controls,
                time_s=times,
                rtol=rtol,
                atol=rtol * 0.01,
                max_step_s=min(dt, 0.02),
            )
            if not result.success:
                raise RuntimeError(result.message)
            rows = []
            for time, state in zip(result.time_s, result.state):
                inputs = controls(time, state)
                evaluated = model.evaluate(state, inputs)
                loads = evaluated.normal_loads_n
                row = {
                    "case": case_id,
                    "time_s": float(time),
                    "ay_mps2": float(evaluated.generalized_acceleration[1] + state[11] * state[6]),
                    "speed_mps": float(np.hypot(state[6], state[7])),
                    "steer_deg": float(np.rad2deg(inputs.steering_rad)),
                    "roll_deg": float(np.rad2deg(state[3])),
                    "yaw_rate_radps": float(state[11]),
                    "front_right_minus_left_n": float(loads[1] - loads[0]),
                    "rear_right_minus_left_n": float(loads[3] - loads[2]),
                    "tire_domain_valid": bool(np.all((loads >= tire.fz_min_n) & (loads <= tire.fz_max_n))),
                }
                for c, corner in enumerate(("FL", "FR", "RL", "RR")):
                    row[f"Fz_{corner}"] = float(loads[c])
                    row[f"camber_{corner}_deg"] = float(np.rad2deg(evaluated.camber_rad[c]))
                    row[f"jounce_{corner}_m"] = float(evaluated.jounce_m[c])
                rows.append(row)
            transient_rows.extend(rows)
            write_csv(case_dir / "transient.csv", rows)
            ay = np.array([r["ay_mps2"] for r in rows])
            tail = ay[result.time_s >= duration - 0.5]
            final = float(np.mean(tail))
            speed_error = max(abs(r["speed_mps"] - speed) for r in rows)
            settled = bool(np.ptp(tail) < max(0.02, 0.02 * abs(final)))
            domain_valid = all(r["tire_domain_valid"] for r in rows)
            valid = bool(domain_valid and speed_error < 0.2 and settled)
            if not valid:
                case_record["errors"].append("Transient rejected: tire domain, speed tracking or final settling.")
            response = (ay - ay[0]) * np.sign(final - ay[0])
            span = abs(final - ay[0])
            t10 = threshold_time(result.time_s, response, 0.1 * span)
            t90 = threshold_time(result.time_s, response, 0.9 * span)
            summary = {
                "case": case_id,
                "label": label,
                "valid": valid,
                "settled": settled,
                "tire_domain_valid": domain_valid,
                "final_ay_mps2": final,
                "max_speed_error_mps": speed_error,
                "rise_10_90_s": t90 - t10 if valid and span > 1e-6 and t10 is not None and t90 is not None else None,
                "overshoot_percent": float(100 * (response.max() / span - 1)) if valid and span > 1e-6 else None,
                "min_tire_load_n": min(r[f"Fz_{c}"] for r in rows for c in ("FL", "FR", "RL", "RR")),
            }
            summaries.append(summary)
            for key, column in (
                ("time", "time_s"),
                ("ay", "ay_mps2"),
                ("roll", "roll_deg"),
                ("df_front", "front_right_minus_left_n"),
                ("df_rear", "rear_right_minus_left_n"),
            ):
                series[key][label + (" [rejected]" if not valid else "")] = [r[column] for r in rows]
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            case_record["errors"].append(str(exc))
        case_record["valid"] = not bool(case_record["errors"])
    (output / "tire.tir").write_bytes(tire_path.read_bytes())
    write_csv(output / "steady.csv", steady_rows)
    write_csv(output / "transient_metrics.csv", summaries)
    manifest["all_cases_valid"] = all(c["valid"] for c in manifest["cases"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if series["ay"]:
        plots = {
            "response": {
                "layout": "quad",
                "title": "Parameterized suspension research (6DOF)",
                "subplots": [
                    {"title": title, "x": {"key": "time", "label": "Time (s)"}, "y": {"key": key, "label": unit}}
                    for key, title, unit in (
                        ("ay", "Lateral acceleration", "Ay (m/s²)"),
                        ("roll", "Body roll", "Roll (deg)"),
                        ("df_front", "Front load difference", "Right - left (N)"),
                        ("df_rear", "Rear load difference", "Right - left (N)"),
                    )
                ],
            }
        }
        PlotEngine({"plots": plots}).save_pngs({"series": series}, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicle", type=Path, default=repo_root() / "parametric_vehicle.yml")
    parser.add_argument("--output", type=Path, default=repo_root() / "_3_StandardSim/generated_results/parametric")
    parser.add_argument("--grid", action="append", default=[], help="front.roll_center_height_m=-0.01,0.02,0.05")
    parser.add_argument("--speed", type=float, default=15)
    parser.add_argument("--ays", type=float, nargs="+", default=[0, 4, 8])
    parser.add_argument("--step-deg", type=float, default=2)
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=1e-7)
    args = parser.parse_args()
    result = run_study(
        args.vehicle,
        args.output,
        grids=args.grid,
        speed=args.speed,
        targets=args.ays,
        step_deg=args.step_deg,
        duration=args.duration,
        dt=args.dt,
        rtol=args.rtol,
    )
    print(json.dumps({"output": args.output.as_posix(), "all_cases_valid": result["all_cases_valid"]}), flush=True)
    if not result["all_cases_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
