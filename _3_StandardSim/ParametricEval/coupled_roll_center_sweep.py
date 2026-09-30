"""Raise front/rear RCs with a steady-state LLTD constraint; vary only RCs."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
import yaml

from _0_Utils.dyn_py.parametric import ParametricVehicle, parameters_from_mapping
from _0_Utils.plotting.plot_engine import PlotEngine
from _0_Utils.vehicle_io import load_yaml, repo_root
from _3_StandardSim.ParametricEval.parametric_eval import matched_ay, run_study, write_csv


def front_lltd(loads: Any) -> float:
    """Front share of axle right-minus-left load differences; undefined at zero transfer."""
    fz = np.asarray(loads, dtype=float)
    front, rear = fz[1] - fz[0], fz[3] - fz[2]
    if abs(front + rear) < 1e-5:
        raise ValueError("LLTD is undefined at zero lateral load transfer.")
    return float(front / (front + rear))


def evaluate_pair(data: dict[str, Any], base_dir: Path, front: float, rear: float, speed: float, ay: float):
    candidate = copy.deepcopy(data)
    candidate["front"]["roll_center_height_m"] = front
    candidate["rear"]["roll_center_height_m"] = rear
    vehicle = ParametricVehicle(parameters_from_mapping(candidate, base_dir=base_dir))
    trim = matched_ay(vehicle, speed, ay)
    loads, tire = trim.output.normal_loads_n, vehicle.parameters.tire
    if (
        not trim.success
        or abs(trim.lateral_acceleration_mps2 - ay) >= 0.002
        or not np.all((loads >= tire.fz_min_n) & (loads <= tire.fz_max_n))
    ):
        raise ValueError(f"Invalid RC trim F={front:g}, R={rear:g}: {trim.message}")
    return candidate, trim


def solve_rear_height(
    data: dict[str, Any], base_dir: Path, front: float, target_lltd: float, speed: float, ay: float
) -> float:
    rear_base = float(data["rear"]["roll_center_height_m"])

    def residual(rear):
        _, trim = evaluate_pair(data, base_dir, front, rear, speed, ay)
        return front_lltd(trim.output.normal_loads_n) - target_lltd

    if abs(residual(rear_base)) < 1e-10:
        return rear_base
    # A bounded root search raises the rear as well; never compensate via springs/ARBs.
    return float(brentq(residual, rear_base, rear_base + 0.20, xtol=1e-10))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(
    source: Path,
    output: Path,
    offsets_mm: list[float],
    *,
    speed: float = 15,
    reference_ay: float = 8,
    dt: float = 0.005,
    rtol: float = 1e-8,
) -> dict[str, Any]:
    if not offsets_mm or not np.all(np.isfinite(offsets_mm)) or min(offsets_mm) < 0:
        raise ValueError("Offsets must be finite and nonnegative.")
    if speed <= 0 or reference_ay <= 0:
        raise ValueError("Positive speed and reference Ay are required.")
    offsets_mm = sorted(set([0.0, *offsets_mm]))
    source = source.resolve()
    data = load_yaml(source)
    front_base = float(data["front"]["roll_center_height_m"])
    rear_base = float(data["rear"]["roll_center_height_m"])
    _, reference = evaluate_pair(data, source.parent, front_base, rear_base, speed, reference_ay)
    target_lltd = front_lltd(reference.output.normal_loads_n)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source.yml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (output / "tire.tir").write_bytes((source.parent / data["tire_file"]).read_bytes())
    targets = sorted(set([2.0, 4.0, 6.0, reference_ay, 10.0]))
    summaries, equal_offset_rows, lltd_rows = [], [], []
    series: dict[str, dict[str, Any]] = {
        k: {} for k in ("time", "ay", "roll", "lltd", "fz_fl", "fz_fr", "fz_rl", "fz_rr")
    }
    manifest: dict[str, Any] = {
        "study": "coupled_rc_constant_steady_lltd",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sweep_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "offsets_mm": offsets_mm,
        "speed_mps": speed,
        "reference_ay_mps2": reference_ay,
        "target_front_lltd_fraction": target_lltd,
        "dt_s": dt,
        "rtol": rtol,
        "varied_fields": ["front.roll_center_height_m", "rear.roll_center_height_m"],
        "lltd_definition": "(Fz_FR-Fz_FL)/[(Fz_FR-Fz_FL)+(Fz_RR-Fz_RL)]",
        "constraint": "Match baseline at reference Ay; other Ay and transient LLTD measured, not constrained.",
        "vehicle_status": "Illustrative 280 kg parametric vehicle, uncorrelated to LHRe car.",
        "cases": [],
    }
    reference_lltd_by_ay = {}
    for ay in targets:
        _, trim = evaluate_pair(data, source.parent, front_base, rear_base, speed, ay)
        reference_lltd_by_ay[ay] = front_lltd(trim.output.normal_loads_n)
    for index, offset in enumerate(offsets_mm):
        front = front_base + offset / 1000
        rear = solve_rear_height(data, source.parent, front, target_lltd, speed, reference_ay)
        candidate, trim = evaluate_pair(data, source.parent, front, rear, speed, reference_ay)
        _, equal = evaluate_pair(data, source.parent, front, rear_base + offset / 1000, speed, reference_ay)
        equal_offset_rows.append(
            {
                "front_rc_mm": front * 1000,
                "rear_rc_mm": rear_base * 1000 + offset,
                "front_lltd_pct": 100 * front_lltd(equal.output.normal_loads_n),
                "change_from_baseline_pp": 100 * (front_lltd(equal.output.normal_loads_n) - target_lltd),
            }
        )
        # Structural audit: both heights are the only changed engineering inputs.
        audited = copy.deepcopy(candidate)
        audited["front"]["roll_center_height_m"] = front_base
        audited["rear"]["roll_center_height_m"] = rear_base
        if audited != data:
            raise AssertionError("Non-RC inputs changed.")
        label = f"F {front * 1000:.0f} / R {rear * 1000:.1f} mm"
        case_dir = output / f"pair_{index:03d}"
        case_dir.mkdir(exist_ok=True)
        candidate["tire_file"] = "../tire.tir"
        candidate_path = case_dir / "input.yml"
        candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
        print(f"Pair {index}: {label}; LLTD {target_lltd * 100:.6f}%", flush=True)
        result = run_study(
            candidate_path,
            case_dir / "results",
            grids=[f"front.roll_center_height_m={front}"],
            speed=speed,
            targets=targets,
            step_deg=2,
            duration=5,
            dt=dt,
            rtol=rtol,
        )
        if not result["all_cases_valid"]:
            raise RuntimeError(f"Rejected {label}; see {case_dir}/results/manifest.json")
        history = read_csv(case_dir / "results/case_000/transient.csv")
        metrics = read_csv(case_dir / "results/transient_metrics.csv")[0]
        steady = read_csv(case_dir / "results/steady.csv")
        for row in steady:
            loads = [float(row[f"Fz_{c}"]) for c in ("FL", "FR", "RL", "RR")]
            ay = float(row["target_ay_mps2"])
            lltd_rows.append(
                {
                    "pair": index,
                    "front_rc_mm": front * 1000,
                    "rear_rc_mm": rear * 1000,
                    "ay_mps2": float(row["ay_mps2"]),
                    "target_ay_mps2": ay,
                    "front_lltd_pct": 100 * front_lltd(loads),
                    "change_from_baseline_pp": 100 * (front_lltd(loads) - reference_lltd_by_ay[ay]),
                }
            )
        loads = trim.output.normal_loads_n
        total_diff = loads[1] - loads[0] + loads[3] - loads[2]
        geo = trim.output.geometric_vertical_forces_n
        summary = {
            "pair": index,
            "front_rc_mm": front * 1000,
            "rear_rc_mm": rear * 1000,
            "front_lltd_pct": 100 * front_lltd(loads),
            "reference_ay_mps2": trim.lateral_acceleration_mps2,
            "steady_roll_deg": float(np.rad2deg(trim.state[3])),
            "geometric_transfer_share_pct": 100 * float(geo[1] - geo[0] + geo[3] - geo[2]) / total_diff,
            **{
                key: float(metrics[key])
                for key in (
                    "final_ay_mps2",
                    "rise_10_90_s",
                    "overshoot_percent",
                    "min_tire_load_n",
                    "max_speed_error_mps",
                )
            },
            **{f"steady_Fz_{c}_n": float(load) for c, load in zip(("FL", "FR", "RL", "RR"), loads)},
        }
        summaries.append(summary)
        manifest["cases"].append(
            {"pair": index, "front_rc_m": front, "rear_rc_m": rear, "relative_path": f"pair_{index:03d}", "valid": True}
        )
        for key, column in (
            ("time", "time_s"),
            ("ay", "ay_mps2"),
            ("roll", "roll_deg"),
            ("fz_fl", "Fz_FL"),
            ("fz_fr", "Fz_FR"),
            ("fz_rl", "Fz_RL"),
            ("fz_rr", "Fz_RR"),
        ):
            series[key][label] = [float(row[column]) for row in history]
        series["lltd"][label] = [
            100 * front_lltd([float(row[f"Fz_{c}"]) for c in ("FL", "FR", "RL", "RR")])
            if abs(float(row["ay_mps2"])) >= 1
            else np.nan
            for row in history
        ]
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "equal_increment_check.csv", equal_offset_rows)
    write_csv(output / "lltd_vs_ay.csv", lltd_rows)
    manifest["max_steady_lltd_change_pp_over_ay_grid"] = max(abs(row["change_from_baseline_pp"]) for row in lltd_rows)
    baseline_label = next(iter(series["time"]))
    transient_deviations = []
    for label in series["time"]:
        a, b = np.array(series["lltd"][label]), np.array(series["lltd"][baseline_label])
        valid = np.isfinite(a) & np.isfinite(b)
        transient_deviations.append(float(np.max(np.abs(a[valid] - b[valid]))))
    manifest["max_transient_lltd_change_from_baseline_pp_at_common_time_ay_above_1"] = max(transient_deviations)
    plots = {}
    for name, fields in (
        (
            "coupled_response",
            [
                ("ay", "Lateral acceleration", "Ay (m/s²)"),
                ("roll", "Body roll", "Roll (deg)"),
                ("lltd", "Transient front LLTD (|Ay| >= 1)", "Front share (%)"),
            ],
        ),
        ("tire_loads", [(f"fz_{c.lower()}", f"{c} tire normal load", "Fz (N)") for c in ("FL", "FR", "RL", "RR")]),
    ):
        plots[name] = {
            "layout": "triple" if len(fields) == 3 else "quad",
            "title": name.replace("_", " ").title(),
            "subplots": [
                {"title": title, "x": {"key": "time", "label": "Time (s)"}, "y": {"key": key, "label": unit}}
                for key, title, unit in fields
            ],
        }
    PlotEngine({"plots": plots}).save_pngs({"series": series}, output)
    manifest["all_cases_valid"] = True
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "max_steady_lltd_change_pp": manifest["max_steady_lltd_change_pp_over_ay_grid"],
            }
        ),
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicle", type=Path, default=repo_root() / "parametric_vehicle.yml")
    parser.add_argument("--output", type=Path, default=repo_root() / "_3_StandardSim/generated_results/coupled_rc")
    parser.add_argument("--offsets-mm", type=float, nargs="+", default=[0, 25, 50, 75, 100])
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=1e-8)
    args = parser.parse_args()
    run(args.vehicle, args.output, args.offsets_mm, dt=args.dt, rtol=args.rtol)


if __name__ == "__main__":
    main()
