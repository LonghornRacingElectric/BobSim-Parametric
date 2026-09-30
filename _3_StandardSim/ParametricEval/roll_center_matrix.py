"""Independent front/rear RC matrix with constant-total ARBs tuned for steady LLTD."""

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

from _0_Utils.plotting.matrix import save_matrix_panels
from _0_Utils.plotting.plot_engine import PlotEngine
from _0_Utils.vehicle_io import load_yaml, repo_root
from _3_StandardSim.ParametricEval.coupled_roll_center_sweep import evaluate_pair, front_lltd, read_csv
from _3_StandardSim.ParametricEval.parametric_eval import run_study, threshold_time, write_csv

ARB = "arb_roll_stiffness_nm_per_rad"
RC = "roll_center_height_m"


def tune_arb(data, base_dir, front_mm, rear_mm, target, total, speed=15.0, ay=8.0):
    """Solve one ARB split; reject negative-rate requirements instead of clipping."""
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Total ARB roll stiffness must be finite and positive.")
    candidate = copy.deepcopy(data)
    candidate["front"][RC], candidate["rear"][RC] = front_mm / 1000, rear_mm / 1000

    def residual(front_rate):
        candidate["front"][ARB], candidate["rear"][ARB] = front_rate, total - front_rate
        _, trim = evaluate_pair(candidate, base_dir, front_mm / 1000, rear_mm / 1000, speed, ay)
        return front_lltd(trim.output.normal_loads_n) - target

    low, high = residual(0), residual(total)
    if low * high > 0:
        raise ValueError(
            f"Target LLTD {100 * target:.6f}% is outside attainable "
            f"[{100 * (low + target):.6f}, {100 * (high + target):.6f}]% at total ARB {total:g}."
        )
    rate = brentq(residual, 0, total, xtol=1e-6)
    residual(rate)
    audited = copy.deepcopy(candidate)
    for axle in ("front", "rear"):
        for key in (RC, ARB):
            audited[axle][key] = data[axle][key]
    if audited != data:
        raise AssertionError("An input other than RC/ARB changed.")
    return candidate


def response_metrics(history: list[dict[str, str]], reference: dict[str, float] | None = None) -> dict[str, float]:
    times = np.array([float(row["time_s"]) for row in history])
    output = {}
    for name, column in (("ay", "ay_mps2"), ("yaw", "yaw_rate_radps")):
        values = np.array([float(row[column]) for row in history])
        final = float(np.mean(values[times >= times[-1] - 0.5]))
        if final <= 1e-6:
            raise ValueError("Turn-in metrics require a positive final response.")
        target = final if reference is None else reference[f"{name}_final"]
        t10 = threshold_time(times, values, 0.1 * final)
        t90 = threshold_time(times, values, 0.9 * target)
        own_t90 = threshold_time(times, values, 0.9 * final)
        if t10 is None or t90 is None or own_t90 is None:
            raise ValueError("Response never reaches its turn-in threshold.")
        outside = np.flatnonzero((times >= 1) & (np.abs(values / final - 1) > 0.02))
        if outside.size and outside[-1] >= len(times) - 1:
            raise ValueError("Response does not settle within 2% before the run ends.")
        settling = float(times[outside[-1] + 1] - 1) if outside.size else 0.0
        output.update(
            {
                f"{name}_final": final,
                f"{name}_t90_s": float(t90 - 1),
                f"{name}_rise_10_90_s": float(own_t90 - t10),
                f"{name}_overshoot_pct": max(0.0, float(100 * (np.max(values) / final - 1))),
                f"{name}_settling_2pct_s": settling,
            }
        )
    output["turnin_mean_t90_ms"] = 500 * (output["ay_t90_s"] + output["yaw_t90_s"])
    output["yaw_t90_ms"] = 1000 * output["yaw_t90_s"]
    output["worst_overshoot_pct"] = max(output["ay_overshoot_pct"], output["yaw_overshoot_pct"])
    output["worst_settling_ms"] = 1000 * max(output["ay_settling_2pct_s"], output["yaw_settling_2pct_s"])
    return output


def execute_case(job: dict[str, Any]) -> dict[str, Any]:
    data, base_dir = job["data"], Path(job["base_dir"])
    root = Path(job["output"])
    front, rear = job["front_rc_mm"], job["rear_rc_mm"]
    row: dict[str, Any] = {"case": job["case"], "front_rc_mm": front, "rear_rc_mm": rear}
    case_dir = root / job["case"]
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        candidate = data if job.get("original") else tune_arb(data, base_dir, front, rear, job["target"], job["total"])
        row.update({"front_arb_nm_per_rad": candidate["front"][ARB], "rear_arb_nm_per_rad": candidate["rear"][ARB]})
        _, trim = evaluate_pair(candidate, base_dir, front / 1000, rear / 1000, 15, 8)
        row.update(
            {
                "front_lltd_pct": 100 * front_lltd(trim.output.normal_loads_n),
                "steady_roll_deg": float(np.rad2deg(trim.state[3])),
            }
        )
        saved = copy.deepcopy(candidate)
        saved["tire_file"] = "../tire.tir"
        source = case_dir / "input.yml"
        source.write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")
        run = run_study(
            source,
            case_dir / "results",
            grids=[f"front.{RC}={front / 1000}"],
            speed=job["speed"],
            targets=[2, 4, 6, 8, 10],
            step_deg=2,
            duration=4,
            dt=job["dt"],
            rtol=job["rtol"],
        )
        if not run["all_cases_valid"]:
            raise ValueError("Case rejected by solver, travel, tire-domain, speed or settling gate.")
        steady = read_csv(case_dir / "results/steady.csv")
        lltds = [100 * front_lltd([float(r[f"Fz_{c}"]) for c in ("FL", "FR", "RL", "RR")]) for r in steady]
        row["max_steady_lltd_error_pp"] = max(abs(x - 100 * job["target"]) for x in lltds)
        history = read_csv(case_dir / "results/case_000/transient.csv")
        row.update(response_metrics(history))
        row["min_tire_load_n"] = min(float(r[f"Fz_{c}"]) for r in history for c in ("FL", "FR", "RL", "RR"))
        row["max_speed_error_mps"] = max(abs(float(r["speed_mps"]) - job["speed"]) for r in history)
        row["valid"] = True
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        row.update({"valid": False, "error": str(exc)})
    (case_dir / "summary.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{job['case']}: valid={row['valid']}", flush=True)
    return row


def rank_rows(rows: list[dict[str, Any]], output: Path, reference: dict[str, float]) -> list[dict[str, Any]]:
    for row in rows:
        if not row["valid"]:
            row["eligible"] = False
            continue
        history = read_csv(output / row["case"] / "results/case_000/transient.csv")
        row.update(response_metrics(history, reference))
        row["eligible"] = bool(
            row["worst_overshoot_pct"] <= 5
            and row["worst_settling_ms"] <= 1000
            and all(abs(row[f"{k}_final"] / reference[f"{k}_final"] - 1) <= 0.01 for k in ("ay", "yaw"))
        )
    ranked = sorted((r for r in rows if r["eligible"]), key=lambda r: r["turnin_mean_t90_ms"])
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
        keys = ("turnin_mean_t90_ms", "worst_overshoot_pct", "worst_settling_ms")
        row["pareto"] = not any(
            all(other[k] <= row[k] for k in keys) and any(other[k] < row[k] for k in keys) for other in ranked
        )
    return ranked


def figures(output: Path, rows: list[dict[str, Any]], levels: list[float], reference_case: str, best: str, worst: str):
    save_matrix_panels(
        output / "response_matrix.png",
        rows,
        levels,
        [
            ("turnin_mean_t90_ms", "Mean Ay/yaw time to 90% of reference (ms); lower is faster", ".2f"),
            ("yaw_t90_ms", "Yaw time to 90% of reference (ms)", ".2f"),
            ("worst_overshoot_pct", "Maximum Ay/yaw overshoot (%)", ".3f"),
            ("worst_settling_ms", "Maximum Ay/yaw settling to 2% (ms)", ".0f"),
        ],
        title="RC/ARB matrix: identical 2° roadwheel ramp, 15 m/s; illustrative 6DOF vehicle",
    )
    save_matrix_panels(
        output / "setup_matrix.png",
        rows,
        levels,
        [
            ("front_arb_nm_per_rad", "Front effective ARB roll stiffness (Nm/rad)", ".0f"),
            ("rear_arb_nm_per_rad", "Rear effective ARB roll stiffness (Nm/rad)", ".0f"),
            ("steady_roll_deg", "Body roll at Ay = 8 m/s² (deg)", ".3f"),
            ("max_steady_lltd_error_pp", "Maximum LLTD error vs target, Ay 2–10 (pp)", ".3f"),
        ],
        title="Nonnegative ARBs; constant total stiffness; front LLTD matched at Ay = 8 m/s²",
    )
    series: dict[str, dict[str, Any]] = {k: {} for k in ("time", "ay", "yaw", "roll", "lltd")}
    for case in dict.fromkeys(("original", reference_case, best, worst)):
        history = read_csv(output / case / "results/case_000/transient.csv")
        history = [r for r in history if 1 <= float(r["time_s"]) <= 2]
        for key, col in (("time", "time_s"), ("ay", "ay_mps2"), ("yaw", "yaw_rate_radps"), ("roll", "roll_deg")):
            series[key][case] = [float(r[col]) for r in history]
        series["lltd"][case] = [
            100 * front_lltd([float(r[f"Fz_{c}"]) for c in ("FL", "FR", "RL", "RR")])
            if abs(float(r["ay_mps2"])) >= 1
            else np.nan
            for r in history
        ]
    subplots = [
        {"title": title, "x": {"key": "time", "label": "Time (s)"}, "y": {"key": key, "label": unit}}
        for key, title, unit in [
            ("ay", "Lateral acceleration", "Ay (m/s²)"),
            ("yaw", "Yaw response", "Yaw rate (rad/s)"),
            ("roll", "Body roll", "Roll (deg)"),
            ("lltd", "Transient front LLTD (|Ay| >= 1)", "Front share (%)"),
        ]
    ]
    PlotEngine(
        {"plots": {"turn_in_comparison": {"layout": "quad", "title": "RC matrix comparison", "subplots": subplots}}}
    ).save_pngs({"series": series}, output)


def run(
    source: Path,
    output: Path,
    levels: list[float],
    *,
    total: float,
    target_pct: float | None,
    workers: int = 4,
    dt: float = 0.005,
    rtol: float = 1e-8,
):
    if not levels or not np.all(np.isfinite(levels)) or len(set(levels)) ** 2 > 100:
        raise ValueError("Provide finite RC heights for at most 100 grid points.")
    levels = sorted(set(levels))
    source, output = source.resolve(), output.resolve()
    data = load_yaml(source)
    _, baseline = evaluate_pair(data, source.parent, data["front"][RC], data["rear"][RC], 15, 8)
    target = front_lltd(baseline.output.normal_loads_n) if target_pct is None else target_pct / 100
    if not 0 < target < 1:
        raise ValueError("Front LLTD target must be between zero and 100 percent.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "source.yml").write_bytes(source.read_bytes())
    (output / "tire.tir").write_bytes((source.parent / data["tire_file"]).read_bytes())
    base = {
        "data": data,
        "base_dir": source.parent.as_posix(),
        "output": output.as_posix(),
        "target": target,
        "total": total,
        "speed": 15,
        "dt": dt,
        "rtol": rtol,
    }
    jobs = [
        {
            **base,
            "case": "original",
            "front_rc_mm": data["front"][RC] * 1000,
            "rear_rc_mm": data["rear"][RC] * 1000,
            "original": True,
        }
    ]
    jobs += [
        {**base, "case": f"F{f:03.0f}_R{r:03.0f}", "front_rc_mm": f, "rear_rc_mm": r} for f in levels for r in levels
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(execute_case, jobs))
    original, rows = results[0], results[1:]
    reference_case = f"F{data['front'][RC] * 1000:03.0f}_R{data['rear'][RC] * 1000:03.0f}"
    reference = next((r for r in rows if r["case"] == reference_case and r["valid"]), original)
    reference_case = reference["case"]
    if not reference["valid"]:
        raise RuntimeError("Reference case failed.")
    ranked = rank_rows(rows, output, reference)
    write_csv(output / "summary.csv", rows)
    write_csv(output / "ranking.csv", ranked)
    for field in ("turnin_mean_t90_ms", "front_arb_nm_per_rad", "rear_arb_nm_per_rad"):
        lookup = {(r["front_rc_mm"], r["rear_rc_mm"]): r for r in rows}
        wide = [{"front_rc_mm": f, **{f"rear_{r:g}_mm": lookup[f, r].get(field) for r in levels}} for f in levels]
        write_csv(output / f"matrix_{field}.csv", wide)
    manifest = {
        "study": "rc_matrix_arb_balanced",
        "levels_mm": levels,
        "target_front_lltd_pct": target * 100,
        "total_arb_nm_per_rad": total,
        "original_total_arb_nm_per_rad": data["front"][ARB] + data["rear"][ARB],
        "reference_speed_mps": 15,
        "reference_ay_mps2": 8,
        "dt_s": dt,
        "rtol": rtol,
        "valid_cases": sum(r["valid"] for r in rows),
        "eligible_cases": len(ranked),
        "reference_case": reference_case,
        "original": original,
        "best_case": ranked[0]["case"] if ranked else None,
        "ranking": "Minimize mean Ay/yaw time from t=1s to 90% of the common reference final response.",
        "gates": "Valid domain; <=5% Ay/yaw overshoot; <=1s 2% settling; final Ay/yaw within 1% of reference.",
        "limitation": "Illustrative uncorrelated 6DOF model; one input/speed and finite grid, not a vehicle optimum.",
        "lltd_constraint": "Matched at 15 m/s and Ay=8; other steady and transient LLTD measured, not constrained.",
        "vehicle_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if ranked:
        figures(output, rows, levels, reference_case, ranked[0]["case"], ranked[-1]["case"])
    print(
        json.dumps({"valid": manifest["valid_cases"], "eligible": len(ranked), "best": manifest["best_case"]}),
        flush=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicle", type=Path, default=repo_root() / "parametric_vehicle.yml")
    parser.add_argument("--output", type=Path, default=repo_root() / "_3_StandardSim/generated_results/rc_matrix")
    parser.add_argument("--heights-mm", type=float, nargs="+", default=[10, 20, 30, 40, 50, 60])
    parser.add_argument("--total-arb", type=float, default=5000)
    parser.add_argument("--target-front-lltd-pct", type=float)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=1e-8)
    args = parser.parse_args()
    result = run(
        args.vehicle,
        args.output,
        args.heights_mm,
        total=args.total_arb,
        target_pct=args.target_front_lltd_pct,
        workers=args.workers,
        dt=args.dt,
        rtol=args.rtol,
    )
    if result["valid_cases"] != len(set(args.heights_mm)) ** 2:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
