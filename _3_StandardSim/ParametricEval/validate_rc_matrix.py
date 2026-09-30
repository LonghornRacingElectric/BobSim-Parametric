"""Refine leading RC-matrix cases and check the same setups at other speeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np

from _0_Utils.vehicle_io import load_yaml, repo_root
from _3_StandardSim.ParametricEval.coupled_roll_center_sweep import read_csv
from _3_StandardSim.ParametricEval.parametric_eval import write_csv
from _3_StandardSim.ParametricEval.roll_center_matrix import execute_case, rank_rows, response_metrics


def run(matrix: Path, speeds: list[float], workers: int = 4, dt: float = 0.0025, rtol: float = 1e-9):
    matrix = matrix.resolve()
    manifest = json.loads((matrix / "manifest.json").read_text(encoding="utf-8"))
    ranked = read_csv(matrix / "ranking.csv")
    # Include the first runner-up, reference and low-roll/low-overshoot alternatives.
    selected = list(
        dict.fromkeys(
            [
                manifest["reference_case"],
                ranked[0]["case"],
                ranked[1]["case"],
                ranked[-1]["case"],
                min(ranked, key=lambda r: float(r["steady_roll_deg"]))["case"],
            ]
        )
    )
    candidates = {r["case"]: r for r in read_csv(matrix / "summary.csv")}
    source = matrix / "source.yml"
    data = load_yaml(source)
    data["tire_file"] = "tire.tir"
    jobs = []
    for speed in speeds:
        output = matrix / "validation" / f"speed_{speed:g}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "tire.tir").write_bytes((matrix / "tire.tir").read_bytes())
        for name in ["original", *selected]:
            config = manifest["original"] if name == "original" else candidates[name]
            jobs.append(
                {
                    "case": name,
                    "front_rc_mm": float(config["front_rc_mm"]),
                    "rear_rc_mm": float(config["rear_rc_mm"]),
                    "original": name == "original",
                    "data": data,
                    "base_dir": matrix.as_posix(),
                    "output": output.as_posix(),
                    "target": manifest["target_front_lltd_pct"] / 100,
                    "total": manifest["total_arb_nm_per_rad"],
                    "speed": speed,
                    "dt": dt,
                    "rtol": rtol,
                }
            )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        all_results = list(pool.map(execute_case, jobs))
    rows, convergence = [], []
    for speed in speeds:
        speed_rows = [r for r, job in zip(all_results, jobs) if job["speed"] == speed]
        output = matrix / "validation" / f"speed_{speed:g}"
        reference = next(r for r in speed_rows if r["case"] == manifest["reference_case"])
        if not reference["valid"]:
            raise RuntimeError(f"Validation reference failed at {speed:g} m/s.")
        ranking = rank_rows(speed_rows, output, reference)
        for row in speed_rows:
            row["speed_mps"] = speed
        rows.extend(speed_rows)
        write_csv(output / "ranking.csv", ranking)
        if speed == 15:
            for row in speed_rows:
                if not row["valid"]:
                    continue
                name = row["case"]
                coarse = read_csv(matrix / name / "results/case_000/transient.csv")
                fine = read_csv(output / name / "results/case_000/transient.csv")
                tc = np.array([float(r["time_s"]) for r in coarse])
                tf = np.array([float(r["time_s"]) for r in fine])
                item: dict[str, Any] = {"case": name}
                for col in ("ay_mps2", "yaw_rate_radps", "Fz_FL", "Fz_FR", "Fz_RL", "Fz_RR"):
                    yc, yf = np.array([float(r[col]) for r in coarse]), np.array([float(r[col]) for r in fine])
                    item[f"max_abs_error_{col}"] = float(np.max(np.abs(yc - np.interp(tc, tf, yf))))
                # Use one common reference for the convergence comparison, avoiding threshold shifts.
                m1, m2 = response_metrics(coarse, reference), response_metrics(fine, reference)
                item["turnin_mean_t90_error_ms"] = m1["turnin_mean_t90_ms"] - m2["turnin_mean_t90_ms"]
                convergence.append(item)
    write_csv(matrix / "validation_summary.csv", rows)
    result = {
        "speeds_mps": speeds,
        "selected_cases": selected,
        "dt_s": dt,
        "rtol": rtol,
        "all_valid": all(r["valid"] for r in rows),
        "convergence_at_15_mps": convergence,
        "scope": "Same ARB/RC setups retained at each speed; tuned at 15 m/s and Ay=8 only.",
        "note": "Shortlist speed check, not a full additional matrix or physical correlation.",
    }
    (matrix / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_valid": result["all_valid"], "validated_runs": len(rows)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=repo_root() / "_3_StandardSim/generated_results/rc_matrix")
    parser.add_argument("--speeds", nargs="+", type=float, default=[10, 15, 20])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--rtol", type=float, default=1e-9)
    args = parser.parse_args()
    result = run(args.matrix, args.speeds, args.workers, args.dt, args.rtol)
    if not result["all_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
