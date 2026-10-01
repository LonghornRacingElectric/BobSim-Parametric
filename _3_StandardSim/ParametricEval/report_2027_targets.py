"""Publish conditional model-test targets without promoting them to hardware limits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from _0_Utils.plotting.matrix import save_matrix_panels
from _0_Utils.plotting.plot_engine import PlotEngine
from _0_Utils.vehicle_io import load_yaml, repo_root
from _3_StandardSim.ParametricEval.coupled_roll_center_sweep import read_csv
from _3_StandardSim.ParametricEval.parametric_eval import write_csv
from _3_StandardSim.ParametricEval.roll_center_matrix import response_metrics
from _3_StandardSim.ParametricEval.targets_2027 import save_json


def report(root: Path):
    phase_rows = {
        name: json.loads((root / f"{name}_summary.json").read_text())
        for name in ("pilot", "roll", "longitudinal", "validation", "refinement")
    }
    audit = json.loads((root / "projection_audit.json").read_text())
    lookup = {r["case"]: r for rows in phase_rows.values() for r in rows}
    references = {speed: lookup[f"fine_baseline_{speed}"] for speed in (10, 15, 20)}
    for rows in phase_rows.values():
        for row in rows:
            if not row["valid"] or row["kind"] != "steer":
                continue
            reference = references[int(row["speed"])]
            trace = read_csv(root / row["case"] / "trace.csv")
            row["common_reference_t90_ms"] = response_metrics(trace, reference)["turnin_mean_t90_ms"]
            row["gain_difference_pct"] = max(
                abs(row[f"{k}_final"] / reference[f"{k}_final"] - 1) * 100 for k in ("ay", "yaw")
            )
            row["response_gate_pass"] = bool(
                row["worst_overshoot_pct"] <= 5 and row["worst_settling_ms"] <= 1000 and row["gain_difference_pct"] <= 1
            )
    ranked = sorted(
        [r for r in phase_rows["roll"] if r["valid"] and r["response_gate_pass"]],
        key=lambda r: r["common_reference_t90_ms"],
    )
    best = ranked[0]
    checks = [r for phase in ("validation", "refinement") for r in phase_rows[phase] if r["kind"] == "steer"]
    check_columns = list(dict.fromkeys(k for row in checks for k in row))
    write_csv(root / "steering_checks.csv", [{k: r.get(k) for k in check_columns} for r in checks])
    near = [r for r in ranked if r["common_reference_t90_ms"] <= best["common_reference_t90_ms"] * 1.01]
    columns = list(dict.fromkeys(k for row in ranked for k in row))
    write_csv(root / "roll_ranking.csv", [{k: r.get(k) for k in columns} for r in ranked])
    convergence = []
    for name in ("baseline", "fast"):
        for speed in (10, 15, 20):
            coarse = read_csv(root / f"check_{name}_{speed}" / "trace.csv")
            fine = read_csv(root / f"fine_{name}_{speed}" / "trace.csv")
            tc = np.array([float(r["time_s"]) for r in coarse])
            tf = np.array([float(r["time_s"]) for r in fine])
            error: dict[str, str | int | float] = {"case": name, "speed": speed}
            for field in ("ay_mps2", "yaw_rate_radps", "Fz_FL", "Fz_FR", "Fz_RL", "Fz_RR"):
                error[f"max_error_{field}"] = float(
                    np.max(
                        np.abs(
                            np.array([float(r[field]) for r in coarse])
                            - np.interp(tc, tf, [float(r[field]) for r in fine])
                        )
                    )
                )
            error["t90_error_ms"] = abs(
                response_metrics(coarse, references[speed])["turnin_mean_t90_ms"]
                - response_metrics(fine, references[speed])["turnin_mean_t90_ms"]
            )
            convergence.append(error)
    save_json(root / "convergence.json", convergence)
    target = {
        "status": "conditional_model_test_window_only",
        "hardware_release": False,
        "all_speed_response_gates_pass": all(r["valid"] and r.get("response_gate_pass", False) for r in checks),
        "basis": "2027 workbook values, fixed-total ARB, rigid 6DOF, no aero, no rough-road/tire-relaxation validation",
        "nominal_test_front_rc_mm": 50,
        "nominal_test_rear_rc_mm": 20,
        "test_front_rc_window_mm": [50, 60],
        "test_rear_rc_window_mm": [10, 20],
        "test_anti_dive_fraction": [0.25, 0.5],
        "test_anti_squat_fraction": [0.25, 0.5],
        "anti_reference": "total vehicle nominal flat-road geometric support / (m*abs(ax)*h/L)",
        "window_selection": (
            "RC rectangle is a compact subset near the fastest 15 m/s grid response, retaining baseline-like roll. "
            "Anti 25-50% is a staged platform-control experiment giving roughly 25-50% less axle compression. "
            "No modeled rough-road or aero tradeoff identifies an optimum anti percentage. Zero anti remains feasible."
        ),
        "grid_fastest": best,
        "all_grid_pairs_within_1pct_of_best": [[r["front_rc_mm"], r["rear_rc_mm"]] for r in near],
        "source_audit": audit,
        "runs": {k: {"count": len(v), "domain_valid": sum(r["valid"] for r in v)} for k, v in phase_rows.items()},
        "maximum_refinement_t90_error_ms": max(r["t90_error_ms"] for r in convergence),
        "open_gates": [
            "10 m/s Ay overshoot >5% for all checked setups",
            "source aero moment/height conventions unresolved",
            "tire relaxation and unsprung/rough-road dynamics absent",
            "ARB range and travel are not hardware-verified",
            "camber/toe migration, full brake/drive torque path and chassis torsion not correlated",
        ],
    }
    # Self-contained editable model-test configurations, preserving the source baseline.
    for name in ("nominal25", "nominal50"):
        config = load_yaml(root / f"fine_{name}_15" / "input.yml")
        config["tire_file"] = "tire.tir"
        config["description"] = "2027 conditional mechanical MODEL TEST candidate; not hardware release"
        (root / f"candidate_{name}.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    rows = phase_rows["roll"]
    save_matrix_panels(
        root / "roll_matrix.png",
        rows,
        list(range(10, 61, 10)),
        [
            ("common_reference_t90_ms", "Mean Ay/yaw t90 to baseline amplitude (ms)", ".2f"),
            ("peak_roll_deg", "Peak roll in 15 m/s step steer (deg)", ".3f"),
            ("front_arb_nm_per_rad", "Front ARB effective roll stiffness (Nm/rad)", ".0f"),
            ("rear_arb_nm_per_rad", "Rear ARB effective roll stiffness (Nm/rad)", ".0f"),
        ],
        title="2027 mechanical screen | fixed total ARB | blank = balance unattainable in bracket",
    )
    diagonal = [r for r in phase_rows["longitudinal"] if r["anti_dive"] == r["anti_squat"]]
    series: dict[str, dict[str, list[float]]] = {k: {} for k in ("anti", "bump", "pitch")}
    for kind in ("brake", "drive"):
        items = sorted([r for r in diagonal if r["kind"] == kind], key=lambda r: r["anti_dive"])
        series["anti"][kind] = [100 * r["anti_dive"] for r in items]
        series["bump"][kind] = [r["peak_front_bump_mm" if kind == "brake" else "peak_rear_bump_mm"] for r in items]
        series["pitch"][kind] = [r["peak_pitch_deg"] for r in items]
    PlotEngine(
        {
            "plots": {
                "anti_platform": {
                    "layout": "dual",
                    "title": "Equal front anti-dive / rear anti-squat sweep",
                    "subplots": [
                        {
                            "title": "Braking: front; drive: rear",
                            "x": {"key": "anti", "label": "Reference anti (%)"},
                            "y": {"key": "bump", "label": "Peak compression (mm)"},
                        },
                        {
                            "title": "Whole-vehicle pitch",
                            "x": {"key": "anti", "label": "Reference anti (%)"},
                            "y": {"key": "pitch", "label": "Peak absolute pitch (deg)"},
                        },
                    ],
                }
            }
        }
    ).save_pngs({"series": series}, root)
    histories: dict[str, dict[str, list[float]]] = {}
    for field in ("time_s", "ay_mps2", "yaw_rate_radps", "roll_deg", "pitch_deg"):
        histories[field] = {}
        for name in ("baseline", "fast", "nominal25", "nominal50"):
            trace = read_csv(root / f"fine_{name}_15" / "trace.csv")
            histories[field][name] = [float(r[field]) for r in trace if 0.9 <= float(r["time_s"]) <= 2]
    PlotEngine(
        {
            "plots": {
                "turnin": {
                    "layout": "quad",
                    "title": "2027 turn-in | 15 m/s | 2 degree ramp over 0.15 s",
                    "subplots": [
                        {"title": title, "x": {"key": "time_s", "label": "Time (s)"}, "y": {"key": key, "label": unit}}
                        for key, title, unit in (
                            ("ay_mps2", "Lateral acceleration", "Ay (m/s2)"),
                            ("yaw_rate_radps", "Yaw rate", "rad/s"),
                            ("roll_deg", "Roll", "deg"),
                            ("pitch_deg", "Pitch", "deg"),
                        )
                    ],
                }
            }
        }
    ).save_pngs({"series": histories}, root)
    comparison = []
    for name in ("baseline", "fast", "nominal25", "nominal50"):
        steer = lookup[f"fine_{name}_15"]
        brake = lookup[f"fine_{name}_brake"]
        drive = lookup[f"fine_{name}_drive"]
        comparison.append(
            {
                "setup": name,
                "t90_ms": steer["common_reference_t90_ms"],
                "brake_front_bump_mm": brake["peak_front_bump_mm"],
                "drive_rear_bump_mm": drive["peak_rear_bump_mm"],
                "brake_pitch_deg": brake["peak_pitch_deg"],
                "drive_pitch_deg": drive["peak_pitch_deg"],
                "front_arb_nm_per_rad": steer["front_arb_nm_per_rad"],
                "rear_arb_nm_per_rad": steer["rear_arb_nm_per_rad"],
            }
        )
    target["comparison"] = comparison
    save_json(root / "targets.json", target)
    write_csv(root / "candidate_comparison.csv", comparison)
    header = """# 2027 parameterized anti targets

**Status: conditional model-test settings. No hardware target is released.**

The 2027 workbook block supplies mass, CG, track, RC and effective rates. Provisional hardpoints
are not used for suspension kinematics. This is rigid 6DOF, massless uprights, instantaneous
reduced tire forces and flat road. Aero is withheld.

## Recommended next model-test window

- Front RC: **50-60 mm**; rear RC: **10-20 mm**. Start at **50/20 mm** and retain the
  original **25.4/29.972 mm** control.
- Front anti-dive / rear anti-squat: test **25% and 50%**; retain **0%** controls.
  These are platform-response steps, not a demonstrated grip optimum.
- Keep wheel rates **11.038/16.274 N/mm**, inferred spring/wheel MR **0.6482/0.6097**,
  damping and alignment fixed.
- Tune ARBs to the source reference LLTD with **4421.95 Nm/rad total** fixed.
  Hardware adjustability is unverified.
- Percentages use total-car nominal load transfer, h=292.1 mm, L=1549.4 mm,
  84% front braking and rear drive. One translating-upright path applies in both force
  directions: rear brake anti-lift = AS * 0.16. Caliper/halfshaft/rotor paths are unvalidated.

This RC rectangle is a compact near-fast subset; the full 1% response band is in targets.json.
The fastest point lies on the grid boundary; no local/global optimum is established.
Zero anti already meets modeled travel limits, so these cases do not require nonzero anti.
The proposed 25-50% anti experiments trade roughly 25-50% less axle compression against
penalties the current model cannot yet quantify.

## Refined comparison at 15 m/s

Steering: 2 degree roadwheel ramp over 0.15 s. Braking/drive: separate -8/+5 m/s2
commanded-force pulses with release. t90 uses one common baseline final amplitude.
These are simulator outputs, not measured car behavior.

| Setup | Ay/yaw t90 (ms) | Front brake bump (mm) | Rear drive bump (mm) | Brake pitch (deg) | Drive pitch (deg) |
|---|---:|---:|---:|---:|---:|
"""
    table = []
    for r in comparison:
        table.append(
            f"| {r['setup']} | {r['t90_ms']:.3f} | {r['brake_front_bump_mm']:.3f} "
            f"| {r['drive_rear_bump_mm']:.3f} | {r['brake_pitch_deg']:.3f} | {r['drive_pitch_deg']:.3f} |"
        )
    footer = """

![Roll matrix](roll_matrix.png)

![Anti platform](anti_platform.png)

![Turn-in](turnin.png)

## Acceptance and limitations

Every checked 10 m/s setup, including baseline, fails the proposed 5% Ay overshoot gate
(about 15-16%). The 15 and 20 m/s candidates pass that response gate. Domain-valid does
not mean all requirements pass. Provisional +/-30 mm travel is not ground/wing clearance.

The source aero map gives front-load fractions of -119.1% to -0.264% under BobSim's
free-moment convention. No aero optimization is attempted. Current-package aero,
physical height datums and clearance limits are required to release platform targets.
Workbook longitudinal anti and camber/toe migration are unidentified. Zero anti is a
study reference, not measured geometry. Camber-gain sensitivity is not geometry validation.
Chassis torsion, tire relaxation, wheel hop, compliance and rough-road grip remain open.

The prior QSS helper allowed fictitious vertical velocity at nonzero pitch/roll.
This study uses a level-road trim adapter and checks zero world height, roll/pitch rates
and damper velocities. The underlying small-angle force model is unchanged. Its body-z
tire forces and finite-pose approximations limit conclusions beyond this screen.

## Reproduction

```sh
make parametric-2027-targets PARAMETRIC_ARGS='--phase pilot'
make parametric-2027-targets PARAMETRIC_ARGS='--phase roll'
make parametric-2027-targets PARAMETRIC_ARGS='--phase longitudinal'
make parametric-2027-targets PARAMETRIC_ARGS='--phase validation'
make parametric-2027-targets PARAMETRIC_ARGS='--phase refinement'
make parametric-2027-report
make parametric-2027-test
```

Make runs through Docker. Each phase retains executable inputs, traces, failure reasons
and source snapshots/hashes. vehicle.yml is the projected baseline; candidate_nominal25.yml
and candidate_nominal50.yml are editable hardpoint-free MODEL TEST inputs.
"""
    evidence = (
        f"\n\nRun counts: {target['runs']}.\n\n"
        f"Maximum 2.5-to-1.25 ms grid t90 change: {target['maximum_refinement_t90_error_ms']:.4f} ms.\n"
    )
    (root / "REPORT.md").write_text(header + "\n".join(table) + evidence + footer, encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=repo_root() / "_3_StandardSim/generated_results/targets_2027_level_trim"
    )
    args = parser.parse_args()
    target = report(args.output)
    print(json.dumps({"status": target["status"], "release": target["hardware_release"], "runs": target["runs"]}))


if __name__ == "__main__":
    main()
