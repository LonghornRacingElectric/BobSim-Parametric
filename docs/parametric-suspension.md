# Parameterized suspension research fork

This branch adds a **hardpoint-free 6DOF research model** to BobSim. Edit
`parametric_vehicle.yml`, then run `make parametric-eval`. The analytic suspension
feeds the existing reduced tire, body-dynamics, trim and transient solvers.
Loading or running this input never solves hardpoints or reads FourPost metrics.
The original hardpoint/modelica workflows remain available.

The shipped numbers are an **illustrative vehicle**, not a correlated LHRe car.
The tire coefficients come from the identified BobSim TIR file, but the reduced
tire law is BobSim's smooth MF-derived projection, not full MF52.

## Inputs and conventions

`schema: bobsim.parametric.v1` deliberately separates this input from the
hardpoint vehicle schema. Unknown fields fail rather than silently doing nothing.
Mass/CG, wheelbase, track and inertia are explicit; suspension attachment points
are absent. Total mass includes four unsprung masses. Inertia tensors are supplied
about the model's total-vehicle CG; the 6DOF model uses the total inertia.

Each axle supports:

| Field | Meaning |
| --- | --- |
| `roll_center_height_m` | Nominal force-line RC height above flat road at zero jounce |
| `force_line_height_gradient` | Gradient of equivalent corner force-line height with jounce (m/m); **not** an independently prescribed whole-axle RC migration |
| `static_camber_deg` | Negative means top inward on both wheels |
| `camber_gain_deg_per_m` | Change per metre of positive corner bump; -20 means -0.5 degrees/25 mm |
| `camber_quadratic_deg_per_m2` | Optional quadratic camber coefficient |
| `toe_in_deg`, `toe_in_gain_deg_per_m` | Positive toe-in, with optional bump dependence |
| `motion_ratio_spring_per_wheel` | Constant spring travel divided by wheel travel |
| `spring_rate_n_per_m`, `damper_n_s_per_m` | Spring/damper element rates, projected by motion ratio squared |
| `arb_roll_stiffness_nm_per_rad` | Effective axle ARB roll stiffness, independent of spring motion ratio |
| `longitudinal_jacking_coefficient` | Signed geometric vertical force divided by longitudinal tire force |
| `travel_min_m`, `travel_max_m` | Valid corner-jounce domain; exceedance fails the case |

The dataclass also exposes wheel radius/inertia, unsprung mass and vertical tire
properties; defaults are explicit in `_0_Utils/dyn_py/parametric.py`. Put a field
in the YAML before selecting it as a sweep axis. Model axes are x forward, y
left, z up. Tire inclination outputs use signed vehicle lateral orientation;
mirrored negative static camber therefore has opposite signs on left and right.
The new backend transforms tire inclination to the road using body roll/pitch.

Constant `MR = spring travel / wheel travel` gives `k_w = k_s MR^2` and
`c_w = c_s MR^2`. This is the reciprocal of the legacy FourPost loader's
wheel/spring convention. Static spring preload is rebalanced to maintain the
specified baseline corner weights when MR or spring rate changes. Thus these
are constant-ride-height sensitivity comparisons. Rising-rate motion ratios and
the associated preload/geometric stiffness are not implemented in v1.

## What roll-center input does

For nominal half-track `b`, signed side `s` (+1 left), corner bump `q`, nominal
height `h`, and gradient `g`, the prescribed contact-patch path is:

```
y_offset = s * (h*q + 0.5*g*q^2) / b
z_offset = q
dy/dq = s * (h + g*q) / b
Fz_geometric / Fy = -dy/dq
```

Longitudinal displacement is similarly the integral of the negative supplied
longitudinal jacking coefficient. This enforces the local reciprocal/virtual-work
relationship. At zero jounce the left/right force lines intersect at `(y=0,z=h)`.
At nonzero jounce their intersection migrates according to those paths; `h` is
not imposed again as a constant moving-vehicle RC. Track migration follows the
same path and cannot independently be held fixed while keeping the same force
transmission. Camber is an independently prescribed research function.

The independent functions need not correspond to a buildable wishbone linkage.
Use the results to select target behavior, then synthesize and validate hardpoints.

## Running in Docker

From this checkout, with the BobSim image available:

```bash
make parametric-test
make parametric-eval
```

The default pilot runs baseline plus front-only and rear-only RC shifts of
plus/minus 20 mm. Each case solves matched Ay at 0, 4 and 8 m/s2, then a
2-degree **roadwheel** step at 15 m/s. The step starts at 1 s and rises over
0.15 s; a longitudinal controller holds speed. This input is not handwheel angle.

Change the YAML for a single vehicle or request a Cartesian grid:

```bash
make parametric-eval PARAMETRIC_ARGS='--grid front.roll_center_height_m=-0.01,0.02,0.05 --grid rear.roll_center_height_m=0,0.03,0.06'
make parametric-eval PARAMETRIC_ARGS='--grid front.camber_gain_deg_per_m=-10,-20,-30 --grid front.motion_ratio_spring_per_wheel=0.7,0.8,0.9'
```

Baseline is retained and duplicate configurations are skipped. A run is capped
at 100 grid combinations. `--vehicle`, `--output`, `--speed`, `--ays`,
`--step-deg`, `--duration`, `--dt`, and `--rtol` are also supported. Use a separate
output directory for each study; the default path is overwritten on reruns.

Outputs under `_3_StandardSim/generated_results/parametric/`:

- `steady.csv`: direct four-corner Fz, achieved Ay, steering, roll, axle load
  differences and trim acceptance. Matched Ay tolerance is 0.002 m/s2.
- `case_*/transient.csv`: Fz/Ay/roll/yaw, road-relative camber and jounce histories.
- `transient_metrics.csv`: final Ay, interpolated 10-90% response rise time, overshoot,
  minimum load, speed tracking and acceptance. Invalid/unsettled runs have no
  accepted rise/overshoot values; sample spacing bounds timing resolution.
- `response.png`: comparative Ay, roll and front/rear load differences.
- `manifest.json`: settings, input/tire/model-code hashes, case errors and limitations.
- `case_*/vehicle.yml` and `tire.tir`: self-contained candidate inputs.

Acceptance requires successful trim/integration, declared jounce limits, saved
tire loads inside the TIR fit domain, final response settling and less than
0.2 m/s speed error. The command exits with code 2 if any case is rejected, while
retaining its diagnostics. These gates are numerical/model-domain checks, not
experimental validation. No performance optimum or maximum Ay is inferred.

## Raising both roll centers at matched steady LLTD

```bash
make parametric-rc-coupled
make parametric-rc-coupled PARAMETRIC_ARGS='--offsets-mm 0 25 50 75 100 --output _3_StandardSim/generated_results/coupled_rc'
```

The sweep increases front nominal RC in the requested nonnegative offsets and
solves rear RC upward to retain the baseline **front total LLTD at 8 m/s2 and
15 m/s**. LLTD is `(Fz_FR-Fz_FL)/[(Fz_FR-Fz_FL)+(Fz_RR-Fz_RL)]`, using direct
simulated tire loads. Only the two RC inputs change; spring, ARB, damping,
motion-ratio and camber-function inputs stay fixed. Rear search is bounded to
baseline through baseline +200 mm; infeasible or invalid trims fail the study.

Equal height increments do not generally preserve total LLTD. The runner saves
an equal-increment diagnostic alongside the constrained pairs. It checks the
resulting steady LLTD at Ay = 2, 4, 6, 8, 10 m/s2, then runs the same 2-degree
roadwheel steer input for every pair (0.15 s rise starting at t = 1 s). Transient
LLTD is measured, not held constant, and is suppressed where |Ay| < 1 m/s2.
LLTD becomes ill-conditioned near zero transfer. This experiment constrains one
steady operating point, not the entire handling envelope or dynamic balance.

`generated_results/coupled_rc/` contains `summary.csv`, `lltd_vs_ay.csv`,
`equal_increment_check.csv`, comparative Ay/roll/LLTD and four-tire-load PNGs,
and a manifest. Each `pair_*/results/` retains the full standard parametric-study
inputs, hashes, CSVs and numerical/domain acceptance checks. `--vehicle`, `--dt`
and `--rtol` are supported; use separate output directories to retain reruns.

### Illustrative paired sweep, 2026-09-29

All 25 matched-Ay points and five transients passed their acceptance gates.
Front LLTD was 46.737693% at the reference Ay for all five pairs:

| Front RC (mm) | Rear RC (mm) | Roll at 8 m/s2 (deg) |
| ---: | ---: | ---: |
| 20 | 30.000 | 0.84166 |
| 45 | 52.626 | 0.76753 |
| 70 | 75.253 | 0.69386 |
| 95 | 97.879 | 0.62065 |
| 120 | 120.505 | 0.54789 |

At the endpoints, equal +100 mm increments (20/30 to 120/130 mm) would instead
reduce front LLTD to 45.903991%, a -0.833702 percentage-point change. The solved
rear increase was 90.505 mm. Across the 2-10 m/s2 steady grid, the largest LLTD
departure from the baseline at the same Ay was 0.106950 percentage points.

Endpoint results (baseline to highest pair):

- Body roll at matched 8 m/s2 fell 34.9%. The geometric share of total axle
  load differences increased from 8.57% to 40.33%.
- Steady Fz (FL/FR/RL/RR) changed from 356.70/882.69/453.38/1052.80 N to
  357.48/881.94/454.32/1052.00 N: less than 1 N per tire.
- Final Ay after the steer input changed from 5.171664 to 5.171549 m/s2
  (-0.00222%). These are finite-input responses, not maximum lateral grip.
- The maximum common-time Ay difference during turn-in was 0.08104 m/s2;
  individual tire-load differences reached 12.09 N. Raising RC therefore has
  a measurable transient effect despite the nearly identical settled loads.
- At t = 1.040 s, front LLTD was 50.88% baseline versus 59.64% highest RC,
  +8.76 percentage points. This is early turn-in at about 1.16 m/s2 in the high
  case, not the matched steady condition. Peak percentages depend strongly on
  the low-Ay gate and sample times: the finer run reaches +9.31 points with both
  cases above 1 m/s2, and +2.29 points with both above 4 m/s2.

Numerical audit: reran the endpoint pairs with 2.5 ms output/maximum integration
step and rtol 1e-9, versus 5 ms and 1e-8. At common timestamps, maximum Ay error
was 4.95e-7 m/s2, individual Fz error 2.47e-5 N and LLTD error 2.94e-6 percentage
points. Interpolated 10-90% rise-time reduction was about 0.30 ms (coarser run:
0.28 ms); do not treat this tiny input-dependent difference as a design benefit.
The saved `convergence.json` includes endpoint effects and refinement errors;
`turn_in.png` expands the first 0.8 s of the response. The convergence audit
script is retained beside these generated artifacts. This is numerical
agreement within the illustrative reduced model, not car correlation.

After adding the coupled runner, Docker CI passed lint, mypy and 396 tests
(6 skipped; saved standard-study artifacts were available). The focused
parameterized suite contains 15 passing tests. A fresh full Modelica baseline
regression again had 15 passes and the same two mismatches detailed below.

## Scope and validation

Only the 6DOF model is enabled through `load_parametric_vehicle`. The inherited
3DOF geometric-force closure is unsuitable here. A pilot static-balance check
exposed a separate total/sprung-CG moment issue in the inherited 14DOF initial
state for the illustrative mass distribution; 10/14DOF are not enabled pending
their own validation. This release has no wheel-hop dynamics, tire relaxation,
Ackermann, compliance, steering jacking, caster/KPI or bump stops. Aero, if enabled,
uses constant area coefficients and balance, without ride-height-dependent maps.

Tests cover no-hardpoint loading, mirrored force paths, virtual work, finite
derivatives, motion-ratio scaling, body-roll tire inclination, static balance,
left/right cornering symmetry and actual RC sensitivity. Repeat with tighter
integration tolerances and smaller output time steps before interpreting small
transient differences. Correlation to hardpoint-derived reduced curves and the
full BobLib reference is still required before design decisions.

The pose-aware hook is optional; legacy hardpoint backends follow their original
code path. Run repository checks and the full regression workflow when changing
shared physics; report unavailable or failed checks instead of altering baselines.

## Verification on 2026-09-29

- Docker lint and mypy passed. Repository suite: 382 passed, 18 skipped.
  After improving rise-time interpolation, the focused suite passed all 14 tests.
- Five RC cases at three matched-Ay points and five transients passed acceptance.
  A separate four-combination camber-gain/MR grid plus baseline also passed.
- RC results rerun with 10 ms instead of 20 ms maximum integration/output step
  and rtol 1e-8 instead of 1e-7 differed by at most 5.4e-6 m/s2 Ay and
  3.8e-4 N normal load at shared timestamps. This is numerical convergence,
  not physical correlation. Rise times interpolate saved samples and still
  depend on sampling resolution; do not rank unresolved timing differences.
- `make regression-baseline` rebuilt and ran all four original Modelica studies:
  15 checks passed, 2 failed. Ramp-steer `steer_excess_fit_nrmse` was
  0.1865292685 versus 0.0876623087 (tolerance 0.08); steady-state `n_cases`
  was 22 versus 40. No Modelica source, standard-study config or regression
  baseline was changed by this fork. This does **not** establish a clean full
  BobLib regression; those discrepancies require separate investigation.
- Docker cannot resolve the Windows absolute `.git` pointer of a linked
  worktree. The successful CI invocation mounted the main repository's `.git`
  read-only at `/git` and set `GIT_DIR=/git/worktrees/bobsim-parametric`,
  `GIT_COMMON_DIR=/git`, `GIT_WORK_TREE=/workspace`. A normal clone does not
  need that workaround. The first CI attempt failed only that Git lookup.
