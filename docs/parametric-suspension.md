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
