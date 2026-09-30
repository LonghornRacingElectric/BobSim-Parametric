"""Hardpoint-free, analytic suspension input for BobSim research studies.

The nominal roll center defines the contact-patch tangent. Its integral supplies
contact-patch migration, and its negative reciprocal slopes supply jacking.
These cannot be set independently without violating virtual work.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike

from _0_Utils.dyn_py.kinematics import (
    CornerInstantLink,
    DoubleWishboneInstantLinks,
    KinematicsMode,
    VehicleKinematicState,
)
from _0_Utils.dyn_py.parameters import G, ReducedVehicleParameters, TireParameters
from _0_Utils.dyn_py.vehicle import Vehicle
from _0_Utils.dyn_py.models import DOFModel, VehicleDynamicsSystem
from _0_Utils.vehicle_io import load_yaml, parse_tir


@dataclass(frozen=True)
class ParametricAxle:
    track_m: float
    roll_center_height_m: float
    # Gradient of lateral tangent with corner jounce, expressed as an
    # equivalent force-line height gradient; not whole-axle RC migration.
    force_line_height_gradient: float = 0.0
    static_camber_deg: float = 0.0
    camber_gain_deg_per_m: float = 0.0
    camber_quadratic_deg_per_m2: float = 0.0
    toe_in_deg: float = 0.0
    toe_in_gain_deg_per_m: float = 0.0
    longitudinal_jacking_coefficient: float = 0.0
    motion_ratio_spring_per_wheel: float = 1.0
    spring_rate_n_per_m: float = 25000.0
    damper_n_s_per_m: float = 1500.0
    arb_roll_stiffness_nm_per_rad: float = 0.0
    wheel_radius_m: float = 0.2032
    wheel_inertia_kg_m2: float = 0.02
    unsprung_mass_per_corner_kg: float = 7.5
    tire_vertical_stiffness_n_per_m: float = 100000.0
    tire_vertical_damping_n_s_per_m: float = 100.0
    travel_min_m: float = -0.08
    travel_max_m: float = 0.08

    def __post_init__(self) -> None:
        if not all(np.isfinite(getattr(self, field.name)) for field in fields(self)):
            raise ValueError("Axle parameters must be finite numbers.")
        for key in (
            "track_m",
            "motion_ratio_spring_per_wheel",
            "spring_rate_n_per_m",
            "wheel_radius_m",
            "wheel_inertia_kg_m2",
            "unsprung_mass_per_corner_kg",
            "tire_vertical_stiffness_n_per_m",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive.")
        for key in ("damper_n_s_per_m", "arb_roll_stiffness_nm_per_rad", "tire_vertical_damping_n_s_per_m"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative.")
        if not self.travel_min_m < 0 < self.travel_max_m:
            raise ValueError("Travel range must straddle zero.")


@dataclass(frozen=True)
class ParametricKinematics:
    front: ParametricAxle
    rear: ParametricAxle
    mode: KinematicsMode = "parametric"

    def at(self, jounce_m: ArrayLike) -> VehicleKinematicState:
        q = np.asarray(jounce_m, dtype=float)
        if q.shape != (4,) or not np.all(np.isfinite(q)):
            raise ValueError("Jounce must be a finite FL, FR, RL, RR vector.")
        offsets = np.zeros((4, 3))
        tangents = np.zeros((4, 3))
        wheel_offsets = np.zeros((4, 3))
        camber, toe = np.zeros(4), np.zeros(4)
        for i, (axle, side) in enumerate(zip((self.front, self.front, self.rear, self.rear), (1, -1, 1, -1))):
            if not axle.travel_min_m <= q[i] <= axle.travel_max_m:
                raise ValueError(f"Corner {i} jounce {q[i]:.6g} m exceeds declared travel domain.")
            half_track = axle.track_m / 2
            lateral_slope = side * (axle.roll_center_height_m + axle.force_line_height_gradient * q[i]) / half_track
            longitudinal_slope = -axle.longitudinal_jacking_coefficient
            tangents[i] = (longitudinal_slope, lateral_slope, 1.0)
            offsets[i] = (
                longitudinal_slope * q[i],
                side
                * (axle.roll_center_height_m * q[i] + 0.5 * axle.force_line_height_gradient * q[i] ** 2)
                / half_track,
                q[i],
            )
            camber[i] = side * np.deg2rad(
                axle.static_camber_deg
                + axle.camber_gain_deg_per_m * q[i]
                + axle.camber_quadratic_deg_per_m2 * q[i] ** 2
            )
            toe[i] = -side * np.deg2rad(axle.toe_in_deg + axle.toe_in_gain_deg_per_m * q[i])
            nominal_camber = side * np.deg2rad(axle.static_camber_deg)
            wheel_offsets[i] = offsets[i] + axle.wheel_radius_m * np.array(
                [0, np.sin(camber[i]) - np.sin(nominal_camber), np.cos(camber[i]) - np.cos(nominal_camber)]
            )
        links = DoubleWishboneInstantLinks(
            tuple(  # type: ignore[arg-type]
                CornerInstantLink(float(-t[0]), float(-t[1])) for t in tangents
            )
        )
        return VehicleKinematicState(
            jounce_m=q.copy(),
            contact_patch_offsets_m=offsets,
            wheel_center_offsets_m=wheel_offsets,
            contact_patch_tangents=tangents,
            camber_rad=camber,
            toe_rad=toe,
            caster_rad=np.zeros(4),
            kpi_rad=np.zeros(4),
            mechanical_trail_m=np.zeros(4),
            scrub_radius_m=np.zeros(4),
            instant_links=links,
        )

    def at_pose(self, jounce_m: ArrayLike, body_to_world: ArrayLike, steering_rad: float) -> VehicleKinematicState:
        """Transform wheel inclination to a flat road, including body roll/pitch.

        Steering is the same imposed roadwheel angle at the two front wheels.
        No caster/KPI steer-camber or steering jacking is synthesized.
        """
        state = self.at(jounce_m)
        rotation = np.asarray(body_to_world, dtype=float)
        delta = state.toe_rad + np.array([steering_rad, steering_rad, 0, 0])
        gamma = state.camber_rad
        radial = np.column_stack((-np.sin(delta) * np.sin(gamma), np.cos(delta) * np.sin(gamma), np.cos(gamma)))
        forward = np.column_stack((np.cos(delta), np.sin(delta), np.zeros(4))) @ rotation.T
        radial_world = radial @ rotation.T
        lateral = np.column_stack((-forward[:, 1], forward[:, 0], np.zeros(4)))
        lateral /= np.linalg.norm(lateral, axis=1)[:, None]
        inclination = np.arctan2(np.sum(radial_world * lateral, axis=1), radial_world[:, 2])
        return replace(state, camber_rad=inclination)

    def instant_links_at(self, jounce_m: ArrayLike) -> DoubleWishboneInstantLinks:
        return self.at(jounce_m).instant_links


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    number = float(value)
    if not np.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive." if positive else "."))
    return number


def _inertia(value: Any, name: str) -> Any:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be a finite symmetric 3x3 tensor about the model CG.")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] <= 0 or eigenvalues[-1] > eigenvalues[:2].sum() + 1e-9:
        raise ValueError(f"{name} must have physical positive principal moments.")
    return tuple(tuple(float(v) for v in row) for row in matrix)


def parameters_from_mapping(data: Mapping[str, Any], *, base_dir: Path) -> ReducedVehicleParameters:
    """Load explicit parameters only: no hardpoint solver or FourPost CSV lookup."""
    allowed = {"schema", "description", "vehicle", "front", "rear", "tire_file", "aero", "powertrain"}
    if set(data) - allowed:
        raise ValueError(f"Unknown top-level parameter fields: {sorted(set(data) - allowed)}")
    if data.get("schema") != "bobsim.parametric.v1":
        raise ValueError("Expected schema bobsim.parametric.v1.")
    front, rear = ParametricAxle(**data["front"]), ParametricAxle(**data["rear"])
    vehicle = data["vehicle"]
    vehicle_keys = {
        "mass_kg",
        "cg_from_front_axle_m",
        "cg_height_m",
        "wheelbase_m",
        "inertia_kg_m2",
        "sprung_inertia_kg_m2",
    }
    if set(vehicle) != vehicle_keys:
        raise ValueError(f"vehicle requires exactly {sorted(vehicle_keys)}")
    mass = _finite(vehicle["mass_kg"], "mass_kg", positive=True)
    wheelbase = _finite(vehicle["wheelbase_m"], "wheelbase_m", positive=True)
    a = _finite(vehicle["cg_from_front_axle_m"], "cg_from_front_axle_m", positive=True)
    height = _finite(vehicle["cg_height_m"], "cg_height_m", positive=True)
    if a >= wheelbase:
        raise ValueError("CG must lie between axles.")
    sprung = mass - 2 * (front.unsprung_mass_per_corner_kg + rear.unsprung_mass_per_corner_kg)
    if sprung <= 0:
        raise ValueError("Total mass must exceed the four unsprung masses.")
    corners = (
        (a, front.track_m / 2, -height),
        (a, -front.track_m / 2, -height),
        (a - wheelbase, rear.track_m / 2, -height),
        (a - wheelbase, -rear.track_m / 2, -height),
    )
    static = (mass * G * (wheelbase - a) / wheelbase / 2,) * 2 + (mass * G * a / wheelbase / 2,) * 2
    if (
        min(static[:2]) <= front.unsprung_mass_per_corner_kg * G
        or min(static[2:]) <= rear.unsprung_mass_per_corner_kg * G
    ):
        raise ValueError("Each axle must support positive sprung load.")
    tire_values = parse_tir((base_dir / str(data["tire_file"])).resolve())
    tire_fields = {"fz_ref_n": "FNOMIN", "fz_min_n": "FZMIN", "fz_max_n": "FZMAX"}
    tire_fields.update(
        {
            name: name.upper()
            for name in (
                "pdx1",
                "pdx2",
                "pdy1",
                "pdy2",
                "pdy3",
                "pkx1",
                "pkx2",
                "pkx3",
                "pky1",
                "pky2",
                "pky3",
                "pvy3",
                "pvy4",
            )
        }
    )
    tire = TireParameters(**{name: _finite(tire_values[key], key) for name, key in tire_fields.items()})
    aero = dict(data.get("aero", {}))
    power = dict(data.get("powertrain", {}))
    aero_defaults = {"rho_air_kg_m3": 1.225, "cl_area_m2": 0.0, "cd_area_m2": 0.0, "balance_front": 0.5}
    power_defaults = {
        "peak_power_w": 80000.0,
        "peak_force_n": 3700.0,
        "maximum_speed_mps": 40.0,
        "drive_front_fraction": 0.0,
        "brake_front_fraction": 0.6,
    }
    for section, defaults in ((aero, aero_defaults), (power, power_defaults)):
        if set(section) - set(defaults):
            raise ValueError(f"Unknown fields: {sorted(set(section) - set(defaults))}")
        for key, default in defaults.items():
            section[key] = _finite(section.get(key, default), key)
            if section[key] < 0:
                raise ValueError(f"{key} must be nonnegative.")
    for value in (aero["balance_front"], power["drive_front_fraction"], power["brake_front_fraction"]):
        if not 0 <= value <= 1:
            raise ValueError("Axle fractions must be between zero and one.")

    def pair(name: str) -> tuple[float, float, float, float]:
        return (getattr(front, name),) * 2 + (getattr(rear, name),) * 2

    rates = tuple(
        axle.spring_rate_n_per_m * axle.motion_ratio_spring_per_wheel**2 for axle in (front, front, rear, rear)
    )
    damping = tuple(
        axle.damper_n_s_per_m * axle.motion_ratio_spring_per_wheel**2 for axle in (front, front, rear, rear)
    )
    return ReducedVehicleParameters(
        mass_kg=mass,
        sprung_mass_kg=sprung,
        inertia_kg_m2=_inertia(vehicle["inertia_kg_m2"], "inertia_kg_m2"),
        sprung_inertia_kg_m2=_inertia(vehicle["sprung_inertia_kg_m2"], "sprung_inertia_kg_m2"),
        corner_positions_m=corners,
        static_wheel_loads_n=static,
        wheel_radius_m=pair("wheel_radius_m"),
        wheel_inertia_kg_m2=pair("wheel_inertia_kg_m2"),
        unsprung_mass_kg=pair("unsprung_mass_per_corner_kg"),
        suspension_stiffness_n_per_m=rates,  # type: ignore[arg-type]
        suspension_damping_n_s_per_m=damping,  # type: ignore[arg-type]
        antiroll_stiffness_nm_per_rad=(front.arb_roll_stiffness_nm_per_rad, rear.arb_roll_stiffness_nm_per_rad),
        kinematics=ParametricKinematics(front, rear),
        tire_vertical_stiffness_n_per_m=pair("tire_vertical_stiffness_n_per_m"),
        tire_vertical_damping_n_s_per_m=pair("tire_vertical_damping_n_s_per_m"),
        tire=tire,
        rho_air_kg_m3=aero["rho_air_kg_m3"],
        cl_area_m2=aero["cl_area_m2"],
        cd_area_m2=aero["cd_area_m2"],
        aero_balance_front=aero["balance_front"],
        aero_cop_m=(a - wheelbase * (1 - aero["balance_front"]), 0.0, 0.0),
        aero_drag_application_m=(0.0, 0.0, 0.0),
        peak_drive_power_w=power["peak_power_w"],
        continuous_drive_power_w=power["peak_power_w"],
        peak_drive_force_n=power["peak_force_n"],
        continuous_drive_force_n=power["peak_force_n"],
        maximum_drive_speed_mps=power["maximum_speed_mps"],
        drive_distribution_front=power["drive_front_fraction"],
        brake_distribution_front=power["brake_front_fraction"],
    )


class ParametricVehicle(Vehicle):
    """Research model deliberately limited to the validated 6DOF closure."""

    def model(self, dof: DOFModel) -> VehicleDynamicsSystem:
        if dof != 6:
            raise ValueError("Parametric v1 supports 6DOF only; other closures require separate validation.")
        return super().model(dof)

    def with_power_limit(self, power_limit_w: float) -> ParametricVehicle:
        return ParametricVehicle(super().with_power_limit(power_limit_w).parameters)


def load_parametric_vehicle(path: str | Path) -> ParametricVehicle:
    source = Path(path).resolve()
    return ParametricVehicle(parameters_from_mapping(load_yaml(source), base_dir=source.parent))
