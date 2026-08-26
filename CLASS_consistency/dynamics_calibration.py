#!/usr/bin/env python3
"""Shared final-accuracy KG-Friedmann calibration utilities.

Conventions follow the CLASS-like Planck 2018 base-LambdaCDM background:
omega_m includes one 0.06-eV massive neutrino with T_ncdm/T_gamma=0.71611,
while N_ur=2.0328 effective species are massless.  The combined early-time
relativistic content is N_eff=3.046.
The comoving distance is integrated as an ODE state and crossings are located
by solve_ivp events, not by a sampled plotting grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
from pathlib import Path
import sys
import json
from typing import Callable

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import solve_ivp, quad

C_KM_S = 299792.458
Z_STAR = 1090.0
T_CMB = 2.7255
N_EFF = 3.046
N_UR = 2.0328
N_NCDM_EFF = N_EFF - N_UR
MNU_EV = 0.06
K_B_EV_K = 8.617333262e-5
T_NCDM_OVER_TGAMMA = 0.71611
T_NU0_EV = T_CMB * T_NCDM_OVER_TGAMMA * K_B_EV_K

# Planck 2018 TT,TE,EE+lowE+lensing best-fit values.
OMEGA_B = 0.022383
OMEGA_CDM = 0.12011
OMEGA_NU = MNU_EV / 93.14
OMEGA_CB = OMEGA_B + OMEGA_CDM
OMEGA_M = OMEGA_CB + OMEGA_NU
OMEGA_GAMMA = 2.4729792808613565e-5

F_NU = (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)
OMEGA_NU_MASSLESS = OMEGA_GAMMA * F_NU * N_UR

D_M_CAL = 13869.57
D_M_SCALE = 25.0  # indicative propagated Planck scale, not an exact chain error
E0_SCALE = 1.0e-4
ZDAGGER_SCALE = 0.03
XI = -1.0


@dataclass(frozen=True)
class Potential:
    name: str
    value: Callable[[np.ndarray | float], np.ndarray | float]
    deriv: Callable[[np.ndarray | float], np.ndarray | float]
    second: Callable[[np.ndarray | float], np.ndarray | float]


def _fermi_integrals(y: float) -> tuple[float, float]:
    """Dimensionless massive-neutrino energy and pressure integrals."""
    f = lambda q: q * q * np.sqrt(q * q + y * y) / (np.exp(q) + 1.0)
    g = lambda q: q**4 / (3.0 * np.sqrt(q * q + y * y) * (np.exp(q) + 1.0))
    return quad(f, 0.0, 40.0, epsabs=1e-10, epsrel=1e-9)[0], quad(g, 0.0, 40.0, epsabs=1e-10, epsrel=1e-9)[0]


_Y0 = MNU_EV / T_NU0_EV
_I0, _ = _fermi_integrals(_Y0)
_LOG1P_Z_TAB = np.linspace(0.0, np.log1p(Z_STAR), 700)
_NU_CACHE = Path(__file__).resolve().parent / "dynamics_final_neff3046/cache/massive_neutrino_table_class_neff3046_class_photon_v3.npz"


def _build_neutrino_table() -> tuple[np.ndarray, np.ndarray]:
    ztab = np.expm1(_LOG1P_Z_TAB)
    itab = np.empty_like(ztab)
    wtab = np.empty_like(ztab)
    for i, z in enumerate(ztab):
        integ, pressure = _fermi_integrals(_Y0 / (1.0 + z))
        itab[i] = integ
        wtab[i] = pressure / integ
    return itab, wtab


def _load_or_build_neutrino_table() -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(_NU_CACHE) as cached:
            if np.isclose(float(cached["mnu_eV"]), MNU_EV) and np.isclose(float(cached["Tnu0_eV"]), T_NU0_EV) and np.isclose(float(cached["T_ncdm_over_Tgamma"]), T_NCDM_OVER_TGAMMA) and np.isclose(float(cached["N_ur"]), N_UR) and np.isclose(float(cached["omega_nu"]), OMEGA_NU) and np.isclose(float(cached["omega_gamma"]), OMEGA_GAMMA) and np.isclose(float(cached["T_CMB_K"]), T_CMB) and np.isclose(float(cached["z_star"]), Z_STAR) and np.array_equal(cached["log1p_z"], _LOG1P_Z_TAB):
                return cached["I"], cached["w"]
    except (FileNotFoundError, KeyError, OSError, ValueError):
        pass
    itab, wtab = _build_neutrino_table()
    try:
        _NU_CACHE.parent.mkdir(parents=True, exist_ok=True)
        temporary = _NU_CACHE.with_name(f"{_NU_CACHE.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, mnu_eV=MNU_EV, Tnu0_eV=T_NU0_EV, T_ncdm_over_Tgamma=T_NCDM_OVER_TGAMMA, N_ur=N_UR, N_eff=N_EFF, omega_nu=OMEGA_NU, omega_gamma=OMEGA_GAMMA, T_CMB_K=T_CMB, z_star=Z_STAR, log1p_z=_LOG1P_Z_TAB, I=itab, w=wtab)
        os.replace(temporary, _NU_CACHE)
    except OSError:
        pass
    return itab, wtab


_I_TAB, _W_TAB = _load_or_build_neutrino_table()


def massive_neutrino(z):
    """Return (rho_nu/rho_c0, w_nu) for one 0.06-eV thermal species."""
    za = np.asarray(z, dtype=float)
    u = np.log1p(np.clip(za, 0.0, Z_STAR))
    integ = np.interp(u, _LOG1P_Z_TAB, _I_TAB)
    w = np.interp(u, _LOG1P_Z_TAB, _W_TAB)
    rho = OMEGA_NU * (1.0 + za) ** 4 * integ / _I0
    return rho, w


def background_terms(z: float | np.ndarray, H0: float):
    h2 = (H0 / 100.0) ** 2
    zp1 = 1.0 + np.asarray(z)
    cb = OMEGA_CB / h2 * zp1**3
    rad = (OMEGA_GAMMA + OMEGA_NU_MASSLESS) / h2 * zp1**4
    nu, wnu = massive_neutrino(z)
    nu = nu / h2
    return cb, rad, nu, wnu


def e2_of(phi, phip, z, H0: float, potential: Potential):
    cb, rad, nu, _ = background_terms(z, H0)
    num = cb + rad + nu + potential.value(phi)
    den = 1.0 - XI * (1.0 + np.asarray(z)) ** 2 * np.asarray(phip) ** 2 / 6.0
    out = np.asarray(num / den)
    if np.ndim(out) == 0:
        return float(out) if den > 0.0 and num > 0.0 else np.nan
    out[(den <= 0.0) | (num <= 0.0)] = np.nan
    return out


def ep_of(phi, phip, z, H0: float, potential: Potential):
    E2 = e2_of(phi, phip, z, H0, potential)
    E = np.sqrt(E2)
    cb, rad, nu, wnu = background_terms(z, H0)
    kinetic = XI * (1.0 + np.asarray(z)) ** 2 * E2 * np.asarray(phip) ** 2 / 6.0
    return (3.0 * cb + 4.0 * rad + 3.0 * nu * (1.0 + wnu) + 6.0 * kinetic) / (2.0 * (1.0 + np.asarray(z)) * E)


def _rhs(H0: float, potential: Potential):
    def rhs(z, y):
        phi, phip, _distance = y
        E2 = e2_of(phi, phip, z, H0, potential)
        if not np.isfinite(E2) or E2 <= 0.0:
            raise FloatingPointError("invalid E^2")
        E = np.sqrt(E2)
        Ep = ep_of(phi, phip, z, H0, potential)
        zp1 = 1.0 + z
        phipp = (2.0 * zp1 * E2 * phip - zp1**2 * E * Ep * phip - 3.0 * XI * potential.deriv(phi)) / (zp1**2 * E2)
        return phip, phipp, -C_KM_S / (H0 * E)
    return rhs


def _events(H0: float, potential: Potential):
    def potential_zero(z, y):
        return float(potential.value(y[0]))

    def density_zero(z, y):
        phi, phip = y[0], y[1]
        E2 = e2_of(phi, phip, z, H0, potential)
        kinetic = XI * (1.0 + z) ** 2 * E2 * phip**2 / 6.0
        return float(potential.value(phi) + kinetic)

    potential_zero.terminal = False
    density_zero.terminal = False
    potential_zero.direction = 0
    density_zero.direction = 0
    return potential_zero, density_zero


def solve_trajectory(H0: float, potential: Potential, phi_ini: float, phip_ini: float = 0.0,
                     rtol: float = 1e-9, atol_field: float = 1e-11,
                     atol_distance: float = 1e-7, max_step: float = 0.2):
    events = _events(H0, potential)
    sol = solve_ivp(
        _rhs(H0, potential), (Z_STAR, 0.0), (phi_ini, phip_ini, 0.0),
        method="Radau", rtol=rtol,
        atol=np.array([atol_field, atol_field, atol_distance]),
        max_step=max_step, dense_output=True, events=events,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def trajectory_is_finite_and_physical(sol, H0: float, potential: Potential) -> bool:
    """Check solver success, finite states, and positive finite E^2 at all solver nodes."""
    if not bool(sol.success) or not np.all(np.isfinite(sol.y)) or not np.all(np.isfinite(sol.t)):
        return False
    e2 = np.asarray(e2_of(sol.y[0], sol.y[1], sol.t, H0, potential), dtype=float)
    return bool(np.all(np.isfinite(e2)) and np.all(e2 > 0.0))


def parameters_pinned_to_bounds(x, lower, upper, relative_tolerance: float = 1.0e-6) -> bool:
    """Return True when an optimiser coordinate is numerically pinned to a bound."""
    x = np.asarray(x, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    tolerance = relative_tolerance * np.maximum(1.0, upper - lower)
    return bool(np.any(x - lower <= tolerance) or np.any(upper - x <= tolerance))


def trajectory_cache_metadata(row, family: str, ngrid: int) -> dict:
    """Physical, numerical, and fitted inputs that uniquely identify a trajectory cache."""
    keys = ("H0", "A", "V0tilde", "eta", "n", "Lambda_over_rhoc0", "phi_a", "sigma_a", "alpha", "phi_b", "sigma_b", "phi_ini", "phip_ini")
    fitted = {key: float(f"{float(row[key]):.12g}") for key in keys if key in row and np.isfinite(row[key])}
    return {
        "cache_schema": 1,
        "family": family,
        "ngrid": int(ngrid),
        "z_star": Z_STAR,
        "T_CMB_K": T_CMB,
        "omega_gamma": OMEGA_GAMMA,
        "N_eff": N_EFF,
        "N_ur": N_UR,
        "T_ncdm_over_Tgamma": T_NCDM_OVER_TGAMMA,
        "mnu_eV": MNU_EV,
        "omega_nu": OMEGA_NU,
        "solver": {"method": "Radau", "rtol": 1e-9, "atol_field": 1e-11, "atol_distance": 1e-7, "max_step": 0.2},
        "fitted": fitted,
    }


def cache_metadata_json(row, family: str, ngrid: int) -> str:
    return json.dumps(trajectory_cache_metadata(row, family, ngrid), sort_keys=True, separators=(",", ":"))


def select_event(events, target: float | None = None):
    vals = np.asarray(events, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    if target is None:
        return float(vals[-1])
    return float(vals[np.argmin(np.abs(vals - target))])


def summarize(sol, H0: float, potential: Potential, zdagger_target: float | None = None):
    phi0, phip0, DM = sol.y[:, -1]
    E20 = e2_of(phi0, phip0, 0.0, H0, potential)
    E0 = np.sqrt(E20)
    cb0, rad0, nu0, _ = background_terms(0.0, H0)
    kinetic0 = XI * E20 * phip0**2 / 6.0
    closure = cb0 + rad0 + nu0 + potential.value(phi0) + kinetic0 - E20
    zt = select_event(sol.t_events[0])
    zdag = select_event(sol.t_events[1], zdagger_target)
    return {
        "E0": E0, "DM": DM, "O_F": closure,
        "phi0": phi0, "phip0": phip0,
        "zt": zt, "zdagger": zdag,
        "Delta_z": zt - zdag if np.isfinite(zt) and np.isfinite(zdag) else np.nan,
    }


def convergence_deltas(H0: float, potential: Potential, phi_ini: float, reference: dict):
    tighter = solve_trajectory(H0, potential, phi_ini, rtol=3e-10,
                               atol_field=3e-12, atol_distance=3e-8, max_step=0.1)
    test = summarize(tighter, H0, potential, reference.get("zdagger_target"))
    return {
        "conv_dE0": test["E0"] - reference["E0"],
        "conv_dDM_Mpc": test["DM"] - reference["DM"],
        "conv_dzt": test["zt"] - reference["zt"],
        "conv_dzdagger": test["zdagger"] - reference["zdagger"],
    }


def lcdm_reference(H0: float = 67.32):
    h2 = (H0 / 100.0) ** 2
    omega_de = h2 - OMEGA_CB - OMEGA_NU - OMEGA_GAMMA - OMEGA_NU_MASSLESS

    def inv_hubble(z):
        cb, rad, nu, _ = background_terms(z, H0)
        E = np.sqrt(cb + rad + nu + omega_de / h2)
        return C_KM_S / (H0 * E)

    DM, err = quad(inv_hubble, 0.0, Z_STAR, epsabs=1e-7, epsrel=1e-10, limit=500)
    return {"H0": H0, "E0": 1.0, "DM": DM, "quad_error_Mpc": err,
            "DM_minus_cal": DM - D_M_CAL}


def version_metadata():
    import matplotlib
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "executable": sys.executable,
    }
