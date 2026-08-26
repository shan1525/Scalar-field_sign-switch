#!/usr/bin/env python3

import numpy as np
from scipy.integrate import cumulative_trapezoid
from classy import Class

import dynamics_calibration as dc

H0 = 67.32
z_max = 1090.0
c_km_s = 299792.458

params = {
    "H0": H0,
    "T_cmb": 2.7255,
    "omega_b": 0.022383,
    "omega_cdm": 0.12011,
    "Omega_k": 0.0,

    # CLASS-like Planck massive-neutrino convention.
    "N_ncdm": 1,
    "m_ncdm": 0.06,
    "omega_ncdm": 0.06 / 93.14,
    "T_ncdm": 0.71611,
    "deg_ncdm": 1.0,
    "N_ur": 2.0328,

    # Lambda is inferred from the flat background budget.
    "Omega_fld": 0.0,
    "Omega_scf": 0.0,

    "output": "",
    "background_verbose": 0,
}

cosmo = Class()
cosmo.set(params)
cosmo.compute()

h2 = (H0 / 100.0)**2
omega_ncdm_requested = 0.06 / 93.14
omega_ncdm_class = float(cosmo.Omega_nu) * h2
omega_m_class = float(cosmo.Omega_m) * h2
h = H0 / 100.0
omega_gamma_class = cosmo.Omega_g() * h**2


print("CLASS input mapping")
print(f"T_ncdm/T_gamma requested = {params['T_ncdm']:.8f}")
print(f"N_ur requested           = {params['N_ur']:.8f}")
print(f"omega_ncdm requested     = {omega_ncdm_requested:.12e}")
print(f"omega_ncdm returned      = {omega_ncdm_class:.12e}")
print(f"omega_m returned         = {omega_m_class:.12e}")
print(f"omega_gamma_CLASS = {omega_gamma_class:.16e}")

if not np.isclose(
    omega_ncdm_class,
    omega_ncdm_requested,
    rtol=1e-8,
    atol=1e-12,
):
    raise RuntimeError(
        "CLASS massive-neutrino density does not match the requested "
        "omega_ncdm normalization."
    )


bg = cosmo.get_background()

# CLASS background arrays need not be returned in increasing-z order.
order = np.argsort(bg["z"])
z_class = np.asarray(bg["z"])[order]
H_class_mpc = np.asarray(bg["H [1/Mpc]"])[order]
chi_class = np.asarray(bg["comov. dist."])[order]

# Common comparison grid: dense near z=0 and logarithmic at high z.
z = np.unique(np.concatenate([
    np.linspace(0.0, 10.0, 5001),
    np.expm1(np.linspace(np.log(11.0), np.log1p(z_max), 5000)),
]))

H_class = np.interp(z, z_class, H_class_mpc)
E_class = H_class / np.interp(0.0, z_class, H_class_mpc)

# Your independent Lambda-CDM calculation.
h2 = (H0 / 100.0)**2
omega_de = (
    h2
    - dc.OMEGA_CB
    - dc.OMEGA_NU
    - dc.OMEGA_GAMMA
    - dc.OMEGA_NU_MASSLESS
)

cb, rad, nu, _ = dc.background_terms(z, H0)
E_ours = np.sqrt(cb + rad + nu + omega_de / h2)

relative_E = (E_ours - E_class) / E_class
imax = np.argmax(np.abs(relative_E))

# CLASS comoving distance.
DM_class = np.interp(z_max, z_class, chi_class)

# Distance obtained by integrating our E(z).
DM_integrand = c_km_s / (H0 * E_ours)
DM_ours_grid = cumulative_trapezoid(
    DM_integrand, z, initial=0.0
)
DM_ours = DM_ours_grid[-1]

# Also retain the higher-accuracy distance from the existing routine.
ours_reference = dc.lcdm_reference(H0)

print(f"CLASS D_M(1090)       = {DM_class:.10f} Mpc")
print(f"Our grid D_M(1090)    = {DM_ours:.10f} Mpc")
print(f"Our quad D_M(1090)    = {ours_reference['DM']:.10f} Mpc")
print(f"Delta D_M (quad-CLASS)= "
      f"{ours_reference['DM'] - DM_class:+.10e} Mpc")

print(f"max |Delta E/E|       = "
      f"{np.max(np.abs(relative_E)):.10e}")
print(f"location of maximum   = z={z[imax]:.10f}")
print(f"Delta E/E there       = {relative_E[imax]:+.10e}")

cosmo.struct_cleanup()
cosmo.empty()
