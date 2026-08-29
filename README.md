# Scalar-field sign-switch numerical release

This repository contains the numerical inputs, reconstruction products, nested-sampling summaries, dynamical trajectories, validation scripts, and publication figures associated with the scalar-field sign-switch analysis.

## Directory structure

- `recon_new1/`: final background and scalar-field reconstruction products for the ECDM, corrected SSCDM, and Ladder histories.
- `runs_new_corrected/sign_switch/ecdm_new/`: final ECDM potential-space fits, posterior samples, evidence summaries, corner plots, and posterior-envelope figures.
- `runs_new_corrected/sign_switch/sscdm/`: final SSCDM potential-space fits on the retained $\widetilde\phi_{\max}=0.23$ fitting interval.
- `runs_new_corrected/sign_switch/sscdm_new/`: SSCDM consistency check using the extended $\widetilde\phi_{\max}\simeq0.237$ interval.
- `dynamics_final_neff3046/`: photon-corrected sigmoid--Gaussian and offset-axion dynamical scans, diagnostics, cached trajectories, and figures using the adopted CLASS-like $N_{\rm eff}=3.046$ convention.
- `dynamical_closure/`: low-redshift dynamical-closure test connecting the reconstructed ECDM and SSCDM histories to forward evolution of the fitted sigmoid--Gaussian and constrained axion-like potentials, including the analysis script, numerical diagnostics, and publication figure.
- `CLASS_consistency/`: CLASS v3.3.4 input, background-comparison material, and the independent same-input consistency test.
- `Phase_space/`: pressure--density phase-space regeneration code and associated figure products.

## Authoritative numerical products

For each potential-space fit, the compact posterior files `posterior.csv`, NumPy posterior arrays, `info/results.json`, parameter summaries, model-comparison tables, and exported figures are retained. Large redundant UltraNest chain streams and internal HDF5 point files are excluded from Git because the retained products contain the quantities used in the manuscript and the omitted files can be regenerated from the archived configuration.

The `ecdm_old` directory is superseded and intentionally excluded. The `sscdm_new` directory is retained because it documents the consistency check using the complete SSCDM field interval near $\widetilde\phi_{\max}=0.237$.

The `dynamical_closure` calculation is a pure initial-value consistency test over $0\leq z\leq5$. It uses the same low-redshift background convention as the reconstruction arrays and does not repeat the $z_\ast=1090$ shooting analyses. Its constrained axion refits are noiseless function-space diagnostics rather than new nested-sampling or evidence calculations.

## Interpretation

The potential-space evidence values are coordinate-fixed conditional scores for synthetic reconstruction experiments. They compare representational efficiency under the stated field coordinate, mock realisation, weighting rule, grid, and prior prescription; they are not observational Bayes factors or posterior odds between physical cosmological theories.

The dynamical-closure test separately asks whether selected fitted potentials reproduce the prescribed histories when evolved from stated initial data. Approximate ECDM closure and failure of the tested axion-like and sigmoid--Gaussian members to reproduce exact compact SSCDM should not be interpreted as observational model selection or as a general exclusion of either potential family.

The forward sigmoid--Gaussian and offset-axion calculations use the photon normalisation associated with $T_{\rm CMB}=2.7255\,\mathrm{K}$ and the CLASS-like one-massive-neutrino mapping $(T_{\rm ncdm}/T_\gamma,N_{\rm ur})=(0.71611,2.0328)$ with $m_\nu=0.06\,\mathrm{eV}$.

## Reproducibility notes

The stored trajectory metadata identify the fitted parameters, solver configuration, photon and neutrino conventions, integration grid, and source family. Figure audit files record the source-trajectory mapping and the field interval used for potential-panel shading. Software-version and calibration JSON files accompany the corresponding analyses.

The dynamical-closure script records the adopted initial conditions, potential vectors, fitting weights, solver tolerances, density-crossing diagnostics, endpoint normalisation, and pointwise deviations from the prescribed histories. The closure outputs should be regenerated whenever the authoritative reconstruction arrays or posterior-summary convention changes.