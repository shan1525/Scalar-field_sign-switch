# Scalar-field sign-switch numerical release

This repository contains the numerical inputs, reconstruction products, nested-sampling summaries, dynamical trajectories, validation scripts, and publication figures associated with the scalar-field sign-switch analysis.

## Directory structure

- `recon_new1/`: final background and scalar-field reconstruction products for the ECDM, corrected SSCDM, and Ladder histories.
- `runs_new_corrected/sign_switch/ecdm_new/`: final ECDM potential-space fits, posterior samples, evidence summaries, corner plots, and posterior-envelope figures.
- `runs_new_corrected/sign_switch/sscdm/`: final SSCDM potential-space fits on the retained $\widetilde\phi_{\max}=0.23$ fitting interval.
- `runs_new_corrected/sign_switch/sscdm_new/`: SSCDM consistency check using the extended $\widetilde\phi_{\max}\simeq0.237$ interval.
- `dynamics_final_neff3046/`: photon-corrected sigmoid--Gaussian and offset-axion dynamical scans, diagnostics, cached trajectories, and figures using the adopted CLASS-like $N_{\rm eff}=3.046$ convention.
- `CLASS_consistency/`: CLASS v3.3.4 input, background-comparison material, and the independent same-input consistency test.
- `Phase_space/`: pressure--density phase-space regeneration code and associated figure products.

## Authoritative numerical products

For each potential-space fit, the compact posterior files, `posterior.csv`, NumPy posterior arrays, `info/results.json`, parameter summaries, model-comparison tables, and exported figures are retained. Large redundant UltraNest chain streams and internal HDF5 point files are excluded from Git because the retained products contain the quantities used in the manuscript and the omitted files can be regenerated from the archived configuration.

The `ecdm_old` directory is superseded and intentionally excluded. The `sscdm_new` directory is retained because it documents the consistency check using the complete SSCDM field interval near $\widetilde\phi_{\max}=0.237$.

## Interpretation

The potential-space evidence values are coordinate-fixed conditional scores for synthetic reconstruction experiments. They compare representational efficiency under the stated field coordinate, mock realization, weighting rule, grid, and prior prescription; they are not observational Bayes factors or posterior odds between physical cosmological theories.

The forward sigmoid--Gaussian and offset-axion calculations use the photon normalization associated with $T_{\rm CMB}=2.7255\,\mathrm{K}$ and the CLASS-like one-massive-neutrino mapping $(T_{\rm ncdm}/T_\gamma,N_{\rm ur})=(0.71611,2.0328)$ with $m_\nu=0.06\,\mathrm{eV}$.

## Reproducibility notes

The stored trajectory metadata identify the fitted parameters, solver configuration, photon and neutrino conventions, integration grid, and source family. Figure audit files record the source trajectory mapping and the field interval used for potential-panel shading. Software-version and calibration JSON files accompany the corresponding analyses.
