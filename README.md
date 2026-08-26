# Numerical release: scalar-field reconstruction and potential comparison

This repository contains the numerical products accompanying our analysis of phenomenological dark-energy histories and their effective scalar-field reconstructions. It includes the ECDM and corrected SSCDM reconstruction tables, synthetic potential-space data sets, UltraNest outputs, posterior chains, parameter contours, pointwise posterior-envelope figures, and the forward sigmoid--Gaussian and offset-axion dynamical products used in the manuscript.

## Scope and interpretation

The potential-space calculations compare five closed-form potential families on fixed reconstructed field intervals:

- sigmoid--Gaussian feature;
- shifted-\(\tanh\);
- generalised axion-like;
- regularised inverse-quadratic;
- Gaussian feature.

These are coordinate-fixed conditional marginal-likelihood comparisons within synthetic experiments. They depend on the adopted field origin and orientation, field grid, mock realisation, weighting rule, and prior prescription. The reported evidence values are not observational Bayes factors, posterior odds between cosmological theories, or coordinate-invariant statements about scalar-field microphysics. Absolute evidence values must not be compared between the ECDM and SSCDM targets.

## Recommended repository layout

```text
.
├── reconstruction/
│   ├── ecdm/
│   └── sscdm/
├── potential_space/
│   ├── ecdm/
│   │   ├── SigmoidBump/
│   │   ├── ShiftedTanh/
│   │   ├── AxionLikeOffset/
│   │   ├── InvertedQuadratic/
│   │   └── GaussianBump/
│   └── sscdm/
│       ├── SigmoidBump/
│       ├── ShiftedTanh/
│       ├── AxionLikeOffset/
│       ├── InvertedQuadratic/
│       └── GaussianBump/
├── dynamics/
│   ├── sigmoid_bump/
│   └── axion/
├── figures/
├── tables/
├── class_validation/
└── software_versions.json
```

The precise contents may be organised differently, but each archived fit should retain its configuration, `results.json`, posterior samples or chains, derived summary tables, and associated figures in the same model directory.

## Reconstruction data

The reconstruction CSV files contain the dimensionless field and potential, conventionally labelled

```text
phi_over_Mpl
V_over_rhoc0
```

or by equivalent names documented in the file header or generation script. The scalar-field origin is fixed at the high-redshift endpoint, and the orientation follows the positive rolling-sign convention used by the reconstruction. The corrected SSCDM benchmark uses \((z_\dagger,\Delta x)=(1.8,0.4)\) with boundaries symmetric in \(x=\ln(1+z)\). The complete unique SSCDM excursion is \(0\leq\widetilde\phi\leq0.23744185\). Any analysis using a shorter retained interval must identify the applied cutoff explicitly.

The ECDM potential-space fits use the verified pre-mock target scale

\[
\widetilde V_{\rm sc}^{\rm ECDM}=0.86922.
\]

The photon-normalisation audit changes the reconstructed inputs only at order \(10^{-6}\), well below the adopted mock uncertainties and sampler errors.

## Synthetic potential-space experiment

The archived potential-space runs use a fixed synthetic mock generated on a uniform field grid. The mock values and uncertainties should be taken directly from the archived CSV or saved configuration rather than regenerated implicitly. In the documented noisy branch, the mock prescription is

\[
V_i^{\rm mock}=V_i^{\rm tar}+\epsilon_i,\qquad
\epsilon_i\sim\mathcal N(0,\sigma_i^2),
\]

with

\[
\sigma_i^2=(\sigma_{\rm rel}|V_i^{\rm tar}|)^2+\sigma_{\rm abs}^2,
\]

using the run-specific values and random seed stored with each experiment. Priors are target dependent and use scales fixed from the noiseless target before the mock is drawn.

## Posterior summaries and curve envelopes

Parameter summaries are weighted posterior medians with weighted equal-tail 16th and 84th percentiles unless a model-specific file states otherwise. Curve panels are constructed by applying the nested-sampling weights once to obtain equal-weight posterior draws and then evaluating the potential on the displayed field grid. The pointwise median and equal-tail intervals are calculated from those equal-weight curves. The 68% and 95% regions are pointwise posterior envelopes, not highest-density regions and not observational posterior-predictive bands.

Figure legends use the terms **synthetic mock** and **pointwise posterior envelope**. The number of curve draws and the plotting RNG seed must be recorded in the corresponding configuration or metadata file.

## Marginal-likelihood results

The final coordinate-fixed conditional scores are:

### ECDM

| Potential family | \(\log\mathcal Z\) | Numerical error | \(\Delta\log\mathcal Z\) | \(N_{\rm like}\) |
|---|---:|---:|---:|---:|
| Sigmoid--Gaussian feature | 107.718 | 0.1404 | 0 | 53,389,099 |
| Shifted-\(\tanh\) | 102.964 | 0.0520 | \(-4.754\pm0.150\) | 2,518,550 |
| Generalised axion-like | 89.467 | 0.0883 | \(-18.250\pm0.166\) | 655,144 |
| Regularised inverse-quadratic | 49.675 | 0.0843 | \(-58.043\pm0.164\) | 3,672,431 |
| Gaussian feature | -431.535 | 0.2181 | \(-539.253\pm0.259\) | 40,138 |

### SSCDM

| Potential family | \(\log\mathcal Z\) | Numerical error | \(\Delta\log\mathcal Z\) | \(N_{\rm like}\) |
|---|---:|---:|---:|---:|
| Generalised axion-like | 72.700 | 0.0494 | 0 | 785,597 |
| Sigmoid--Gaussian feature | 66.721 | 0.1060 | \(-5.979\pm0.117\) | 7,773,842 |
| Regularised inverse-quadratic | 49.309 | 0.0953 | \(-23.391\pm0.107\) | 3,697,800 |
| Shifted-\(\tanh\) | -3.160 | 0.0857 | \(-75.860\pm0.099\) | 2,611,495 |
| Gaussian feature | -365.831 | 0.1625 | \(-438.531\pm0.170\) | 45,693 |

For each nonzero difference, the quoted uncertainty is the quadrature combination of the two sampler-reported numerical errors. It does not include sensitivity to the synthetic design, field convention, or prior ranges.

## Physical qualification of the axion-like fits

For both targets, the inferred zero-phase exponent of the best-fitting axion-like reconstruction lies below \(1/2\). Consequently, \(V_{,\phi}\) diverges at the included periodic-minimum endpoint \(\widetilde\phi=0\). The corresponding evidence scores assess potential values over the sampled intervals; these fitted members do not provide differentiable Klein--Gordon completions on the closed intervals. The regular \(n=1\) offset-axion solutions in the forward dynamical examples are distinct smooth members of the broader family.

## Forward dynamical calculations

The dynamical release contains the photon-corrected sigmoid--Gaussian and offset-axion shooting outputs, including diagnostics CSV files, calibration metadata, cached trajectories, zero crossings, tolerance-refinement checks, and figures. The calculations use the documented CLASS-like one-massive-neutrino convention

\[
(T_{\rm ncdm}/T_\gamma,N_{\rm ur})=(0.71611,2.0328),\qquad
m_\nu=0.06\,{\rm eV},\qquad
\omega_{\nu0}=0.06/93.14,
\]

and the photon density associated with \(T_{\rm CMB}=2.7255\,\mathrm K\). The ODE state is \((\varphi,{\rm d}\varphi/{\rm d}z,\chi)\); \(E^2\) is imposed algebraically rather than evolved independently. The comoving-distance state is initialised with \(\chi(z_*)=0\) and integrated from \(z_*=1090\) to the present.

The CLASS v3.3.4 same-input validation gives

\[
\max_{0\leq z\leq1090}|\Delta E/E|=1.51426\times10^{-7},
\]

\[
D_{\rm M}^{\rm CLASS}(1090)=13869.6389789848\,\mathrm{Mpc},\qquad
D_{\rm M}^{\rm code}(1090)=13869.6387503307\,\mathrm{Mpc}.
\]

The corresponding distance difference is \(-2.28654\times10^{-4}\,\mathrm{Mpc}\).

## CLASS validation archive

The numerical release should include at least:

```text
class_planck_neff3046.ini
class_planck_neff3046_background.dat
consistency_lcdm.py
consistency_lcdm_same_photons.txt
```

These files provide the exact CLASS input, exported background, independent comparison calculation, and captured numerical output.

## Reproducibility metadata

Record the following in `software_versions.json` or an equivalent machine-readable file:

- Python version;
- NumPy, SciPy, pandas, and Matplotlib versions;
- UltraNest and ChainConsumer versions;
- operating system and architecture;
- mock-generation seed;
- curve-resampling seed and number of draws;
- sampler settings and stopping criterion;
- likelihood-call count for every fit;
- commit hash of the released code.

Package versions can be queried in the fitting environment with:

```bash
python -c "import importlib.metadata as m; print('UltraNest:', m.version('ultranest')); print('ChainConsumer:', m.version('chainconsumer'))"
```

## Reproducing the figures

Figures should be regenerated from the archived reconstruction, mock, posterior, and trajectory files rather than digitised from the manuscript. Use the plotting scripts and paths documented in the release manifest. PDF is preferred for the manuscript; PNG copies may be included for convenient preview. Curve identities must remain distinguishable by line style or marker as well as colour, and all text must remain legible at final journal width.

## Data integrity

Do not modify archived chain or results files in place. If a product is regenerated, place it in a new versioned directory and record the source configuration, code commit, and creation date. Checksums are recommended for the final release manifest.

## Citation

If you use these data or scripts, please cite the accompanying paper and this numerical release. Bibliographic and archival identifiers should be added here when assigned.

## Contact

Questions about the reconstruction, potential-space fits, or dynamical calculations should be directed to the corresponding authors listed in the accompanying manuscript.
