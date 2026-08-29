"""Dynamical closure test — reproduces Fig. 18 and every number quoted in Subsec. VI C ("A dynamical closure test") of

  Adil, Akarsu, Bouhmadi-Lopez, Ibarra-Uriondo, Katirci & Vazquez,
  "Reconstructing Sign-Switching Dark-Energy Histories: ..."

Fully self-contained: no input files, no random numbers. Run

    python3 dynamical_closure.py

and it writes  dynamical_closure.pdf  (the manuscript figure, TrueType fonts) and prints the quoted diagnostics:

ECDM  <- sigmoid-Gaussian median : max|dOmega| = 0.158 at z ~ 2.2,
           dOmega(0) = +0.087 (= plateau offset L - Omega_de0),
           z_dagger = 1.913 (target 1.800), E(0) = 1.043
SSCDM <- sigmoid-Gaussian median : max|dOmega| = 1.72, z_dagger = 3.14
           axion refits (chi^2, Eq.-(50) weights): unconstrained n = 0.43,
           chi2/N = 0.033;  n >= 1/2 pinned at bound, chi2/N = 0.086,
           z_dagger = 3.33, one-sided cusp force A/(sqrt2 eta) ~ 24;
           n = 1 fixed, chi2/N = 3.60, phi_in = 1e-8 -> z_dagger = 0.64;
           Delta chi2 (n>=1/2 vs free) ~ 5.3, i.e. Delta lnL ~ 2.7.

WHAT IT DOES (all conventions as in the manuscript)
  1. Builds the ECDM and compact SSCDM target histories (z_dagger = 1.8; erf rate eta = 5; smooth-step half-width Delta x = 0.2 per side) on the LOW-REDSHIFT background of the reconstruction arrays: massless-nu radiation, Omega_m0 = 0.31, h = 0.7 (Subsec. IV A).
  2. Reconstructs the phantom-branch field and on-shell potential on 0 <= z <= 5 [xi = -1;  (dphi/dz)^2 = -Omega'/((1+z) E^2), V = Omega - (1+z) Omega'/6 in rho_c0 / M_Pl units].
  3. Evolves the Sec.-V marginal-median sigmoid-Gaussian potentials forward through the Klein-Gordon-Friedmann system (pure initial-value problem: nothing is shot for, E(0) = 1 not imposed).
     Initial data at z = 5 from the reconstruction:
       ECDM  : (phi, phi') = (0, -Q(5)),  Q(5) ~ 2.9e-5;
       SSCDM : exactly frozen (0, 0).
  4. Refits the axion family  V = A |1 - cos(phi/eta)|^n + V0  to the noiseless SSCDM on-shell target on the retained interval phi <= 0.23, with the synthetic weights of Eq. (50)
     [sigma_i^2 = (0.10 |V_i|)^2 + 0.05^2], under (a) n free,
     (b) n >= 1/2, (c) n = 1 fixed; evolves (b) and (c) from the seeded
     state (1e-8, 0) — the seed selects the outgoing branch at the
     n = 1/2 cusp (one-sided |V_phi| = A/(sqrt2 eta), bounded away from
     zero, so that departure is seed- and regularisation-independent)
     and is the departure clock for the regular n = 1 member.
  5. Draws the two-column figure (house style of Figs. 15-17: z = 0 at
     the origin increasing rightward, framed panels, top legend strip).

These are function-space chi^2 fits under the synthetic weights — the evidence comparison of Table II is NOT repeated here.

"""
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.special import erf

# ------------------------------------------------------------------
# 1. background + target histories (manuscript conventions)
# ------------------------------------------------------------------
h, Om = 0.7, 0.31
Or = 2.47297928e-5*(1 + 7/8*(4/11)**(4/3)*3.046)/h**2   # massless-nu radiation
Ode0 = 1 - Om - Or

zg = np.linspace(100, 0, 30000)          # dense grid, z decreasing
xg = -np.log(1 + zg)                     # x = ln a
zd = 1.8                                 # prescribed density zero
xd = -np.log(1 + zd)

# ECDM: Omega_de(x) = Ode0 * erf(eta (x - xd)) / erf(-eta xd), eta = 5
D = erf(-5.0*xd)
O_ecdm = Ode0*erf(5.0*(xg - xd))/D

# compact SSCDM: C^4 smooth step of total width Delta x = 0.4 centred at xd
dxw = 0.4
xi_, xf_ = xd - dxw/2, xd + dxw/2
tt = (xf_ - xg)/(xf_ - xi_)
ff = 126*tt**5 - 420*tt**6 + 540*tt**7 - 315*tt**8 + 70*tt**9
O_sscdm = np.where(xg <= xi_, -Ode0, np.where(xg >= xf_, Ode0, Ode0*(1 - 2*ff)))

# ------------------------------------------------------------------
# 2. phantom-branch reconstruction on 0 <= z <= 5   (phi(z=5) = 0)
# ------------------------------------------------------------------
def reconstruct(Ow):
    E2 = Or*(1 + zg)**4 + Om*(1 + zg)**3 + Ow
    dO = np.gradient(Ow, zg, edge_order=2)
    V = Ow - (1 + zg)/6*dO                       # on-shell potential
    Q = np.sqrt(np.clip(-dO/((1 + zg)*E2), 0, None))   # |dphi/dz|, xi = -1
    m = zg <= 5.0
    zz, QQ = zg[m][::-1], Q[m][::-1]
    c = cumulative_trapezoid(QQ, zz, initial=0.0)
    phi = (c[-1] - c)[::-1]
    return zg[m], phi, V[m], Ow[m], Q[m]

zE, phiE, VE, OE, QE = reconstruct(O_ecdm)
zS, phiS, VS, OS, QS = reconstruct(O_sscdm)

# ------------------------------------------------------------------
# 3. potential families
# ------------------------------------------------------------------
def sigmoid_gauss(p):
    """V = L [ tanh((phi-pa)/sa) + al exp(-(phi-pb)^2/sb^2) ]  (Sec. V)."""
    L, pa, sa, al, pb, sb = p
    V  = lambda f: L*(np.tanh((f - pa)/sa) + al*np.exp(-(f - pb)**2/sb**2))
    dV = lambda f: L*((1 - np.tanh((f - pa)/sa)**2)/sa
                      - 2*al*(f - pb)/sb**2*np.exp(-(f - pb)**2/sb**2))
    return V, dV

def axion(p):
    """V = A |1 - cos(phi/eta)|^n + V0  (generalised axion-like family)."""
    A, eta, n, V0 = p
    V  = lambda f: A*np.abs(1 - np.cos(f/eta))**n + V0
    dV = lambda f: A*n/eta*np.abs(1 - np.cos(f/eta))**(n - 1)*np.sin(f/eta)
    return V, dV

# marginal-median parameter points of Sec. V (representative members,
# not posterior samples): (L, pa, sa, al, pb, sb)
MED_SG_ECDM  = (0.778, 0.0763, 0.0651, 0.107, 0.178,  0.055)   # Sec. V A
MED_SG_SSCDM = (0.852, 0.0318, 0.0358, 0.77,  0.1385, 0.0676)  # Sec. V B

# ------------------------------------------------------------------
# 4. forward Klein-Gordon-Friedmann integration (pure IVP, z: 5 -> 0)
#    phantom branch: Omega_K = -(1+z)^2 E^2 phi'^2 / 6  <=  0
# ------------------------------------------------------------------
def forward(Vf, dVf, phi0, dphi0, z0=5.0):
    def rhs(z, y):
        f, df = y
        B = -(1 + z)**2*df**2/6
        E2 = (Or*(1 + z)**4 + Om*(1 + z)**3 + Vf(f))/(1 - B)
        OK = -(1 + z)**2*E2*df**2/6
        EEp = (3*Om*(1 + z)**3 + 4*Or*(1 + z)**4 + 6*OK)/(2*(1 + z))
        ddf = -((1 + z)**2*EEp*df - 2*(1 + z)*E2*df - 3*dVf(f))/((1 + z)**2*E2)
        return [df, ddf]
    sol = solve_ivp(rhs, [z0, 0.0], [phi0, dphi0], method='Radau',
                    rtol=1e-9, atol=1e-11, dense_output=True, max_step=0.05)
    zs = np.linspace(z0, 0, 2001)
    f, df = sol.sol(zs)
    B = -(1 + zs)**2*df**2/6
    E2 = (Or*(1 + zs)**4 + Om*(1 + zs)**3 + Vf(f))/(1 - B)
    Ophi = Vf(f) - (1 + zs)**2*E2*df**2/6
    return zs, Ophi, f, np.sqrt(E2)

def compare(tag, zs, Ophi, ztar, Otar, E0):
    Ot = np.interp(zs[::-1], ztar[::-1], Otar[::-1])[::-1]
    d = Ophi - Ot
    i = np.argmax(np.abs(d))
    zx = zs[np.argmin(np.abs(Ophi))] if (Ophi.min() < 0 < Ophi.max()) else np.nan
    print(f"{tag}: max|dOmega| = {np.abs(d).max():.4f} at z = {zs[i]:.2f}; "
          f"dOmega(z=0) = {d[-1]:+.4f}; z_dagger = {zx:.3f} (target 1.800); "
          f"E(0) = {E0:.4f}")

print("=== A. forward evolution of the Sec.-V marginal-median potentials ===")
V1, dV1 = sigmoid_gauss(MED_SG_ECDM)
zs1, Op1, f1, E1 = forward(V1, dV1, 0.0, -QE[0])   # ECDM: (0, -Q(5))
compare("ECDM  <- sigmoid-Gaussian median", zs1, Op1, zE, OE, E1[-1])
print(f"   [Q(5) = {QE[0]:.3e};  plateau offset L - Omega_de0 = "
      f"{MED_SG_ECDM[0] - Ode0:+.4f}  (cf. dOmega(z=0) above)]")
V2, dV2 = sigmoid_gauss(MED_SG_SSCDM)
zs2, Op2, f2, E2_ = forward(V2, dV2, 0.0, 0.0)     # SSCDM: exactly frozen
compare("SSCDM <- sigmoid-Gaussian median", zs2, Op2, zS, OS, E2_[-1])

print("\n=== B. constrained axion refits to the SSCDM target (Eq.-(50) weights) ===")
mfit = phiS <= 0.23                                 # retained field interval
order = np.argsort(phiS[mfit])
pgrid = np.linspace(0, 0.23, 100)
Vtar = np.interp(pgrid, phiS[mfit][order], VS[mfit][order])
sig = np.sqrt((0.10*np.abs(Vtar))**2 + 0.05**2)     # Eq. (50)

def resid(p, nfix=None):
    q = list(p)
    if nfix is not None:
        q = [q[0], q[1], nfix, q[2]]
    Vf, _ = axion(q)
    return (Vf(pgrid) - Vtar)/sig

r0 = least_squares(resid, [1.726, 0.04697, 0.428, -0.811],       # start: paper median
                   bounds=([1e-3, 1e-3, 0.05, -10], [10, 1.0, 2.0, 10]))
r1 = least_squares(resid, [1.726, 0.04697, 0.60, -0.811],        # n >= 1/2
                   bounds=([1e-3, 1e-3, 0.5, -10], [10, 1.0, 2.0, 10]))
r2 = least_squares(lambda p: resid(p, nfix=1.0), [1.0, 0.06, -0.85],  # n = 1
                   bounds=([1e-3, 1e-3, -10], [10, 1.0, 10]))
c0, c1, c2 = [2*r.cost for r in (r0, r1, r2)]
print(f"n free     : (A,eta,n,V0) = ({r0.x[0]:.4f}, {r0.x[1]:.5f}, "
      f"{r0.x[2]:.4f}, {r0.x[3]:.4f});  chi2/N = {c0/100:.4f}")
print(f"n >= 1/2   : (A,eta,n,V0) = ({r1.x[0]:.4f}, {r1.x[1]:.5f}, "
      f"{r1.x[2]:.4f}, {r1.x[3]:.4f});  chi2/N = {c1/100:.4f}")
print(f"n  = 1     : (A,eta,V0)   = ({r2.x[0]:.4f}, {r2.x[1]:.5f}, "
      f"{r2.x[2]:.4f});            chi2/N = {c2/100:.4f}")
print(f"Delta chi2 (n>=1/2 vs free) = {c1 - c0:.2f}  "
      f"->  Delta lnL = {(c1 - c0)/2:.2f}")
print(f"one-sided cusp force at the n=1/2 bound: A/(sqrt2 eta) = "
      f"{r1.x[0]/(np.sqrt(2)*r1.x[1]):.1f}")

q1 = [r1.x[0], r1.x[1], max(r1.x[2], 0.5), r1.x[3]]
V3, dV3 = axion(q1)
zs3, Op3, f3, E3 = forward(V3, dV3, 1e-8, 0.0)      # seed selects the branch
compare("SSCDM <- axion refit, n = 1/2 (bound)", zs3, Op3, zS, OS, E3[-1])
q2 = [r2.x[0], r2.x[1], 1.0, r2.x[2]]
V4, dV4 = axion(q2)
zs4, Op4, f4, E4 = forward(V4, dV4, 1e-8, 0.0)      # seed is the departure clock
compare("SSCDM <- axion refit, n = 1 fixed    ", zs4, Op4, zS, OS, E4[-1])

# ------------------------------------------------------------------
# 5. the figure 
# ------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'font.size': 8, 'axes.linewidth': 1.0})
TEAL, ORANGE, RED = '#1B9E77', '#E66101', '#B22222'

fig, axs = plt.subplots(2, 2, figsize=(7.0, 3.4), sharex='col',
                        gridspec_kw=dict(height_ratios=[2.6, 1], hspace=0.08,
                                         wspace=0.24))
for j, (tag, ztar, Otar, runs) in enumerate([
    ('ECDM target', zE, OE,
     [(zs1, Op1, TEAL, '-', 'sigmoid–Gaussian median')]),
    ('SSCDM target', zS, OS,
     [(zs2, Op2, TEAL, '-', 'sigmoid–Gaussian median'),
      (zs3, Op3, ORANGE, '-', r'axion refit, $n=1/2$ (bound)'),
      (zs4, Op4, RED, (0, (4, 2)), r'axion refit, $n=1$, $\varphi_{\rm in}=10^{-8}$')])]):
    a, b = axs[0, j], axs[1, j]
    a.plot(ztar, Otar, color='black', lw=1.8, label='prescribed history', zorder=5)
    for zs, Op, c, ls, lab in runs:
        a.plot(zs, Op, color=c, lw=1.6, ls=ls, label=lab, zorder=4)
        Ot = np.interp(zs[::-1], ztar[::-1], Otar[::-1])[::-1]
        b.plot(zs, Op - Ot, color=c, lw=1.3, ls=ls)
    a.axhline(0, color='0.3', lw=0.6, ls=(0, (1, 1.6)), zorder=1)
    b.axhline(0, color='0.3', lw=0.6, ls=(0, (1, 1.6)), zorder=1)
    a.set_title(tag, fontsize=9)
    a.set_ylabel(r'$\widetilde{\Omega}_\phi(z)$' if j == 0 else '')
    b.set_ylabel(r'$\Delta\widetilde{\Omega}_\phi$' if j == 0 else '')
    b.set_xlabel(r'$z$')
    a.set_xlim(0, 5); a.set_ylim(-1.05, 1.65 if j else 0.95)
axs[1, 0].set_ylim(-0.19, 0.19); axs[1, 1].set_ylim(-1.1, 2.0)
hnd, lbl = axs[0, 1].get_legend_handles_labels()
fig.legend(hnd, lbl, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.955),
           fontsize=6.4, frameon=True, edgecolor='0.6', framealpha=1.0,
           borderpad=0.45, handlelength=1.9, columnspacing=1.3,
           handletextpad=0.6)
fig.subplots_adjust(top=0.86)
fig.savefig('dynamical_closure.pdf', bbox_inches='tight')
print("\nsaved: dynamical_closure.pdf")
