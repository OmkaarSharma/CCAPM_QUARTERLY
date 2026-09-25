#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sdf_quarterly_userlambda.py   ->   SDF_quarterly_lambda/

Phase-1 SDF pipeline, re-run QUARTERLY with the cell-specific lambda from
Lambda_Average_Values_For_Each_Region.csv (one lambda per STATE x REGION).

Settings agreed before implementation
  frequency      quarterly (CONSUMPTION_INCOME_Q.csv: TOTEXP_Q, REAL_TOT_INC_Q = 3-month sums)
  risk-free      10Y only:  RF gross = exp(RF_REAL_10Y_LOG)
  market         Sensex:    RW gross = 1 + SENSEX_REAL_Q        (same as the Weber estimation)
  lambda         your cell value, plus lambda = 0 as the baseline
  infeasible     if C - lambda*Y <= 0 in ANY quarter the cell is reported as infeasible and
                 no beta/gamma is estimated (your lambda is never changed)
  output         SDF stats (HJ, BM, CAPM), CRRA beta/gamma by GMM + L-BFGS-B, and the fitted
                 consumption SDF  m_hat = beta * (S_t+1 / S_t)^(-gamma),  S = C - lambda*Y

What lambda does and does not touch
  SDF-HJ, SDF-BM and SDF-CAPM are built from the two returns only, so they are IDENTICAL for
  every lambda.  lambda enters only through S = C - lambda*Y in the CCAPM estimation and in the
  fitted consumption SDF.

Methods (unchanged from Phase 1 except dt = 1 quarter)
  SDF-HJ    m = R c,  c = E[RR']^-1 1                         (eq 7)
  SDF-CAPM  m = a + b Rw,  E[m Rf] = E[m Rw] = 1              (eq 11)
  SDF-BM    m = exp(-(rf + mu*theta/2) - theta (r - rbar)),  theta = mu / sigma^2   (eq 9)
  CRRA      (I)  level:  m_t+1 - beta exp(-gamma dlnS_t+1)
            (II) log:    ln m_t+1 - ln beta + gamma dlnS_t+1   (quarters with m <= 0 dropped)
            instruments z_t = [1, dlnS_t, m_t]; two-step GMM, Newey-West weighting,
            L-BFGS-B multi-start + Nelder-Mead polish; beta in [0.5, 1.5], gamma in [-50, 50].
"""
import os, sys, math, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
warnings.filterwarnings("ignore")

# =============================================================== HOW TO RUN ===
#   pip install numpy pandas scipy matplotlib
#   python sdf_quarterly_userlambda.py
#       looks for the two input files next to this script, then in the folder you run it from.
#   or give the paths explicitly (Windows example):
#   python sdf_quarterly_userlambda.py --panel "C:\path\CONSUMPTION_INCOME_Q.csv"
#          --lam "C:\path\Lambda_Average_Values_For_Each_Region.csv" --out "C:\path\results"
#   Runtime: about 1-2 minutes.  Results go to --out (default: SDF_quarterly_lambda_results
#   next to this script).  A built-in verification runs at the end and prints PASS/FAIL.
# =============================================================================
import argparse
HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_NAME = "CONSUMPTION_INCOME_Q.csv"
LAM_NAME   = "Lambda_Average_Values_For_Each_Region.csv"

def find(name, given):
    if given:
        if not os.path.isfile(given): sys.exit("file not found: %s" % given)
        return given
    for d in (HERE, os.getcwd()):
        p = os.path.join(d, name)
        if os.path.isfile(p): return p
    sys.exit("could not find %s next to the script or in %s.\nPut it there or pass its path "
             "(--panel / --lam)." % (name, os.getcwd()))

ap = argparse.ArgumentParser(description="Quarterly SDFs + CRRA GMM with cell-specific lambda")
ap.add_argument("--panel", help="path to %s" % PANEL_NAME)
ap.add_argument("--lam",   help="path to %s" % LAM_NAME)
ap.add_argument("--out",   help="output folder", default=os.path.join(HERE, "SDF_quarterly_lambda_results"))
ARGS = ap.parse_args()
PANEL = find(PANEL_NAME, ARGS.panel)
LAMF  = find(LAM_NAME, ARGS.lam)
OUT   = ARGS.out
os.makedirs(OUT, exist_ok=True)

BETA_BOUNDS  = (0.50, 1.50)
GAMMA_BOUNDS = (-50.0, 50.0)
STARTS       = [(1.00, 0.0), (1.00, 2.0), (1.00, 10.0), (1.00, 30.0), (0.98, -5.0)]
MIN_OBS_LOG  = 24                     # log spec needs >= 24 usable quarters (Phase 1: 30 months)
NATIONAL     = ("India", "Comb")
REGION_ORDER = {"Comb": 0, "RURAL": 1, "URBAN": 2}
SDF_NAMES    = ["HJ", "BM", "CAPM"]

LOG = open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()

# ---------------------------------------------------------------- data -------
P = pd.read_csv(PANEL, encoding="utf-8-sig")
for c in ("STATE", "REGION", "QUARTER"): P[c] = P[c].astype(str).str.strip()
P["C"]  = P.TOTEXP_Q.astype(float)
P["Y"]  = P.REAL_TOT_INC_Q.astype(float)
P["RW"] = 1.0 + P.SENSEX_REAL_Q
P["RF"] = np.exp(P.RF_REAL_10Y_LOG)
P["QDATE"] = [pd.Period(q, freq="Q").end_time.normalize() for q in P.QUARTER]   # 2014Q1 -> 2014-03-31
log("inputs: panel = %s\n        lambda = %s\n        output = %s" % (PANEL, LAMF, OUT))
need = {"STATE", "REGION", "QUARTER", "TOTEXP_Q", "REAL_TOT_INC_Q", "SENSEX_REAL_Q", "RF_REAL_10Y_LOG"}
if need - set(P.columns): sys.exit("panel is missing columns: %s" % sorted(need - set(P.columns)))

L = pd.read_csv(LAMF, encoding="utf-8-sig")
for c in ("STATE", "REGION"): L[c] = L[c].astype(str).str.strip()
LAM = {(r.STATE, r.REGION): float(r.LAMBDA_AVG) for r in L.itertuples()}
cells = list(dict.fromkeys(zip(P.STATE, P.REGION)))
log("panel: %d rows, %d cells, %s .. %s" % (len(P), len(cells), P.QUARTER.min(), P.QUARTER.max()))
no_panel = [k for k in LAM if k not in set(cells)]
no_lam   = [k for k in cells if k not in LAM]
log("lambda rows with no panel cell (skipped): %s" % no_panel)
log("panel cells with no lambda: %s" % no_lam)
if no_lam: sys.exit("every panel cell needs a lambda")

# ---------------------------------------------------------------- SDFs -------
def build_sdfs(RW, RF):
    T = len(RW)
    R = np.column_stack([RW, RF])
    c = np.linalg.solve(R.T @ R / T, np.ones(2))
    m_hj = R @ c
    A = np.array([[RF.mean(), (RW * RF).mean()], [RW.mean(), (RW * RW).mean()]])
    a, b = np.linalg.solve(A, np.ones(2))
    m_capm = a + b * RW
    r, rf = RW - 1.0, RF - 1.0
    mu = (r - rf).mean(); sig2 = ((r - r.mean()) ** 2).mean(); th = mu / sig2
    m_bm = np.exp(-(rf + 0.5 * mu * th) - th * (r - r.mean()))
    return {"HJ": m_hj, "BM": m_bm, "CAPM": m_capm}

# ---------------------------------------------------------------- GMM --------
def nw_lag(T): return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
def nw_cov(U, lag):
    T = U.shape[0]; S = U.T @ U / T
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0); G = U[l:].T @ U[:-l] / T; S += w * (G + G.T)
    return S

def gmm(err, Z, starts, bounds):
    T, k = Z.shape; p = len(bounds)
    mom = lambda th: err(th)[:, None] * Z
    gbar = lambda th: mom(th).mean(0)
    def obj(th, W):
        g = gbar(th); v = T * (g @ W @ g)
        return v if np.isfinite(v) else 1e300
    def solve(W, st):
        best = None
        for s in st:
            r = minimize(obj, np.array(s, float), args=(W,), method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=2000, ftol=1e-12, gtol=1e-8))
            if best is None or r.fun < best.fun: best = r
        r2 = minimize(obj, best.x, args=(W,), method="Nelder-Mead",
                      options=dict(xatol=1e-9, fatol=1e-13, maxiter=4000))
        if r2.fun < best.fun - 1e-10:
            x = np.clip(r2.x, [b[0] for b in bounds], [b[1] for b in bounds]); best.x = x; best.fun = obj(x, W)
        f0 = best.fun; flat = True
        for j in range(p):
            if not (bounds[j][0] < best.x[j] < bounds[j][1]): continue
            h = 1e-6 * max(1.0, abs(best.x[j])); e = np.zeros(p); e[j] = h
            if min(obj(best.x + e, W), obj(best.x - e, W)) < f0 - 1e-9 * max(1.0, abs(f0)): flat = False
        best.success = bool(best.success or flat)
        return best
    r1 = solve(np.eye(k), starts)
    W2 = np.linalg.pinv(nw_cov(mom(r1.x), nw_lag(T)))
    r2 = solve(W2, [tuple(r1.x)] + starts)
    th = r2.x
    G = np.zeros((k, p)); h = 1e-6
    for j in range(p):
        e = np.zeros(p); e[j] = h; G[:, j] = (gbar(th + e) - gbar(th - e)) / (2 * h)
    V = np.linalg.pinv(G.T @ np.linalg.pinv(nw_cov(mom(th), nw_lag(T))) @ G) / T
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    df = k - p; J = float(r2.fun)
    at_bound = [not (b[0] + 1e-6 < v < b[1] - 1e-6) for v, b in zip(th, bounds)]
    return dict(theta=th, se=se, J=J, J_df=df, J_p=float(chi2.sf(J, df)) if df > 0 else np.nan,
                converged=bool(r2.success), T=T, nw_lag=nw_lag(T), at_bound=at_bound)

def estimate(m, dlnS):
    """m, dlnS on the return quarters.  error dated t+1, instruments dated t."""
    out = []
    m1, g1 = m[1:], dlnS[1:]
    Z = np.column_stack([np.ones(len(m1)), dlnS[:-1], m[:-1]])
    r = gmm(lambda th: m1 - th[0] * np.exp(-th[1] * g1), Z, STARTS, [BETA_BOUNDS, GAMMA_BOUNDS])
    b, g = r["theta"]
    out.append(dict(SPEC="I_level", BETA=b, SE_BETA=r["se"][0], GAMMA=g, SE_GAMMA=r["se"][1],
                    J=r["J"], J_DF=r["J_df"], J_PVAL=r["J_p"], T_USED=r["T"], N_DROPPED_NONPOS_M=0,
                    NW_LAG=r["nw_lag"], CONVERGED=r["converged"],
                    BETA_AT_BOUND=r["at_bound"][0], GAMMA_AT_BOUND=r["at_bound"][1]))
    ok = m1 > 0; nd = int((~ok).sum())
    if ok.sum() >= MIN_OBS_LOG:
        lm1, gg, ZZ = np.log(m1[ok]), g1[ok], Z[ok]
        lb = (math.log(BETA_BOUNDS[0]), math.log(BETA_BOUNDS[1]))
        r = gmm(lambda th: lm1 - th[0] + th[1] * gg, ZZ, [(math.log(bb), gm) for bb, gm in STARTS], [lb, GAMMA_BOUNDS])
        b = math.exp(r["theta"][0]); g = r["theta"][1]
        out.append(dict(SPEC="II_log", BETA=b, SE_BETA=b * r["se"][0], GAMMA=g, SE_GAMMA=r["se"][1],
                        J=r["J"], J_DF=r["J_df"], J_PVAL=r["J_p"], T_USED=r["T"], N_DROPPED_NONPOS_M=nd,
                        NW_LAG=r["nw_lag"], CONVERGED=r["converged"],
                        BETA_AT_BOUND=r["at_bound"][0], GAMMA_AT_BOUND=r["at_bound"][1]))
    else:
        out.append(dict(SPEC="II_log", T_USED=int(ok.sum()), N_DROPPED_NONPOS_M=nd, CONVERGED=False,
                        NOTE="fewer than %d quarters with m > 0" % MIN_OBS_LOG))
    return out

def describe(s):
    return dict(T=len(s), MEAN=float(np.mean(s)), MEDIAN=float(np.median(s)), MAX=float(np.max(s)),
                MIN=float(np.min(s)), STD=float(np.std(s, ddof=1)), N_NONPOS=int((s <= 0).sum()))

# ---------------------------------------------------------------- main -------
stats_rows, est_rows, feas_rows, fit_stats, ser_rows, fit_rows = [], [], [], [], [], []
figdata = {}
for (st, rg) in cells:
    d = P[(P.STATE == st) & (P.REGION == rg)].sort_values("QDATE").reset_index(drop=True)
    ret = (d.RW.notna() & d.RF.notna()).to_numpy().copy(); ret[0] = False
    idx = np.where(ret)[0]
    RW, RF = d.RW.values[idx], d.RF.values[idx]
    sdfs = build_sdfs(RW, RF)
    corr = pd.DataFrame(sdfs).corr()
    for name in SDF_NAMES:
        s = sdfs[name]
        stats_rows.append(dict(STATE=st, REGION=rg, SDF=name, **describe(s),
                               E_m_Rw=float((s * RW).mean()), E_m_Rf=float((s * RF).mean()),
                               CORR_HJ=corr.loc[name, "HJ"], CORR_BM=corr.loc[name, "BM"], CORR_CAPM=corr.loc[name, "CAPM"]))
    for i, j in enumerate(idx):
        ser_rows.append(dict(STATE=st, REGION=rg, QUARTER=d.QUARTER[j], RW_GROSS=RW[i], RF_GROSS=RF[i],
                             SDF_HJ=sdfs["HJ"][i], SDF_BM=sdfs["BM"][i], SDF_CAPM=sdfs["CAPM"][i]))
    C, Y = d.C.values, d.Y.values
    cy = C / Y
    lam_user = LAM[(st, rg)]
    fig_fit = {}
    for lam_src, lam in (("BASELINE_0", 0.0), ("USER", lam_user)):
        S = C - lam * Y
        n_bad = int((S <= 0).sum())
        bad_q = ";".join(d.QUARTER[S <= 0])
        feas_rows.append(dict(STATE=st, REGION=rg, LAMBDA_SOURCE=lam_src, LAMBDA=lam,
                              MIN_C_OVER_Y_Q=float(cy.min()), MEDIAN_C_OVER_Y_Q=float(np.median(cy)),
                              N_QUARTERS=len(d), N_QUARTERS_S_NONPOS=n_bad, QUARTERS_S_NONPOS=bad_q,
                              FEASIBLE=(n_bad == 0)))
        if n_bad:
            for name in SDF_NAMES:
                for spec in ("I_level", "II_log"):
                    est_rows.append(dict(STATE=st, REGION=rg, LAMBDA_SOURCE=lam_src, LAMBDA=lam, SDF=name, SPEC=spec,
                                         STATUS="infeasible: C - lambda*Y <= 0 in %d of %d quarters" % (n_bad, len(d))))
            continue
        dlnS_all = np.full(len(d), np.nan); dlnS_all[1:] = np.diff(np.log(S))
        dlnS = dlnS_all[idx]
        for name in SDF_NAMES:
            for r in estimate(sdfs[name], dlnS):
                row = dict(STATE=st, REGION=rg, LAMBDA_SOURCE=lam_src, LAMBDA=lam, SDF=name, STATUS="ok", **r)
                est_rows.append(row)
                if not np.isfinite(r.get("BETA", np.nan)): continue
                # fitted consumption SDF on the same quarters as the target SDF
                mh = r["BETA"] * np.exp(-r["GAMMA"] * dlnS)
                fit_stats.append(dict(STATE=st, REGION=rg, LAMBDA_SOURCE=lam_src, LAMBDA=lam, TARGET_SDF=name,
                                      SPEC=r["SPEC"], BETA=r["BETA"], GAMMA=r["GAMMA"], **describe(mh),
                                      CORR_WITH_TARGET=float(np.corrcoef(mh, sdfs[name])[0, 1]),
                                      RMSE_VS_TARGET=float(np.sqrt(np.mean((mh - sdfs[name]) ** 2))),
                                      E_mhat_Rw=float((mh * RW).mean()), E_mhat_Rf=float((mh * RF).mean())))
                for i, j in enumerate(idx):
                    fit_rows.append(dict(STATE=st, REGION=rg, LAMBDA_SOURCE=lam_src, LAMBDA=lam, TARGET_SDF=name,
                                         SPEC=r["SPEC"], QUARTER=d.QUARTER[j], DLN_S=dlnS[i], M_TARGET=sdfs[name][i],
                                         M_HAT_CONSUMPTION=mh[i]))
                if r["SPEC"] == "I_level": fig_fit[(lam_src, name)] = mh
    figdata[(st, rg)] = dict(q=d.QDATE.values, C=C, Y=Y, rq=d.QDATE.values[idx], sdfs=sdfs, fit=fig_fit, lam=lam_user)
    log("  done %-18s %-6s lambda=%.4f  T_ret=%d  feasible(user)=%s" % (st, rg, lam_user, len(idx), feas_rows[-1]["FEASIBLE"]))

stats = pd.DataFrame(stats_rows); est = pd.DataFrame(est_rows); feas = pd.DataFrame(feas_rows)
fits = pd.DataFrame(fit_stats); ser = pd.DataFrame(ser_rows); fser = pd.DataFrame(fit_rows)
ECOLS = ["STATE", "REGION", "LAMBDA_SOURCE", "LAMBDA", "SDF", "SPEC", "STATUS", "BETA", "SE_BETA", "GAMMA", "SE_GAMMA",
         "J", "J_DF", "J_PVAL", "T_USED", "N_DROPPED_NONPOS_M", "NW_LAG", "CONVERGED", "BETA_AT_BOUND", "GAMMA_AT_BOUND", "NOTE"]
est = est.reindex(columns=ECOLS)
stats.to_csv(os.path.join(OUT, "SDF_STATS_Q.csv"), index=False, float_format="%.6f")
est.to_csv(os.path.join(OUT, "CCAPM_ESTIMATES_Q.csv"), index=False, float_format="%.6f")
feas.to_csv(os.path.join(OUT, "LAMBDA_FEASIBILITY_Q.csv"), index=False, float_format="%.6f")
fits.to_csv(os.path.join(OUT, "CONSUMPTION_SDF_STATS_Q.csv"), index=False, float_format="%.6f")
ser.to_csv(os.path.join(OUT, "SDF_SERIES_Q.csv"), index=False, float_format="%.8f")
fser.to_csv(os.path.join(OUT, "CONSUMPTION_SDF_SERIES_Q.csv"), index=False, float_format="%.8f")

# ---------------------------------------------------------------- figures ----
def fmt(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax.grid(alpha=.3)

def draw_cell(axs, st, rg, fd):
    axs[0].plot(fd["q"], fd["C"], "k-", lw=1.6, label="Real consumption (quarter sum)")
    axs[0].plot(fd["q"], fd["Y"], "k:", lw=1.6, label="Real income (quarter sum)")
    axs[0].set_title("%s / %s: real consumption and real income, quarterly" % (st, rg)); axs[0].legend(fontsize=8); fmt(axs[0])
    axs[0].set_ylabel("Rs. per household per quarter (real)")
    axs[1].plot(fd["rq"], fd["sdfs"]["BM"], "k-", lw=1.4, label="SDF-BM")
    axs[1].plot(fd["rq"], fd["sdfs"]["HJ"], "k:", lw=1.4, label="SDF-HJ")
    axs[1].plot(fd["rq"], fd["sdfs"]["CAPM"], "k--", lw=1.4, label="SDF-CAPM")
    if ("USER", "HJ") in fd["fit"]:
        axs[1].plot(fd["rq"], fd["fit"][("USER", "HJ")], color="tab:blue", lw=1.4,
                    label="Consumption SDF fitted to HJ, level spec (lambda=%.3f)" % fd["lam"])
    else:
        axs[1].text(0.01, 0.03, "lambda=%.3f infeasible: no consumption SDF" % fd["lam"], transform=axs[1].transAxes, fontsize=8)
    axs[1].axhline(1.0, color="grey", lw=.6)
    axs[1].set_title("SDFs (risk-free = 10Y, market = Sensex), quarterly"); axs[1].legend(fontsize=8); fmt(axs[1])
    axs[1].set_ylabel("m(t+1)")

st, rg = NATIONAL; fd = figdata[(st, rg)]
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(fd["q"], fd["C"], "k-", lw=1.8, label="Real consumption (quarter sum)")
ax.plot(fd["q"], fd["Y"], "k:", lw=1.8, label="Real income (quarter sum)")
ax.set_title("India / Comb: real consumption and real income, quarterly"); ax.set_ylabel("Rs. per household per quarter (real)")
ax.legend(); fmt(ax); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_consumption_income_India_Q.png"), dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(fd["rq"], fd["sdfs"]["BM"], "k-", lw=1.6, label="sdf_browniano (SDF-BM)")
ax.plot(fd["rq"], fd["sdfs"]["HJ"], "k:", lw=1.6, label="sdf_Hansen_Jagannathan (SDF-HJ)")
ax.plot(fd["rq"], fd["sdfs"]["CAPM"], "k--", lw=1.6, label="sdf_Capm (SDF-CAPM)")
if ("USER", "HJ") in fd["fit"]:
    ax.plot(fd["rq"], fd["fit"][("USER", "HJ")], color="tab:blue", lw=1.6,
            label="consumption SDF fitted to HJ, level spec (lambda=%.3f)" % fd["lam"])
ax.axhline(1.0, color="grey", lw=.6)
ax.set_title("India / Comb: stochastic discount factors, quarterly (risk-free = 10Y)"); ax.set_ylabel("m(t+1)")
ax.legend(fontsize=8); fmt(ax); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_sdf_India_Q.png"), dpi=160); plt.close(fig)
with PdfPages(os.path.join(OUT, "figures_all_cells_Q.pdf")) as pdf:
    for (st, rg), fd in figdata.items():
        fig, axs = plt.subplots(2, 1, figsize=(10, 8)); draw_cell(axs, st, rg, fd)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

# ---------------------------------------------------------------- summary ----
fu = feas[feas.LAMBDA_SOURCE == "USER"]
log("\n==== feasibility of your lambda (C - lambda*Y > 0 in every quarter) ====")
log("  feasible in %d / %d cells" % (int(fu.FEASIBLE.sum()), len(fu)))
log(fu[~fu.FEASIBLE][["STATE", "REGION", "LAMBDA", "MIN_C_OVER_Y_Q", "N_QUARTERS_S_NONPOS"]].round(4).to_string(index=False))
log("\n==== SDF statistics, India / Comb (identical for every lambda) ====")
log(stats[(stats.STATE == "India") & (stats.REGION == "Comb")].drop(columns=["STATE", "REGION"]).round(4).to_string(index=False))
log("\n==== CRRA estimates, India cells ====")
e = est[(est.STATE == "India")]
log(e[["REGION", "LAMBDA_SOURCE", "LAMBDA", "SDF", "SPEC", "BETA", "SE_BETA", "GAMMA", "SE_GAMMA", "J_PVAL", "T_USED", "GAMMA_AT_BOUND"]].round(4).to_string(index=False))
# ---------------------------------------------------------------- verification
# (1) every SDF prices both assets in-sample (HJ and CAPM exactly by construction)
# (2) HJ series re-derived from the raw panel for every cell
# (3) log spec is linear GMM: redo both GMM steps in closed form and compare with L-BFGS-B
log("\n==== verification ====")
ok_all = True
def check(label, cond, extra=""):
    global ok_all; ok_all &= bool(cond)
    log("  [%s] %-62s %s" % ("PASS" if cond else "FAIL", label, extra))
s_hc = stats[stats.SDF.isin(["HJ", "CAPM"])]
check("HJ and CAPM: E[m Rw] = E[m Rf] = 1 in every cell",
      np.allclose(s_hc.E_m_Rw, 1, atol=1e-8) and np.allclose(s_hc.E_m_Rf, 1, atol=1e-8))
worst_hj, worst_cf, n_cf = 0.0, 0.0, 0
for (st, rg) in cells:
    d = P[(P.STATE == st) & (P.REGION == rg)].sort_values("QDATE").reset_index(drop=True)
    j = np.where((d.RW.notna() & d.RF.notna()).to_numpy())[0]; j = j[j > 0]
    R = np.column_stack([d.RW.values[j], d.RF.values[j]])
    m = R @ np.linalg.solve(R.T @ R / len(j), np.ones(2))
    s = ser[(ser.STATE == st) & (ser.REGION == rg)].SDF_HJ.values
    worst_hj = max(worst_hj, float(np.max(np.abs(m - s))))
    for src, lam in (("BASELINE_0", 0.0), ("USER", LAM[(st, rg)])):
        S = d.C.values - lam * d.Y.values
        if (S <= 0).any(): continue
        g = np.diff(np.log(S))[j - 1]
        m1, g1 = m[1:], g[1:]; Z = np.column_stack([np.ones(len(m1)), g[:-1], m[:-1]])
        k = m1 > 0
        if k.sum() < MIN_OBS_LOG: continue
        y = np.log(m1[k]); X = np.column_stack([np.ones(k.sum()), -g1[k]]); Zk = Z[k]
        A = Zk.T @ X / len(y); b = Zk.T @ y / len(y)
        t1 = np.linalg.solve(A.T @ A, A.T @ b)
        W = np.linalg.pinv(nw_cov((y - X @ t1)[:, None] * Zk, nw_lag(len(y))))
        t2 = np.linalg.solve(A.T @ W @ A, A.T @ W @ b)
        r = est[(est.STATE == st) & (est.REGION == rg) & (est.LAMBDA_SOURCE == src) &
                (est.SDF == "HJ") & (est.SPEC == "II_log")].iloc[0]
        worst_cf = max(worst_cf, abs(math.exp(t2[0]) - r.BETA), abs(t2[1] - r.GAMMA)); n_cf += 1
check("HJ series re-derived independently (all cells)", worst_hj < 1e-8, "max diff %.1e" % worst_hj)
check("log spec: L-BFGS-B equals closed-form two-step GMM", worst_cf < 1e-4,
      "%d fits, max diff %.1e" % (n_cf, worst_cf))
okr = est[(est.STATUS == "ok") & est.NOTE.isna()]
check("every feasible fit converged", okr.CONVERGED.astype(bool).all(),
      "%d fits" % len(okr))
log("  verification %s" % ("PASSED" if ok_all else "FAILED - do not use these results"))

log("\nwrote to %s:\n  %s" % (OUT, "\n  ".join(sorted(os.listdir(OUT)))))
LOG.close()