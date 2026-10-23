"""Figures.
  median            : median-MAE example patient per trajectory group (test set), data table only -> results/figures/median_patients_test.csv
  trajectories      : median/IQR forecast trajectories by first-grade / converter group, pooled cross-validation out-of-fold seed-mean forecasts -> results/figures/fig_trajectories.{pdf,png}, fig_trajectories_panel_{a,b,c,d}.csv, fig_trajectories_info.json
  example_patients  : one example TEST patient per group, final model, seed mean +/- range -> results/figures/fig_example_patients.{pdf,png}, fig_example_panel_{a,b,c,d}.csv, fig_example_patients_info.json

Can be run on its own or through scripts/driver.sh:
  python -m scripts.figures median
  python -m scripts.figures trajectories
  python -m scripts.figures example_patients
"""
import json
import sys

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from glance.config import RESULTS_LOSSSEL, RESULTS_FINAL, RESULTS_FIGURES
from glance.data import SEEDS, test_pids, rounds
from glance.evaluate import patients, forecasts_from_csv, seed_mean, cv_forecasts, cv_seeds, at, G
from glance.models import run_name

ORANGE = "#D55E00"; MINN = 5
PAPER_RCPARAMS = {"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times"], "mathtext.fontset": "stix",
                  "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                  "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                  "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
plt.rcParams.update(PAPER_RCPARAMS)


def median_patients(Fv, Fv0, PAT, P):
    groups = [("Stable", lambda d: d["traj"] == "Stable"), ("Only up", lambda d: d["traj"] == "Only up"),
              ("Up and down", lambda d: d["traj"] == "Up and down"), ("E2 converter", lambda d: d["E2"])]
    fa = seed_mean(Fv); rows = []
    for g, sel in groups:
        cand = [p for p in P if sel(PAT[p])]
        if not cand:
            continue
        mae = {p: float(np.mean(np.abs(fa[p]["vp"][fa[p]["vt"] > 0] - fa[p]["vy"][fa[p]["vt"] > 0]))) for p in cand}
        order = sorted(cand, key=lambda p: (mae[p], p)); p = order[(len(order) - 1) // 2]
        rows.append(dict(group=g, pid=p, n=len(cand), mae=mae[p]))
    return rows


def median():
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    vs = json.load(open(RESULTS_LOSSSEL / "selection.json"))["V_star"]; PAT = patients()
    T = test_pids(); Pt = [p for p in T if not PAT[p]["late0"]]
    load3 = lambda m, v: [forecasts_from_csv(RESULTS_FINAL / "runs" / run_name(m, 1, v, 0, s) / "predictions_test.csv", PAT) for s in SEEDS]
    Fv, Fv0 = load3("lnn", vs), load3("lnn", "V0")
    rows = median_patients(Fv, Fv0, PAT, Pt)
    pd.DataFrame(rows).to_csv(RESULTS_FIGURES / "median_patients_test.csv", index=False)
    print(rows)


KEY_TRAJ = "lnn_grade_V4"


def _panel_data(F, PAT, ps):
    M = np.array([at(F[p], G) for p in ps])
    q25, md, q75 = np.percentile(M, [25, 50, 75], axis=0)
    tm, tn = [], []
    for t in G:
        v = [PAT[p]["y"][np.isclose(PAT[p]["t"], t)][0] for p in ps if np.isclose(PAT[p]["t"], t).any()]
        tn.append(len(v)); tm.append(np.median(v) if len(v) >= MINN else np.nan)
    ok = len(ps) >= MINN
    return pd.DataFrame(dict(x=G, forecast_median=np.where(ok, md, np.nan), forecast_q25=np.where(ok, q25, np.nan), forecast_q75=np.where(ok, q75, np.nan),
                             true_median=tm, n_true=tn, n_forecast=len(ps)))


def trajectories():
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    PAT = patients(); P = [p for p in rounds()[0][0] if not PAT[p]["late0"]]
    F = seed_mean([cv_forecasts(KEY_TRAJ, PAT)[0][s] for s in cv_seeds(KEY_TRAJ)])
    conv = [p for p in P if PAT[p]["E2"]]; non = [p for p in P if not PAT[p]["E2"]]
    panels = [("a", "Progressive, first grade 3", [p for p in conv if PAT[p]["g0"] == 3], True),
              ("b", "Progressive, first grade 1-2", [p for p in conv if PAT[p]["g0"] in (1, 2)], True),
              ("c", "Stable, first grade 0", [p for p in non if PAT[p]["g0"] == 0], False),
              ("d", "Stable, first grade 1", [p for p in non if PAT[p]["g0"] == 1], False)]
    fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.4), sharex=True, sharey=True)
    info = dict(model="GLANCE = LNN + grade V4 (V*), CV out-of-fold seed-mean forecasts", left_out_grade0_converters=sorted(p for p in conv if PAT[p]["g0"] == 0))
    for ax, (letter, title, ps, top) in zip(axs.ravel(), panels):
        D = _panel_data(F, PAT, ps); D.to_csv(RESULTS_FIGURES / f"fig_trajectories_panel_{letter}.csv", index=False, float_format="%.4f")
        ax.fill_between(D.x, D.forecast_q25, D.forecast_q75, color=ORANGE, alpha=0.22, lw=0)
        ax.plot(D.x, D.forecast_median, color=ORANGE, lw=1.3)
        T = D[D.true_median.notna()]
        ax.plot(T.x, T.true_median, color="black", lw=0.8, marker="o", ms=2.2)
        ax.axhline(3.5, color="#888888", ls=":", lw=0.8)
        short = title.split(", ")[1]
        ax.text(0.03, 0.985, f"{letter}) {short}, $n$ = {len(ps)}", transform=ax.transAxes, va="top", ha="left", fontsize=8)
        rec = dict(title=title, n=len(ps), true_points=len(T), last_true_x=float(T.x.max()) if len(T) else None)
        if top:
            mid = np.array([(PAT[p]["E2_lo"] + PAT[p]["E2_T"]) / 2 for p in ps]); inr = mid[(mid >= 0) & (mid <= 10)]
            ax.plot(inr, np.full(len(inr), 0.0), "|", color="#333333", ms=5, mew=0.6, clip_on=False)
            own = [float(np.interp(m, G, at(F[p], G))) for p, m in zip(ps, mid) if m <= 10]
            mm = float(np.median(mid))
            rec.update(onset_mid_median=mm, onset_mid_iqr=[float(np.percentile(mid, 25)), float(np.percentile(mid, 75))], rug_n=int(len(inr)),
                       rug_left_out_after_10y=int((mid > 10).sum()),
                       median_forecast_curve_at_median_onset=float(np.interp(mm, D.x, D.forecast_median)),
                       median_of_own_forecast_at_own_onset=float(np.median(own)), n_own=len(own))
        info[letter] = rec
        ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.set_xticks([0, 2, 4, 6, 8, 10]); ax.set_yticks([0, 1, 2, 3, 4, 5])
    for ax in axs[1]:
        ax.set_xlabel("Years from first visit")
    for ax in axs[:, 0]:
        ax.set_ylabel("Grade")
    h = [Line2D([], [], color=ORANGE, lw=1.3), Patch(color=ORANGE, alpha=0.22, lw=0), Line2D([], [], color="black", lw=0.8, marker="o", ms=2.2),
         Line2D([], [], color="#888888", ls=":", lw=0.8), Line2D([], [], color="#333333", marker="|", ls="", ms=5, mew=0.6)]
    lab = ["GLANCE forecast (median)", "Forecast IQR", "Observed grade (median)", "Late-AMD threshold (3.5)", "Observed onset (a, b)"]
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.53, 0.0), handlelength=1.5, columnspacing=0.8, handletextpad=0.4, labelspacing=0.25)
    for ax, t in zip(axs[:, 0], ("Progressive (E2)", "Stable (no E2)")):
        ax.text(-0.30, 0.5, t, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.17, right=0.965, top=0.985, bottom=0.27, wspace=0.16, hspace=0.10)
    fig.savefig(RESULTS_FIGURES / "fig_trajectories.pdf"); fig.savefig(RESULTS_FIGURES / "fig_trajectories.png", dpi=300); plt.close(fig)
    json.dump(info, open(RESULTS_FIGURES / "fig_trajectories_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


def example_patients():
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    PAT = patients(); P = [p for p in test_pids() if not PAT[p]["late0"]]
    Fs = [forecasts_from_csv(RESULTS_FINAL / "runs" / run_name("lnn", 1, "V4", 0, s) / "predictions_test.csv", PAT) for s in SEEDS]
    Fa = seed_mean(Fs)
    mae = {p: float(np.mean(np.abs(Fa[p]["vp"][Fa[p]["vt"] > 0] - Fa[p]["vy"][Fa[p]["vt"] > 0]))) for p in P}
    groups = [("a", "Progressors", "first grade 3", lambda d: d["E2"] and d["g0"] == 3), ("b", "Progressors", "first grade 1-2", lambda d: d["E2"] and d["g0"] in (1, 2)),
              ("c", "Stable", "first grade 0", lambda d: (not d["E2"]) and d["g0"] == 0), ("d", "Stable", "first grade 1", lambda d: (not d["E2"]) and d["g0"] == 1)]
    pick, info = {}, {}
    for letter, row, lab, sel in groups:
        allg = [p for p in P if sel(PAT[p])]
        elig = sorted([p for p in allg if (PAT[p]["t"] > 0).sum() >= 3], key=lambda p: (mae[p], p))
        n = len(elig)
        assert n, (letter, "no eligible patient")
        mids = [elig[(n - 1) // 2]] if n % 2 else sorted([elig[n // 2 - 1], elig[n // 2]])
        p = mids[0]; pick[letter] = p; d = PAT[p]
        info[letter] = dict(group=f"{row}, {lab}", patient=p, first_grade=int(d["g0"]), times=d["t"].tolist(), grades=d["y"].tolist(),
                            patient_mae=mae[p], group_n=len(allg), eligible_n=n, eligible_median_mae=float(np.median([mae[q] for q in elig])),
                            group_median_mae_all=float(np.median([mae[q] for q in allg])), e2_time=d["E2_T"] if d["E2"] else None)
    xmax = max(10.0, max(PAT[p]["t"][-1] for p in pick.values()))
    fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.4), sharex=True, sharey=True)
    for ax, (letter, row, lab, _) in zip(axs.ravel(), groups):
        p = pick[letter]; d = PAT[p]
        gt = Fa[p]["gt"]; keep = np.isin(np.round(gt, 4), np.round(G, 4)) | (gt > 10)
        t = gt[keep]; m = Fa[p]["gp"][keep]; S = np.array([F[p]["gp"][keep] for F in Fs]); lo, hi = S.min(0), S.max(0)
        pd.DataFrame(dict(x=t, forecast_mean=m, seed_min=lo, seed_max=hi, **{f"seed{s}": S[i] for i, s in enumerate(SEEDS)})).merge(
            pd.DataFrame(dict(x=d["t"], observed=d["y"])), on="x", how="outer").sort_values("x").to_csv(RESULTS_FIGURES / f"fig_example_panel_{letter}.csv", index=False, float_format="%.4f")
        ax.fill_between(t, lo, hi, color=ORANGE, alpha=0.22, lw=0)
        ax.plot(t, m, color=ORANGE, lw=1.3)
        ax.plot(d["t"], d["y"], color="black", lw=0.6, marker="o", ms=2.6, clip_on=False, zorder=4)
        ax.axhline(4, color="#888888", ls=":", lw=0.8)
        ax.text(0.03, 0.985, f"{letter}) {lab}", transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.set_xlim(0, xmax); ax.set_ylim(0, 5); ax.set_xticks(np.arange(0, xmax + 0.01, 2)); ax.set_yticks(range(6))
    for ax in axs[1]:
        ax.set_xlabel("Years from first visit")
    for ax, rlab in zip(axs[:, 0], ("Progressors", "Stable")):
        ax.set_ylabel("Grade"); ax.text(-0.30, 0.5, rlab, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=8)
    h = [Line2D([], [], color=ORANGE, lw=1.3), Patch(color=ORANGE, alpha=0.22, lw=0), Line2D([], [], color="black", lw=0.6, marker="o", ms=2.6),
         Line2D([], [], color="#888888", ls=":", lw=0.8)]
    fig.legend(h, ["GLANCE forecast", "Range over 3 seeds", "Observed grade", "Late AMD (grade 4)"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.53, 0.0), handlelength=1.5, columnspacing=0.8, handletextpad=0.4, labelspacing=0.25)
    fig.subplots_adjust(left=0.17, right=0.965, top=0.985, bottom=0.22, wspace=0.16, hspace=0.10)
    fig.savefig(RESULTS_FIGURES / "fig_example_patients.pdf"); fig.savefig(RESULTS_FIGURES / "fig_example_patients.png", dpi=300); plt.close(fig)
    json.dump(dict(rule="median MAE (seed-mean forecast, future visits) among the group's evaluated TEST patients with >= 3 future visits; even count -> lower ID of the two middle",
                   x_max=xmax, **info), open(RESULTS_FIGURES / "fig_example_patients_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


def main():
    cmd = sys.argv[1]
    {"median": median, "trajectories": trajectories, "example_patients": example_patients}[cmd]()


if __name__ == "__main__":
    main()
