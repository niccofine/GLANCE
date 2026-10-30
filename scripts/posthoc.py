"""Post-pipeline analyses (no training; saved forecasts and saved explainability outputs only). Cross-validation out-of-fold = primary; test set = descriptive only.
  onset                   : onset timing  -> results/posthoc/onset_{summary,tests,per_patient}.csv
  onset_class             : 4-way onset classification (on time / early / late / missed), by first-grade stratum -> results/posthoc/onset_class.csv
  late_per_seed           : "% late" onset error per seed (not pooled), cross-validation fixed grid / no grid (S0) / test -> results/posthoc/late_per_seed.csv
  onset_by_grade          : the same onset-timing errors broken out by starting grade -> results/posthoc/onset_by_grade.csv
  figs                    : converters aligned at onset, non-converters by first grade -> results/posthoc/table_{converters_aligned,nonconverters}.csv
  bygrade                 : discrimination within first-grade strata -> results/posthoc/bygrade_{metrics,tests}.csv
  ablation_extra          : no-grid AUC 5y for the irregular-visit experiment + loss-ablation V0-V4 rising/late stats -> results/posthoc/{irregular/auc5_nogrid.csv,ablation_extra.csv}
  explain59               : explainability -> results/posthoc/{A_shares_eval59.json, B2_concepts_eval59.csv, C_macula_eval59.csv, C_maps_eval59.npz, explain_summary_eval59.json}
  irregular               : evaluation of the irregular-visit training experiment, seed 0, 5 backbones -> results/posthoc/irregular/{irreg_metrics.csv, irreg_tests.csv, fig_irregular.{pdf,png}}
  irregular_3seeds        : the same experiment over 3 seeds (LNN/RNN/LSTM/NO), + AUC 5y and C -> results/posthoc/{irreg3_metrics.csv, irreg3_tests.csv}
  querystep               : re-runs saved crossvalidation checkpoints at 3 query-grid resolutions (no training) -> results/posthoc/querystep/{querystep_metrics.csv, querystep_tests.csv, check_reproduce_saved.json, querystep_info.json, fig_querystep_example.{pdf,png}}
  offgrid_step0           : visit-time grid checks -> results/posthoc/offgrid/step0.json
  offgrid_unseen_visits   : forecasts at unseen (dropped) visit dates, no-grid -50% irregular-visit models (glance.irregular), 3 seeds -> results/posthoc/offgrid/{unseen_per_seed.csv, unseen_points.csv.gz, unseen_tests.csv}
  offgrid_annual_grid     : annual-grid RNN / LSTM vs the existing 6-month V4 cross-validation runs -> results/posthoc/offgrid/annual_per_seed.csv
  offgrid_monthly         : fixed-grid V4 GLANCE / NO CV checkpoints queried monthly (0-10 y, step 1/12 y) vs their saved 6-month forecasts -> results/posthoc/offgrid/{monthly_per_seed.csv, monthly_tests.csv, monthly_check_6m_points.csv}
  progression_metrics     : progression metrics, saved forecasts only -> results/posthoc/progression_{per_seed,tests}_{cv,test}.csv
  continuous_consistency  : step-consistency of the continuous-time variants  -> results/posthoc/continuous/{consistency.csv, consistency_verdict.json}
  continuous_unseen_visits: forecasts at UNSEEN / dropped visit dates for the continuous-time variants -> results/posthoc/continuous/{unseen_points.csv.gz, unseen_per_seed.csv}
  continuous_criteria     : performance criterion of each continuous-time variant -> results/posthoc/continuous/{performance_criterion_<variant>.csv, tests_criteria.csv}

Needs allresults.py to have been run first for continuous_criteria.
Can be run on its own or through scripts/driver.sh:
    python -m scripts.posthoc onset
    python -m scripts.posthoc onset_class
    python -m scripts.posthoc late_per_seed
    python -m scripts.posthoc onset_by_grade
    python -m scripts.posthoc figs
    python -m scripts.posthoc bygrade
    python -m scripts.posthoc ablation_extra
    python -m scripts.posthoc explain59
    python -m scripts.posthoc irregular
    python -m scripts.posthoc irregular_3seeds
    python -m scripts.posthoc querystep
    python -m scripts.posthoc offgrid_step0
    python -m scripts.posthoc offgrid_unseen_visits
    python -m scripts.posthoc offgrid_annual_grid
    python -m scripts.posthoc offgrid_monthly
    python -m scripts.posthoc progression_metrics
    python -m scripts.posthoc continuous_consistency
    python -m scripts.posthoc continuous_unseen_visits
    python -m scripts.posthoc continuous_criteria
"""
import json
import pickle
import shutil
import sys
import time
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon
from glance.config import RESULTS_CV, RESULTS_EVAL, RESULTS_FINAL, RESULTS_EXPLAIN, RESULTS_POSTHOC, DROP_TOL
from glance.data import SEEDS, FOLDS, rounds, round_data, test_pids, load_samples, load_X, V2Dataset, MinMaxNorm
from glance.evaluate import Scorer, boot_weights, case_ctrl, cindex_w, cv_forecasts, cv_seeds, first_cross, forecasts_from_csv, holm, label, metrics, paired_p, patients, seed_mean, summarise, at, G, FIX_C, valid_times, auc_w, wmedian
from glance.models import run_name, build
from glance.irregular import OUT as IRREG_OUT, MODELS as IRREG_MODELS, SETTINGS as IRREG_SETTINGS, build as irreg_build, thin as irreg_thin, rdir as irreg_rdir
from glance.annual import rdir as annual_rdir
from glance.continuous import build as continuous_build, rdir as continuous_rdir, pure_grid as continuous_pure_grid
from scripts.allresults import OUT as ALLRESULTS_OUT, CNAME as CONT_CNAME, agg as ar_agg
from scripts.explainability import CONCEPTS, SHORT, bh, boot_mean_ci, ig_all, part_C, setup as explain_setup
from scripts.evaluation import TEST_OUT
from scripts.figures import PAPER_RCPARAMS, ORANGE

NEVER = 10.0


def _onset(f, c):
    v = at(f, G); h = np.where(v >= c)[0]
    return G[h[0]] if len(h) else np.inf


def _per_patient(F, PAT, P, c, is_cox):
    cc = 0.5 if is_cox else c; rows = []
    for p in P:
        d = PAT[p]; pr = _onset(F[p], cc); pe = pr if np.isfinite(pr) else NEVER
        r = dict(pid=p, g0=d["g0"], conv=d["E2"], pred=pr)
        if d["E2"]:
            lo, hi = d["E2_lo"], d["E2_T"]
            r.update(lo=lo, hi=hi, abs=max(lo - pe, 0.0, pe - hi), signed=pe - (lo + hi) / 2)
        else:
            r.update(fa=bool(np.isfinite(pr) and pr <= d["last"]))
        rows.append(r)
    return pd.DataFrame(rows)


def _onset_summary(D):
    out = {}
    for tag, sel in (("all", D.conv), ("g3", D.conv & (D.g0 == 3)), ("g12", D.conv & D.g0.isin([1, 2]))):
        x = D[sel]
        if not len(x):
            continue
        a = x["abs"].values
        out.update({f"{tag}_n": len(x), f"{tag}_median_abs": np.median(a), f"{tag}_mean_abs": a.mean(), f"{tag}_q25": np.percentile(a, 25),
                    f"{tag}_q75": np.percentile(a, 75), f"{tag}_within1": 100 * (a <= 1).mean(), f"{tag}_within2": 100 * (a <= 2).mean(),
                    f"{tag}_median_signed": np.median(x.signed.values), f"{tag}_missed": 100 * (~np.isfinite(x.pred.values)).mean()})
    nc = D[~D.conv]; out["fa"] = 100 * nc.fa.mean(); out["n_nonconv"] = len(nc)
    return out


def _onset_models_cv(PAT):
    keys = ["lnn_grade_V0", "lnn_grade_V1", "lnn_grade_V2", "lnn_grade_V3", "lnn_grade_V4", "lnn_img_V1", "rnn_grade_V4", "lstm_grade_V4", "no_grade_V4"]
    M = {}
    for k in keys:
        F, _ = cv_forecasts(k, PAT); ss = cv_seeds(k); M[k] = {s: F[s] for s in ss}; M[k]["a"] = seed_mean([F[s] for s in ss])
    cox = {}
    for k in FOLDS:
        cox.update(pickle.load(open(RESULTS_FINAL / "baselines" / "cv" / f"fold{k}.pkl", "rb"))["pred"]["cox"])
    M["cox"] = {0: cox, "a": cox}
    return M


def _onset_models_test(PAT):
    keys = ["lnn_grade_V0", "lnn_grade_V1", "lnn_grade_V4", "lnn_img_V1", "rnn_grade_V4", "lstm_grade_V4", "no_grade_V4"]
    M = {}
    for k in keys:
        m, g, v = k.split("_")
        Fs = [forecasts_from_csv(RESULTS_FINAL / "runs" / run_name(m, int(g == "grade"), v, 0, s) / "predictions_test.csv", PAT) for s in SEEDS]
        M[k] = {s: Fs[i] for i, s in enumerate(SEEDS)}; M[k]["a"] = seed_mean(Fs)
    M["cox"] = {0: pickle.load(open(TEST_OUT / "baseline_predictions_test.pkl", "rb"))["cox"]}; M["cox"]["a"] = M["cox"][0]
    return M


def _onset_analyse(M, PAT, P, c, setname):
    rows, pp = [], []
    for k, seeds in M.items():
        per = []
        for s in [s for s in seeds if s != "a"]:
            D = _per_patient(seeds[s], PAT, P, c, k == "cox"); per.append(_onset_summary(D))
            pp.append(D.assign(model=k, seed=s))
        for met in per[0]:
            v = np.array([x.get(met, np.nan) for x in per], float)
            rows.append(dict(set=setname, threshold=c, model=k, label=label(k), metric=met, mean=np.nanmean(v),
                             sd=np.nanstd(v, ddof=1) if len(v) > 1 else np.nan, n_seeds=len(v)))
    S = pd.DataFrame(rows)
    W = boot_weights(len(P)); conv = np.array([PAT[p]["E2"] for p in P])
    A = {}
    for k in M:
        D = _per_patient(M[k]["a"], PAT, P, c, k == "cox").set_index("pid").loc[P]
        a = np.where(conv, D["abs"].fillna(0).values, 0.0)
        A[k] = (a[conv].mean(), (W[:, conv] @ a[conv]) / W[:, conv].sum(1))
    pairs = [("lnn_grade_V1", "lnn_grade_V0"), ("lnn_grade_V2", "lnn_grade_V1"), ("lnn_grade_V3", "lnn_grade_V1"), ("lnn_grade_V4", "lnn_grade_V1"),
             ("lnn_grade_V4", "rnn_grade_V4"), ("lnn_grade_V4", "lstm_grade_V4"), ("lnn_grade_V4", "no_grade_V4"), ("lnn_grade_V4", "cox")]
    T = []
    for a, b in pairs:
        if a not in M or b not in M:
            continue
        d = A[a][1] - A[b][1]
        T.append(dict(set=setname, threshold=c, model_a=a, model_b=b, mean_abs_a=A[a][0], mean_abs_b=A[b][0], diff=A[a][0] - A[b][0],
                      lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p=paired_p(d)))
    T = pd.DataFrame(T); T["p_holm"] = holm(T.p.values)
    return S, T, pd.concat(pp)


def onset():
    RESULTS_POSTHOC.mkdir(parents=True, exist_ok=True)
    PAT = patients()
    Pc = [p for p in rounds()[0][0] if not PAT[p]["late0"]]; Pt = [p for p in test_pids() if not PAT[p]["late0"]]
    Mc, Mt = _onset_models_cv(PAT), _onset_models_test(PAT)
    res = [_onset_analyse(Mc, PAT, Pc, 3.5, "CV"), _onset_analyse(Mt, PAT, Pt, 3.5, "TEST"), _onset_analyse(Mc, PAT, Pc, 3.0, "CV")]
    pd.concat([r[0] for r in res]).to_csv(RESULTS_POSTHOC / "onset_summary.csv", index=False)
    pd.concat([r[1] for r in res]).to_csv(RESULTS_POSTHOC / "onset_tests.csv", index=False)
    pd.concat([r[2].assign(set=s, threshold=c) for r, s, c in zip(res, ("CV", "TEST", "CV"), (3.5, 3.5, 3.0))]).to_csv(RESULTS_POSTHOC / "onset_per_patient.csv", index=False)
    print("onset summary rows:", sum(len(r[0]) for r in res))


OC_KEYS = {"lnn_grade_V4": "GLANCE", "rnn_grade_V4": "RNN V4", "lstm_grade_V4": "LSTM V4", "no_grade_V4": "NO V4"}


def _onsetclass_forecasts(setname, PAT):
    F = {}
    for k in OC_KEYS:
        if setname == "CV":
            Fk, _ = cv_forecasts(k, PAT); F[k] = seed_mean([Fk[s] for s in cv_seeds(k)])
        else:
            m, g, v = k.split("_")
            F[k] = seed_mean([forecasts_from_csv(RESULTS_FINAL / "runs" / run_name(m, 1, v, 0, s) / "predictions_test.csv", PAT) for s in SEEDS])
    return F


def _onsetclass_classify(f, d):
    v = at(f, G); h = np.where(v >= 3.5)[0]
    if not len(h):
        return "missed", np.nan
    o = G[h[0]]; lo, hi = d["E2_lo"], d["E2_T"]
    return ("early" if o < lo - 1 else "late" if o > hi + 1 else "on time"), o


def onset_class():
    RESULTS_POSTHOC.mkdir(parents=True, exist_ok=True)
    PAT = patients(); rows = []
    for sname, P in (("CV", [p for p in rounds()[0][0] if not PAT[p]["late0"]]), ("TEST", [p for p in test_pids() if not PAT[p]["late0"]])):
        F = _onsetclass_forecasts(sname, PAT); conv = [p for p in P if PAT[p]["E2"]]
        for k in OC_KEYS:
            for p in conv:
                c, o = _onsetclass_classify(F[k][p], PAT[p]); d = PAT[p]
                rows.append(dict(set=sname, model=k, pid=p, g0=d["g0"], cls=c, onset=o, lo=d["E2_lo"], hi=d["E2_T"],
                                 forecast_at_hi=float(at(F[k][p], d["E2_T"])[0])))
    D = pd.DataFrame(rows); D.to_csv(RESULTS_POSTHOC / "onset_class.csv", index=False)
    print("onset_class rows:", len(D))


LPS_MODELS = ("rnn", "lstm", "no", "lnn")
LPS_LAB = {"rnn": "RNN + grade", "lstm": "LSTM + grade", "no": "NO + grade", "lnn": "GLANCE (LNN + grade V4)"}


def _lps_late_grid(f, d):
    g = at(f, G); h = np.where(g >= FIX_C)[0]
    return bool(len(h)) and G[h[0]] > d["E2_T"] + 1


def _lps_late_visits(q, d):
    t, yp = q.t.values, q.y_pred.values; h = np.where(yp >= FIX_C)[0]
    return bool(len(h)) and t[h[0]] > d["E2_T"] + 1


def _lps_row(part, m, conv, flags):
    n = np.array([sum(fl) for fl in flags]); pct = 100 * n / len(conv)
    return dict(part=part, model=m, n_conv=len(conv), n_seeds=len(n), late_mean=pct.mean(), late_sd=pct.std(ddof=1) if len(n) > 1 else np.nan,
                late_pct_per_seed=";".join(f"{x:.1f}" for x in pct), late_n_per_seed="/".join(str(x) for x in n))


def late_per_seed():
    RESULTS_POSTHOC.mkdir(parents=True, exist_ok=True)
    PAT = patients()
    Pcv = sorted(p for p in rounds()[0][0] if not PAT[p]["late0"]); Ccv = [p for p in Pcv if PAT[p]["E2"]]
    Pte = sorted(p for p in test_pids() if not PAT[p]["late0"]); Cte = [p for p in Pte if PAT[p]["E2"]]
    rows, pending = [], []
    for m in LPS_MODELS:
        F, _ = cv_forecasts(f"{m}_grade_V4", PAT); assert sorted(F) == list(SEEDS)
        rows.append(_lps_row("CV fixed grid", m, Ccv, [[_lps_late_grid(F[s][p], PAT[p]) for p in Ccv] for s in SEEDS]))
    g = rows[-1]; assert f"{g['late_mean']:.1f} +/- {g['late_sd']:.1f}" == "0.8 +/- 1.4", g
    for m in LPS_MODELS:
        if not all((irreg_rdir(m, "S0", k, s) / "done.json").exists() for k in FOLDS for s in SEEDS):
            pending.append(m); continue
        fl = []
        for s in SEEDS:
            D = pd.concat([pd.read_csv(irreg_rdir(m, "S0", k, s) / "predictions_heldout.csv") for k in FOLDS]); Q = dict(list(D.groupby("pid")))
            assert set(Ccv) <= set(Q)
            fl.append([_lps_late_visits(Q[p].sort_values("t"), PAT[p]) for p in Ccv])
        rows.append(_lps_row("CV no grid (S0)", m, Ccv, fl))
    for m in LPS_MODELS:
        fl = []
        for s in SEEDS:
            F = forecasts_from_csv(RESULTS_FINAL / "runs" / run_name(m, 1, "V4", 0, s) / "predictions_test.csv", PAT, set(Pte))
            fl.append([_lps_late_grid(F[p], PAT[p]) for p in Cte])
        rows.append(_lps_row("TEST fixed grid", m, Cte, fl))
    R = pd.DataFrame(rows); R.to_csv(RESULTS_POSTHOC / "late_per_seed.csv", index=False)
    print("late_per_seed rows:", len(R), "| pending (S0 runs not finished):", pending or "none")


OBG_MODELS = ("rnn", "lstm", "no", "lnn")
OBG_GRADES = (0, 1, 2, 3)
OBG_CNT = ["n_progressors", "on_time", "early", "late", "missed", "n_stable", "false_alerts", "false_alerts_any"]


def _obg_classify(pr, d):
    if not np.isfinite(pr):
        return "missed"
    if pr < d["E2_lo"] - 1:
        return "early"
    if pr > d["E2_T"] + 1:
        return "late"
    return "on_time"


def onset_by_grade():
    PAT = patients(); P = sorted(p for p in rounds()[0][0] if not PAT[p]["late0"])
    assert set(PAT[p]["g0"] for p in P) <= set(OBG_GRADES)
    rows = []
    for m in OBG_MODELS:
        F, _ = cv_forecasts(f"{m}_grade_V4", PAT); assert sorted(F) == list(SEEDS)
        for s in SEEDS:
            for g in OBG_GRADES:
                r = dict(model=m, seed=s, first_grade=g, **{c: 0 for c in OBG_CNT})
                for p in (p for p in P if PAT[p]["g0"] == g):
                    d = PAT[p]; pr = first_cross(at(F[s][p], G), FIX_C)
                    if d["E2"]:
                        r["n_progressors"] += 1; r[_obg_classify(pr, d)] += 1
                    else:
                        r["n_stable"] += 1; r["false_alerts"] += int(pr <= d["last"]); r["false_alerts_any"] += int(np.isfinite(pr))
                assert r["on_time"] + r["early"] + r["late"] + r["missed"] == r["n_progressors"]
                rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(RESULTS_POSTHOC / "onset_by_grade.csv", index=False)

    LP = pd.read_csv(RESULTS_POSTHOC / "late_per_seed.csv"); LP = LP[LP.part == "CV fixed grid"].set_index("model")
    for m in OBG_MODELS:
        T = R[R.model == m]; pooled = 100 * T.late.sum() / T.n_progressors.sum()
        ref = LP.loc[m, "late_mean"]; n_ref = "/".join(str(T[T.seed == s].late.sum()) for s in SEEDS)
        assert abs(pooled - ref) < 1e-9 and n_ref == LP.loc[m, "late_n_per_seed"], (m, pooled, ref)
    print("onset_by_grade rows:", len(R))


XA = np.round(np.arange(-8, 4.0001, 0.25), 2); MINN = 5


def _fc_at(f, t):
    v = at(f, G); t = np.asarray(t, float)
    return np.where((t >= 0) & (t <= 10), np.interp(t, G, v), np.nan)


def _true_at(d, t):
    t = np.asarray(t, float)
    return np.where((t >= d["t"][0]) & (t <= d["t"][-1]), np.interp(t, d["t"], d["y"]), np.nan)


def _med_iqr(M):
    n = np.isfinite(M).sum(0); ok = n >= MINN
    with np.errstate(all="ignore"):
        md, q1, q3 = (np.where(ok, np.nanpercentile(M, q, axis=0), np.nan) for q in (50, 25, 75))
    return md, q1, q3, n


def _figs_load(PAT):
    cv = {v: seed_mean([cv_forecasts(f"lnn_grade_{v}", PAT)[0][s] for s in cv_seeds(f"lnn_grade_{v}")]) for v in ("V0", "V1", "V2", "V3", "V4")}
    te = {v: seed_mean([forecasts_from_csv(RESULTS_FINAL / "runs" / run_name("lnn", 1, v, 0, s) / "predictions_test.csv", PAT) for s in SEEDS])
          for v in ("V0", "V1", "V4")}
    return cv, te


def _table_converters_aligned(F, PAT, P, tag):
    vs = list(F); conv = [p for p in P if PAT[p]["E2"]]
    groups = [("First grade 3", [p for p in conv if PAT[p]["g0"] == 3]), ("First grade 1-2", [p for p in conv if PAT[p]["g0"] in (1, 2)])]
    rows = []
    for title, ps in groups:
        mid = {p: (PAT[p]["E2_lo"] + PAT[p]["E2_T"]) / 2 for p in ps}
        for v in vs:
            at0 = np.array([_fc_at(F[v][p], mid[p]) for p in ps]); ok3 = [i for i, p in enumerate(ps) if mid[p] - 3 >= 0]
            rise = np.array([_fc_at(F[v][ps[i]], mid[ps[i]]) - _fc_at(F[v][ps[i]], mid[ps[i]] - 3) for i in ok3])
            rows.append(dict(set=tag, panel=title, series=f"LNN + grade {v}", n=len(ps), at_onset=np.nanmedian(at0), n_onset=int(np.isfinite(at0).sum()),
                             rise_3y=np.nanmedian(rise) if np.isfinite(rise).any() else np.nan, n_rise=int(np.isfinite(rise).sum())))
        t0 = np.array([_true_at(PAT[p], mid[p]) for p in ps]); ok3 = [p for p in ps if mid[p] - 3 >= 0]
        rows.append(dict(set=tag, panel=title, series="True grade", n=len(ps), at_onset=np.nanmedian(t0), n_onset=int(np.isfinite(t0).sum()),
                         rise_3y=np.nanmedian([_true_at(PAT[p], mid[p]) - _true_at(PAT[p], mid[p] - 3) for p in ok3]) if ok3 else np.nan, n_rise=len(ok3)))
    return rows


def _table_nonconverters(F, PAT, P, tag):
    vs = [v for v in ("V0", "V1", "V4") if v in F]; non = [p for p in P if not PAT[p]["E2"]]
    rows = []
    for g in range(4):
        ps = [p for p in non if PAT[p]["g0"] == g]
        for v in vs:
            fa = np.mean([np.any(at(F[v][p], G) >= 3.5) for p in ps])
            sabs = sum(np.abs(F[v][p]["vp"][F[v][p]["vt"] > 0] - F[v][p]["vy"][F[v][p]["vt"] > 0]).sum() for p in ps)
            nv = sum((F[v][p]["vt"] > 0).sum() for p in ps)
            rows.append(dict(set=tag, first_grade=g, series=f"LNN + grade {v}", n=len(ps), false_alarm_pct=100 * fa, mae=sabs / nv,
                             pred_rise_0_10=np.median([at(F[v][p], 10.0)[0] - at(F[v][p], 0.0)[0] for p in ps])))
        rows.append(dict(set=tag, first_grade=g, series="True grade", n=len(ps), obs_rise=np.median([PAT[p]["y"][-1] - PAT[p]["y"][0] for p in ps])))
    return rows


def figs():
    RESULTS_POSTHOC.mkdir(parents=True, exist_ok=True)
    PAT = patients(); Pc = [p for p in rounds()[0][0] if not PAT[p]["late0"]]; Pt = [p for p in test_pids() if not PAT[p]["late0"]]
    Fcv, Fte = _figs_load(PAT)
    A = pd.DataFrame(_table_converters_aligned(Fcv, PAT, Pc, "CV") + _table_converters_aligned(Fte, PAT, Pt, "TEST"))
    B = pd.DataFrame(_table_nonconverters(Fcv, PAT, Pc, "CV") + _table_nonconverters(Fte, PAT, Pt, "TEST"))
    A.to_csv(RESULTS_POSTHOC / "table_converters_aligned.csv", index=False)
    B.to_csv(RESULTS_POSTHOC / "table_nonconverters.csv", index=False)
    print("converters aligned rows:", len(A), "| nonconverters rows:", len(B))


KEYS = ["lnn_grade_V0", "lnn_grade_V1", "lnn_grade_V4", "lnn_img_V1", "rnn_grade_V4", "lstm_grade_V4", "no_grade_V4"]
METS = ["iauc_E2", "auc2_E2", "auc5_E2", "c_E2"]


def _bygrade_scorers(setname, PAT):
    SC = {}
    for k in KEYS:
        if setname == "CV":
            F, _ = cv_forecasts(k, PAT); Fs = [F[s] for s in cv_seeds(k)]
        else:
            m, g, v = k.split("_")
            Fs = [forecasts_from_csv(RESULTS_FINAL / "runs" / run_name(m, int(g == "grade"), v, 0, s) / "predictions_test.csv", PAT) for s in SEEDS]
        SC[k] = {i: Scorer("grade", f) for i, f in enumerate(Fs)}; SC[k]["a"] = Scorer("grade", seed_mean(Fs))
    if setname == "CV":
        cox = {}
        for k in FOLDS:
            cox.update(pickle.load(open(RESULTS_FINAL / "baselines" / "cv" / f"fold{k}.pkl", "rb"))["pred"]["cox"])
    else:
        cox = pickle.load(open(TEST_OUT / "baseline_predictions_test.pkl", "rb"))["cox"]
    SC["cox"] = {0: Scorer("cox", cox)}; SC["cox"]["a"] = SC["cox"][0]
    return SC


def bygrade():
    RESULTS_POSTHOC.mkdir(parents=True, exist_ok=True)
    PAT = patients(); rows, trows = [], []
    sets = {"CV": [p for p in rounds()[0][0] if not PAT[p]["late0"]], "TEST": [p for p in test_pids() if not PAT[p]["late0"]]}
    for sname, P in sets.items():
        SC = _bygrade_scorers(sname, PAT)
        for stratum, gs in (("first grade 3", (3,)), ("first grade 1-2", (1, 2))):
            Ps = [p for p in P if PAT[p]["g0"] in gs]
            S, B, tv = summarise(SC, Ps, PAT, boot=True)
            S = S[S.metric.isin(METS)].assign(set=sname, stratum=stratum, n_conv=sum(PAT[p]["E2"] for p in Ps), n_nonconv=sum(not PAT[p]["E2"] for p in Ps),
                                               iauc_times=" ".join(f"{t:g}" for t in tv["E2"]))
            rows.append(S)
            ONE = np.ones((1, len(Ps))); T = []
            for b in ("cox", "lstm_grade_V4"):
                for met in ("iauc_E2", "c_E2"):
                    va = metrics(SC["lnn_grade_V4"]["a"], Ps, ONE, PAT, tv)[met][0]; vb = metrics(SC[b]["a"], Ps, ONE, PAT, tv)[met][0]
                    d = B["lnn_grade_V4"][met] - B[b][met]; dd = d[np.isfinite(d)]
                    T.append(dict(set=sname, stratum=stratum, model_a="lnn_grade_V4", model_b=b, metric=met, value_a=va, value_b=vb, diff=va - vb,
                                  lo=np.percentile(dd, 2.5) if len(dd) else np.nan, hi=np.percentile(dd, 97.5) if len(dd) else np.nan, p=paired_p(d)))
            T = pd.DataFrame(T); k = T.p.notna() & np.isfinite(T["diff"])
            T["p_holm"] = np.nan; T.loc[k, "p_holm"] = holm(T.loc[k, "p"].values) if k.any() else []
            trows.append(T)
    pd.concat(rows).to_csv(RESULTS_POSTHOC / "bygrade_metrics.csv", index=False)
    pd.concat(trows).to_csv(RESULTS_POSTHOC / "bygrade_tests.csv", index=False)
    print("bygrade done")


@torch.no_grad()
def _ae_score5(model, ds, dev):
    out = {}
    for i in range(len(ds)):
        x, t, _, _ = ds[i]
        q = torch.unique(torch.cat([t, torch.tensor([5.0])]), sorted=True)
        p = model(x[None].to(dev), q[None].to(dev))
        p = p[0] if isinstance(p, tuple) else p
        out[ds.s.pid.iloc[i]] = float(p[0, int(torch.nonzero(torch.isclose(q, torch.tensor(5.0)))[0])])
    return out


def _ae_part1(PAT, P, dev):
    case, ctrl = case_ctrl(PAT, P, "E2", 5.0); W = boot_weights(len(P)); ONE = np.ones((1, len(P))); rows = []
    for m in ("lnn", "tlstm", "no"):
        for s in IRREG_SETTINGS:
            sc = {}
            for k in FOLDS:
                D = round_data(k); ho = irreg_thin(D["heldout_first"], IRREG_SETTINGS[s])
                mod = irreg_build(m, 1025).to(dev); mod.load_state_dict(torch.load(irreg_rdir(m, s, k) / "best.pt", map_location=dev)["model_state_dict"]); mod.eval()
                sc.update(_ae_score5(mod, V2Dataset(ho, D["X"], D["scaler"], 1), dev))
            v = np.array([sc[p] for p in P]); b = auc_w(v, case, ctrl, W)
            rows.append(dict(model=m, setting=s, auc5=auc_w(v, case, ctrl, ONE)[0], lo=np.nanpercentile(b, 2.5), hi=np.nanpercentile(b, 97.5),
                             n_cases=int(case.sum()), n_controls=int(ctrl.sum())))
    T = pd.DataFrame(rows); T.to_csv(IRREG_OUT / "auc5_nogrid.csv", index=False)
    M = pd.read_csv(RESULTS_EVAL / "cv" / "metrics_cv.csv")
    ref = {m: M[(M.model == f"{m}_grade_V4") & (M.metric == "auc5_E2")].per_seed.iloc[0].split(";")[0] for m in ("lnn", "no")}
    return T, ref, int(case.sum()), int(ctrl.sum())


def _ae_part2(PAT, P):
    conv = [p for p in P if PAT[p]["E2"]]; rows = []
    for v in ("V0", "V1", "V2", "V3", "V4"):
        F, _ = cv_forecasts(f"lnn_grade_{v}", PAT); per = []
        for s in SEEDS:
            rising = np.mean([(at(F[s][p], 10.0)[0] - at(F[s][p], 8.0)[0]) / 2 > 0.1 for p in P])
            late = []
            for p in conv:
                g = at(F[s][p], G); h = np.where(g >= 3.5)[0]
                late.append(bool(len(h)) and G[h[0]] > PAT[p]["E2_T"] + 1)
            per.append((100 * rising, 100 * np.mean(late)))
        per = np.array(per)
        rows.append(dict(variant=v, rising_mean=per[:, 0].mean(), rising_sd=per[:, 0].std(ddof=1), late_mean=per[:, 1].mean(), late_sd=per[:, 1].std(ddof=1),
                         rising_per_seed=";".join(f"{x:.1f}" for x in per[:, 0]), late_per_seed=";".join(f"{x:.1f}" for x in per[:, 1])))
    D = pd.DataFrame(rows); D.to_csv(RESULTS_POSTHOC / "ablation_extra.csv", index=False)
    return D, len(conv)


def ablation_extra():
    RESULTS_POSTHOC.mkdir(parents=True, exist_ok=True); IRREG_OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PAT = patients(); P = sorted(p for p in rounds()[0][0] if not PAT[p]["late0"])
    T, ref, nca, nco = _ae_part1(PAT, P, dev); D, nconv = _ae_part2(PAT, P)
    print("ablation_extra: auc5_nogrid rows", len(T), f"(n_cases={nca}, n_controls={nco})",
          "| ablation_extra rows", len(D), f"(n_conv={nconv})")


def explain59():
    RESULTS_POSTHOC.mkdir(parents=True, exist_ok=True)
    PAT = patients()
    A = pd.read_csv(RESULTS_EXPLAIN / "A_shares_per_patient.csv"); A = A[[not PAT[p]["late0"] for p in A.pid]].reset_index(drop=True)
    sumA = dict(n=len(A), share_left=float(A.share_left.mean()), share_right=float(A.share_right.mean()), share_grade=float(A.share_grade.mean()),
                completeness_err_max=float(A.completeness_err.max()),
                by_group={g: dict(n=int((A.group == g).sum()), share_grade=float(A[A.group == g].share_grade.mean()), growth=float(A[A.group == g].growth.mean()))
                          for g in ("Only up", "Stable", "Up and down", "Only down") if (A.group == g).any()})
    json.dump(sumA, open(RESULTS_POSTHOC / "A_shares_eval59.json", "w"), indent=1, default=float)

    B = pd.read_csv(RESULTS_EXPLAIN / "B2_concepts_per_patient.csv"); B = B[B.pid.isin(A.pid)].reset_index(drop=True)
    names = [en for en, _ in CONCEPTS]; pr, st = (B.group == "Only up").values, (B.group == "Stable").values; rows = []
    for nm in names:
        v = B[SHORT[nm]].values; r = dict(concept=nm, mean_all=float(v.mean()), n_all=len(v))
        for g, msk in (("progressors", pr), ("stable", st)):
            m, lo, hi = boot_mean_ci(v[msk]); r.update({f"{g}_mean": m, f"{g}_lo": lo, f"{g}_hi": hi, f"{g}_n": int(msk.sum())})
        r["mwu_p"] = float(mannwhitneyu(v[pr], v[st]).pvalue); r["wilcoxon_vs0_p"] = float(wilcoxon(v).pvalue); rows.append(r)
    B2 = pd.DataFrame(rows); B2["mwu_q"] = bh(B2.mwu_p.values); B2.to_csv(RESULTS_POSTHOC / "B2_concepts_eval59.csv", index=False)

    C = explain_setup(); assert C["vs"] == "V4" and len(C["F"]) == 76
    ORDER = ["G277", "G590", "G271", "G204"]
    idx = [int(np.where(C["F"].pid.values == p)[0][0]) for p in ORDER]
    IG, GR, _ = ig_all(C)
    CT = part_C(C, GR, pick=idx, csv_name="C_macula_eval59.csv", maps_name="C_maps_eval59.npz")
    for name in ("C_macula_eval59.csv", "C_maps_eval59.npz"):
        shutil.move(str(RESULTS_EXPLAIN / name), str(RESULTS_POSTHOC / name))
    S = dict(restricted_to="59 evaluated TEST patients (not late at the first imaged visit)", A=sumA,
             B2=B2.to_dict("records"), C=CT.to_dict("records"), n_patients=59,
             C_image_quality_exclusion="G178 (Stable, first grade 1): bright crescent artefact in both eyes -> replaced by G204")
    json.dump(S, open(RESULTS_POSTHOC / "explain_summary_eval59.json", "w"), indent=1, default=float)
    print("explain59 done")


def _irreg_load(m, s, PAT):
    D = pd.concat([pd.read_csv(irreg_rdir(m, s, f) / "predictions_heldout.csv") for f in FOLDS])
    return {p: (q.t.values, q.y_true.values, q.y_pred.values) for p, q in D.groupby("pid") if p in PAT}


def _irreg_per_patient(F, PAT, P):
    rows = []
    for p in P:
        t, y, yp = F[p]; d = PAT[p]; k = t > 0
        al = t[np.argmax(yp >= 3.5)] if (yp >= 3.5).any() else np.inf
        r = dict(sabs=np.abs(yp[k] - y[k]).sum(), nv=k.sum(), up=d["traj"] == "Only up", down=float(np.any(np.diff(yp) < -DROP_TOL)),
                 E=d["E2"], T=d["E2_T"], risk=yp[-1], alarm=al)
        if d["E2"]:
            lo, hi = d["E2_lo"], d["E2_T"]; det = np.isfinite(al)
            r.update(det=det, ont=det and lo - 1 <= al <= hi + 1, dist=(max(lo - al, 0, al - hi) if det else np.nan))
        else:
            r.update(fa=np.isfinite(al))
        rows.append(r)
    return pd.DataFrame(rows)


def _irreg_metrics(D, W):
    E = D.E.values; up = D.up.values; out = {}
    with np.errstate(all="ignore"):
        out["mae"] = (W @ D.sabs.values) / (W @ D.nv.values)
        out["mae_up"] = (W[:, up] @ D.sabs.values[up]) / (W[:, up] @ D.nv.values[up])
        out["down"] = 100 * (W @ D.down.values) / W.sum(1)
        det = D.det.fillna(False).values.astype(bool); ont = D.ont.fillna(False).values.astype(bool); fa = D.fa.fillna(False).values.astype(bool)
        out["detected"] = 100 * (W @ det) / (W @ E); out["ontime"] = 100 * (W @ ont) / (W @ E); out["fa"] = 100 * (W @ fa) / (W @ ~E)
        dist = D.dist.values; out["mediandist"] = np.array([wmedian(dist[det], w[det]) for w in W])
        out["c"] = cindex_w(D["T"].values, E, D.risk.values, W)
    return out


def irregular():
    IRREG_OUT.mkdir(parents=True, exist_ok=True)
    PAT = patients(); P = sorted(p for p in rounds()[0][0] if not PAT[p]["late0"]); n = len(P)
    W = boot_weights(n); ONE = np.ones((1, n)); R, B, rows = {}, {}, []
    for m in IRREG_MODELS:
        for s in IRREG_SETTINGS:
            D = _irreg_per_patient(_irreg_load(m, s, PAT), PAT, P); R[(m, s)] = _irreg_metrics(D, ONE); B[(m, s)] = _irreg_metrics(D, W)
        for met in R[(m, "S0")]:
            v0, v5 = R[(m, "S0")][met][0], R[(m, "S50")][met][0]; b0, b5 = B[(m, "S0")][met], B[(m, "S50")][met]; ch = b5 - b0
            rows.append(dict(model=m, metric=met, S0=v0, S0_lo=np.nanpercentile(b0, 2.5), S0_hi=np.nanpercentile(b0, 97.5), S50=v5,
                             S50_lo=np.nanpercentile(b5, 2.5), S50_hi=np.nanpercentile(b5, 97.5), change=v5 - v0,
                             change_lo=np.nanpercentile(ch, 2.5), change_hi=np.nanpercentile(ch, 97.5)))
    M = pd.DataFrame(rows); M.to_csv(IRREG_OUT / "irreg_metrics.csv", index=False)
    T = []
    for met in ("mae", "ontime"):
        for b in ("rnn", "lstm", "tlstm", "no"):
            ca = R[("lnn", "S50")][met][0] - R[("lnn", "S0")][met][0]; cb = R[(b, "S50")][met][0] - R[(b, "S0")][met][0]
            d = (B[("lnn", "S50")][met] - B[("lnn", "S0")][met]) - (B[(b, "S50")][met] - B[(b, "S0")][met])
            T.append(dict(metric=met, a="lnn", b=b, change_a=ca, change_b=cb, diff=ca - cb, lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5), p=paired_p(d)))
    T = pd.DataFrame(T); T["p_holm"] = holm(T.p.values); T.to_csv(IRREG_OUT / "irreg_tests.csv", index=False)
    nmodels = len(IRREG_MODELS)
    plt.rcParams.update(PAPER_RCPARAMS)
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.9))
    for ax, met, yl in ((axs[0], "mae", "MAE (grade)"), (axs[1], "ontime", "On time (%)")):
        for j, (s, col, hatch) in enumerate((("S0", ORANGE, ""), ("S50", "#F3C8A8", "///"))):
            r = [M[(M.model == m) & (M.metric == met)].iloc[0] for m in IRREG_MODELS]
            v = np.array([x[s] for x in r]); lo = np.array([x[s + "_lo"] for x in r]); hi = np.array([x[s + "_hi"] for x in r])
            ax.bar(np.arange(nmodels) + (j - 0.5) * 0.38, v, 0.38, yerr=[v - lo, hi - v], capsize=1.5, color=col, hatch=hatch, edgecolor="#333333", lw=0.4,
                   error_kw=dict(lw=0.6), label={"S0": "All visits (S0)", "S50": "50% visits dropped (S50)"}[s])
        ax.set_xticks(range(nmodels)); ax.set_xticklabels(["LNN", "RNN", "LSTM", "T-LSTM", "NO"][:nmodels], rotation=30, ha="right"); ax.set_ylabel(yl)
    axs[0].text(-0.02, 1.03, "a)", transform=axs[0].transAxes, va="bottom"); axs[1].text(-0.02, 1.03, "b)", transform=axs[1].transAxes, va="bottom")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.53, 0.0), handlelength=1.4, columnspacing=0.8)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.36, wspace=0.42)
    fig.savefig(IRREG_OUT / "fig_irregular.pdf"); fig.savefig(IRREG_OUT / "fig_irregular.png", dpi=300); plt.close(fig)
    print("irregular done:", len(M), "metric rows,", len(T), "tests")


I3_MODELS = ("lnn", "rnn", "lstm", "no"); I3_SEEDS = (0, 1, 2)


def _i3_load(m, s, sd, PAT):
    D = pd.concat([pd.read_csv(irreg_rdir(m, s, f, sd) / "predictions_heldout.csv") for f in FOLDS])
    return {p: (q.t.values, q.y_true.values, q.y_pred.values) for p, q in D.groupby("pid") if p in PAT}


def _i3_auc5(m, s, sd, P, case, ctrl, W, dev):
    if m in ("rnn", "lstm"):
        return np.full(W.shape[0], np.nan), np.nan
    sc = {}
    for k in FOLDS:
        D = round_data(k); ho = irreg_thin(D["heldout_first"], IRREG_SETTINGS[s], seed=sd)
        mod = irreg_build(m, 1025).to(dev); mod.load_state_dict(torch.load(irreg_rdir(m, s, k, sd) / "best.pt", map_location=dev)["model_state_dict"]); mod.eval()
        sc.update(_ae_score5(mod, V2Dataset(ho, D["X"], D["scaler"], 1), dev))
    v = np.array([sc[p] for p in P])
    return auc_w(v, case, ctrl, W), auc_w(v, case, ctrl, np.ones((1, len(P))))[0]


def irregular_3seeds():
    RESULTS_POSTHOC.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PAT = patients(); P = sorted(p for p in rounds()[0][0] if not PAT[p]["late0"]); W = boot_weights(len(P)); ONE = np.ones((1, len(P)))
    case, ctrl = case_ctrl(PAT, P, "E2", 5.0)
    V, B, rows = {}, {}, []
    runs = [(m, sd) for m in I3_MODELS for sd in I3_SEEDS] + [("tlstm", 0)]
    for m, sd in runs:
        for s in IRREG_SETTINGS:
            D = _irreg_per_patient(_i3_load(m, s, sd, PAT), PAT, P)
            V[(m, s, sd)] = {k: v[0] for k, v in _irreg_metrics(D, ONE).items()}; B[(m, s, sd)] = _irreg_metrics(D, W)
            ab, a1 = _i3_auc5(m, s, sd, P, case, ctrl, W, dev); V[(m, s, sd)]["auc5"] = a1; B[(m, s, sd)]["auc5"] = ab
            rows.append(dict(model=m, setting=s, seed=sd, **V[(m, s, sd)]))
        print("done", m, sd, flush=True)
    R = pd.DataFrame(rows); R.to_csv(RESULTS_POSTHOC / "irreg3_metrics.csv", index=False)
    T = []
    for met in ("mae", "ontime"):
        for b in ("rnn", "lstm", "no"):
            ch = lambda m, X: np.mean([X[(m, "S50", sd)][met] - X[(m, "S0", sd)][met] for sd in I3_SEEDS], 0)
            d = ch("lnn", B) - ch(b, B); pa, pb = ch("lnn", V), ch(b, V)
            T.append(dict(metric=met, other=b, change_glance=pa, change_other=pb, diff=pa - pb, lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5), p=paired_p(d)))
    T = pd.DataFrame(T); T["p_holm"] = holm(T.p.values); T.to_csv(RESULTS_POSTHOC / "irreg3_tests.csv", index=False)
    print("irregular_3seeds done:", len(R), "metric rows,", len(T), "tests")


QSTEPS = (0.5, 1.0, 0.25); QMODELS = ("lnn", "rnn", "lstm", "no"); QLAB = {"lnn": "LNN (GLANCE)", "rnn": "RNN", "lstm": "LSTM", "no": "NO"}


def _qgrid(step):
    return np.round(np.arange(0, 10 + 1e-9, step), 4)


@torch.no_grad()
def _qinfer(mod, name, x, step, dev):
    q = torch.tensor(_qgrid(step), dtype=torch.float32, device=dev)[None].expand(x.size(0), -1)
    if name in ("lnn", "no"):
        return mod(x, q).cpu().numpy()
    h = mod.input_projection(x); c = torch.zeros_like(h) if mod.mode == "LSTM" else None
    cur = mod.trajectory_projector(h); outs = [cur]
    for _ in range(q.size(1) - 1):
        if mod.mode == "LSTM":
            h, c = mod.rnn_cell(cur, (h, c))
        else:
            h = mod.rnn_cell(cur, h)
        cur = mod.trajectory_projector(h); outs.append(cur)
    return torch.cat(outs, 1).cpu().numpy()


def _qlookup(traj, step, t):
    g = _qgrid(step); i = np.searchsorted(g, np.asarray(t) + 1e-9) - 1
    return traj[..., np.clip(i, 0, len(g) - 1)]


def querystep():
    QOUT = RESULTS_POSTHOC / "querystep"; QOUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PAT = patients(); S = load_samples(); X = load_X(); R = rounds()
    P = [p for p in R[0][0] if not PAT[p]["late0"]]; tv = valid_times(PAT, P)
    TR = {m: {st: {s: {} for s in SEEDS} for st in QSTEPS} for m in QMODELS}; chk = {}
    for m in QMODELS:
        for s in SEEDS:
            for k in FOLDS:
                d = RESULTS_CV / "runs" / run_name(m, 1, "V4", k, s)
                z = np.load(d / "scaler.npz"); sc = MinMaxNorm(np.zeros(1), np.ones(1)); sc.min, sc.scale = z["data_min"], z["scale"]
                F = S[S.pid.isin(R[k][1]) & S["first"]].reset_index(drop=True); ds = V2Dataset(F, X, sc, 1)
                mod = build(m, ds.x.shape[1]).to(dev); mod.load_state_dict(torch.load(d / "best.pt", map_location=dev)["model_state_dict"]); mod.eval()
                x = ds.x.to(dev)
                for st in QSTEPS:
                    Y = _qinfer(mod, m, x, st, dev)
                    for i, p in enumerate(F.pid):
                        TR[m][st][s][p] = Y[i]
                sv = pd.read_csv(d / "predictions_heldout.csv"); sv = sv[sv.t <= 10 + 1e-9]
                err = max(float(np.max(np.abs(TR[m][0.5][s][p][np.round(q.t.values / 0.5).astype(int)] - q.y_pred.values)))
                          for p, q in sv[sv.t.isin(_qgrid(0.5))].groupby("pid"))
                chk[f"{m}_s{s}_f{k}"] = err
    json.dump(chk, open(QOUT / "check_reproduce_saved.json", "w"), indent=1)
    print("max |0.5-y rerun - saved| =", max(chk.values()), flush=True)

    def metr(traj, st, Pl, W=None):
        sabs, nv = [], []
        for p in Pl:
            d = PAT[p]; k = (d["t"] > 0) & (d["t"] <= 10 + 1e-9)
            e = np.abs(_qlookup(traj[p], st, d["t"][k]) - d["y"][k]); sabs.append(e.sum()); nv.append(k.sum())
        sabs, nv = np.array(sabs), np.array(nv)
        if W is not None:
            return (W @ sabs) / (W @ nv)
        out = dict(mae=sabs.sum() / nv.sum()); g = _qgrid(st)
        E = np.array([PAT[p]["E2"] for p in Pl]); lo = np.array([PAT[p]["E2_lo"] for p in Pl]); hi = np.array([PAT[p]["E2_T"] for p in Pl])
        pr = np.array([g[np.argmax(traj[p] >= 3.5)] if (traj[p] >= 3.5).any() else np.inf for p in Pl])
        det = E & np.isfinite(pr); ont = det & (pr >= lo - 1) & (pr <= hi + 1)
        err = np.where(pr < lo, lo - pr, np.where(pr > hi, pr - hi, 0.0))
        out.update(ontime=100 * ont.sum() / E.sum(), detected=100 * det.sum() / E.sum(), medabs=float(np.median(err[det])) if det.any() else np.nan)
        ONE = np.ones((1, len(Pl))); A = []
        for t in tv["E2"]:
            case, ctrl = case_ctrl(PAT, Pl, "E2", t)
            A.append(auc_w(np.array([_qlookup(traj[p], st, t) for p in Pl]), case, ctrl, ONE)[0])
        out["iauc"] = float(np.mean(A))
        return out

    rows = []
    for m in QMODELS:
        for st in QSTEPS:
            per = []
            for s in SEEDS:
                r = metr(TR[m][st][s], st, P)
                base = np.array([TR[m][0.5][s][p][::2] for p in P]); cur = np.array([_qlookup(TR[m][st][s][p], st, np.arange(0, 11.0)) for p in P])
                r.update(maxdiff=float(np.abs(cur - base).max()), meandiff=float(np.abs(cur - base).mean())); per.append(r)
            for met in per[0]:
                v = np.array([x[met] for x in per]); rows.append(dict(model=m, step=st, metric=met, mean=v.mean(), sd=v.std(ddof=1)))
    D = pd.DataFrame(rows); D.to_csv(QOUT / "querystep_metrics.csv", index=False)
    W = boot_weights(len(P)); SM = {m: {st: {p: np.mean([TR[m][st][s][p] for s in SEEDS], 0) for p in P} for st in QSTEPS} for m in QMODELS}
    ONEW = np.ones((1, len(P))); dl = {}
    for m in QMODELS:
        dl[m] = (metr(SM[m][1.0], 1.0, P, ONEW)[0] - metr(SM[m][0.5], 0.5, P, ONEW)[0], metr(SM[m][1.0], 1.0, P, W) - metr(SM[m][0.5], 0.5, P, W))
    T = pd.DataFrame([dict(a="lnn", b=b, change_a=dl["lnn"][0], change_b=dl[b][0], diff=dl["lnn"][0] - dl[b][0],
                           lo=np.percentile(dl["lnn"][1] - dl[b][1], 2.5), hi=np.percentile(dl["lnn"][1] - dl[b][1], 97.5), p=paired_p(dl["lnn"][1] - dl[b][1]))
                      for b in ("rnn", "lstm", "no")])
    T["p_holm"] = holm(T.p.values); T.to_csv(QOUT / "querystep_tests.csv", index=False)
    conv = [p for p in P if PAT[p]["E2"]]
    mae0 = {p: float(np.mean(np.abs(_qlookup(SM["lnn"][0.5][p], 0.5, PAT[p]["t"][(PAT[p]["t"] > 0) & (PAT[p]["t"] <= 10)]) - PAT[p]["y"][(PAT[p]["t"] > 0) & (PAT[p]["t"] <= 10)]))) for p in conv}
    order = sorted(conv, key=lambda p: (mae0[p], p)); ex = order[(len(order) - 1) // 2]
    plt.rcParams.update(PAPER_RCPARAMS)
    COLS = {0.5: ORANGE, 1.0: "#0072B2", 0.25: "#009E73"}; LSS = {0.5: "-", 1.0: "--", 0.25: ":"}
    fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.2), sharex=True, sharey=True)
    for ax, m, letter in zip(axs.ravel(), QMODELS, "abcd"):
        for st in QSTEPS:
            ax.plot(_qgrid(st), SM[m][st][ex], color=COLS[st], ls=LSS[st], lw=1.1, label=f"step {st:g} y")
        d = PAT[ex]; ax.plot(d["t"], d["y"], color="black", lw=0.6, marker="o", ms=2.3, clip_on=False, zorder=4, label="Observed grade")
        ax.axhline(4, color="#888888", ls=":", lw=0.7); ax.text(0.03, 0.985, f"{letter}) {QLAB[m]}", transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.set_xticks(range(0, 11, 2))
    for ax in axs[1]:
        ax.set_xlabel("Years from first visit")
    for ax in axs[:, 0]:
        ax.set_ylabel("Grade")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.53, 0.0), handlelength=1.6, columnspacing=0.8, labelspacing=0.25)
    fig.subplots_adjust(left=0.13, right=0.965, top=0.985, bottom=0.25, wspace=0.12, hspace=0.10)
    fig.savefig(QOUT / "fig_querystep_example.pdf"); fig.savefig(QOUT / "fig_querystep_example.png", dpi=300); plt.close(fig)
    json.dump(dict(example=ex, example_grades=list(zip(PAT[ex]["t"].tolist(), PAT[ex]["y"].tolist())), example_rule="median MAE (LNN, 0.5-y, seed mean) among CV E2 converters",
                   n_patients=len(P), n_conv=int(sum(PAT[p]["E2"] for p in P)), iauc_times=tv["E2"], reproduce_max=max(chk.values())),
              open(QOUT / "querystep_info.json", "w"), indent=1, default=float)
    print(D.pivot_table(index=["model", "step"], columns="metric", values="mean").round(3).to_string()); print(T.to_string()); print("example", ex)


OFFGRID_OUT = RESULTS_POSTHOC / "offgrid"
OG_DBINS = (("< 1 y", 0, 1), ("1-2 y", 1, 2), ("> 2 y", 2, np.inf))
OG_MONTH = np.round(np.arange(121) / 12, 6)


def _og_cohort():
    PAT = patients(); P = sorted(p for p in rounds()[0][0] if not PAT[p]["late0"])
    return PAT, P


def _og_batched(model, x, qs, bs=256):
    out = []
    for i in range(0, len(qs), bs):
        Q = qs[i:i + bs]; L = max(len(q) for q in Q)
        q = torch.tensor(np.stack([np.r_[q, np.full(L - len(q), q[-1])] for q in Q]), dtype=torch.float32)
        xx = x[i:i + bs] if x.dim() == 2 and x.size(0) == len(qs) else x
        with torch.no_grad():
            p = model(xx, q); p = p[0] if isinstance(p, tuple) else p
        out += [p[j, :len(Q[j])].numpy() for j in range(len(Q))]
    return out


def _og_onset_rows(alarm, PAT, P):
    R = []
    for p in P:
        d = PAT[p]; al = alarm[p]; E = d["E2"]; r = dict(E=E)
        if E:
            lo, hi = d["E2_lo"], d["E2_T"]; det = np.isfinite(al)
            r.update(det=det, ont=det and lo - 1 <= al <= hi + 1, dist=max(lo - al, 0, al - hi) if det else np.nan, late=det and al > hi + 1, fa=False)
        else:
            r.update(det=False, ont=False, dist=np.nan, late=False, fa=bool(al <= d["last"]))
        R.append(r)
    return pd.DataFrame(R)


def _og_onset_metrics(D, W):
    E = D.E.values.astype(bool); det = D.det.values.astype(bool); out = {}
    with np.errstate(all="ignore"):
        out["det"] = 100 * (W @ det) / (W @ E); out["ontime"] = 100 * (W @ D.ont.values.astype(bool)) / (W @ E)
        out["late"] = 100 * (W @ D.late.values.astype(bool)) / (W @ E); out["fa"] = 100 * (W @ D.fa.values.astype(bool)) / (W @ ~E)
    out["d"] = np.array([wmedian(D.dist.values[det], w[det]) for w in W])
    return out


def _og_grid_metrics(F, PAT, P, W):
    sabs, nv, down, risk, s5 = [], [], [], [], []; al = {}
    for p in P:
        gt, gp, vt, vp = F[p]; d = PAT[p]; k = vt > 0
        sabs.append(np.abs(vp[k] - d["y"][k]).sum()); nv.append(k.sum())
        g = (gt >= -1e-9) & (gt <= 10 + 1e-9); gg, gv = gt[g], gp[g]
        down.append(float(np.any(np.diff(gv) < -DROP_TOL)))
        tt = np.r_[gt[gt <= d["last"] + 1e-9], vt]; vv = np.r_[gp[gt <= d["last"] + 1e-9], vp]
        risk.append(np.max(vv)); s5.append(float(np.interp(5.0, gt, gp)))
        h = np.where(gv >= FIX_C)[0]; al[p] = gg[h[0]] if len(h) else np.inf
    sabs, nv, down = map(np.asarray, (sabs, nv, down)); out = {}
    with np.errstate(all="ignore"):
        out["mae"] = (W @ sabs) / (W @ nv); out["down"] = 100 * (W @ down) / W.sum(1)
    case, ctrl = case_ctrl(PAT, P, "E2", 5.0); out["auc5"] = auc_w(np.array(s5), case, ctrl, W)
    out["c"] = cindex_w([PAT[p]["E2_T"] for p in P], [PAT[p]["E2"] for p in P], np.array(risk), W)
    out.update(_og_onset_metrics(_og_onset_rows(al, PAT, P), W))
    return out, al


def offgrid_step0():
    OFFGRID_OUT.mkdir(parents=True, exist_ok=True)
    S = load_samples(); C = S[S.pid.isin(set(rounds()[0][0]))]; res = {}
    for name, sub in (("first_visit_trajectories", C[C["first"]]), ("all_starting_samples", C)):
        t = np.concatenate([a[a > 0] for a in sub.t]); i = np.isclose(t, np.round(t))
        res[name] = dict(patients=int(sub.pid.nunique()), samples=len(sub), followup_visits=len(t), integer_years=int(i.sum()),
                         half_years=int((~i).sum()), pct_integer=100 * i.mean(), pct_half=100 * (~i).mean())
    PAT, P = _og_cohort(); res["S50_dropped_per_seed"] = {}
    for sd in SEEDS:
        nk = nd = 0
        for k in FOLDS:
            H = round_data(k)["heldout_first"]; H = H[H.pid.isin(P)]; T = irreg_thin(H, 0.5, seed=sd)
            nk += sum((a > 0).sum() for a in T.t); nd += sum((a > 0).sum() for a in H.t) - sum((a > 0).sum() for a in T.t)
        res["S50_dropped_per_seed"][sd] = dict(kept_future=int(nk), dropped=int(nd))
    json.dump(res, open(OFFGRID_OUT / "step0.json", "w"), indent=1)
    print("offgrid_step0:", json.dumps(res, indent=1))


def offgrid_unseen_visits():
    OFFGRID_OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    t0 = time.time(); PAT, P = _og_cohort(); Ps = set(P)
    models = ["lnn", "no", "rnn", "lstm"] + (["tlstm"] if all((irreg_rdir("tlstm", "S50", k, s) / "done.json").exists() for k in FOLDS for s in SEEDS) else [])
    tl_seeds = [s for s in SEEDS if all((irreg_rdir("tlstm", "S50", k, s) / "done.json").exists() for k in FOLDS)]
    if "tlstm" not in models and tl_seeds:
        models.append("tlstm")
    RD = {k: round_data(k) for k in FOLDS}
    pts, chk = [], []
    for m in models:
        for sd in (SEEDS if m != "tlstm" else tl_seeds):
            for k in FOLDS:
                D = RD[k]; full = D["heldout_first"]; full = full[full.pid.isin(Ps)].reset_index(drop=True); kept = irreg_thin(full, 0.5, seed=sd)
                sv = pd.read_csv(irreg_rdir(m, "S50", k, sd) / "predictions_heldout.csv"); sv = {p: q.sort_values("t") for p, q in sv.groupby("pid")}
                ds = V2Dataset(kept, D["X"], D["scaler"], 1)
                direct = m in ("lnn", "no", "tlstm")
                if direct:
                    mod = irreg_build(m, 1025); mod.load_state_dict(torch.load(irreg_rdir(m, "S50", k, sd) / "best.pt", map_location=dev)["model_state_dict"]); mod.eval()
                xs, qs, meta = [], [], []
                for i, r in kept.iterrows():
                    ft, fy = full.loc[i, "t"], full.loc[i, "y"]; kt = r.t
                    q = sv[r.pid]; assert np.allclose(q.t.values, kt), (m, sd, r.pid)
                    kp = q.y_pred.values
                    for tt, pp, yy in zip(kt, kp, r.y):
                        pts.append(dict(model=m, seed=sd, pid=r.pid, t=float(tt), y_true=float(yy), kind="kept", interp=pp, carry=pp, direct=pp, dist=0.0))
                    drop = [(tt, yy) for tt, yy in zip(ft, fy) if not np.any(np.isclose(kt, tt))]
                    for tt, yy in drop:
                        cf = kp[np.where(kt <= tt + 1e-9)[0][-1]]
                        pts.append(dict(model=m, seed=sd, pid=r.pid, t=float(tt), y_true=float(yy), kind="dropped", interp=float(np.interp(tt, kt, kp)),
                                        carry=float(cf), direct=np.nan, dist=float(np.min(np.abs(kt - tt)))))
                        if direct:
                            xs.append(ds.x[i]); qs.append(np.sort(np.r_[kt, tt]).astype(np.float32)); meta.append((len(pts) - 1, tt))
                    if direct and i == 0:
                        p0 = _og_batched(mod, ds.x[i][None], [kt.astype(np.float32)])[0]; chk.append(dict(model=m, seed=sd, fold=k, max_abs=float(np.max(np.abs(p0 - kp)))))
                if direct and qs:
                    outs = _og_batched(mod, torch.stack(xs), qs)
                    for (j, tt), o, q in zip(meta, outs, qs):
                        pts[j]["direct"] = float(o[int(np.argmin(np.abs(q - tt)))])
            print(f"offgrid_unseen_visits {m} seed {sd} done ({time.time() - t0:.0f}s)", flush=True)
    X = pd.DataFrame(pts); X.to_csv(OFFGRID_OUT / "unseen_points.csv.gz", index=False)
    C = pd.DataFrame(chk); print("kept-visit reproduction max |diff|:", C.max_abs.max(), flush=True); C.to_csv(OFFGRID_OUT / "unseen_repro_check.csv", index=False)
    W = boot_weights(len(P)); ONE = np.ones((1, len(P))); idx = {p: i for i, p in enumerate(P)}
    rows, B = [], {}
    for (m, sd), g in X.groupby(["model", "seed"]):
        for meth in (("direct", "interp", "carry", "kept") if m in ("lnn", "no", "tlstm") else ("interp", "carry", "kept")):
            dr = g[g.kind == "dropped"]; col = "interp" if meth == "kept" else meth
            res = {}
            if meth != "kept":
                e = np.abs(dr[col].values - dr.y_true.values); pi = dr.pid.map(idx).values
                for name, sel in [("mae_drop", np.ones(len(dr), bool))] + [(f"mae_{b}", (dr.dist.values >= lo) & (dr.dist.values < hi)) for b, lo, hi in OG_DBINS] + \
                                 [("mae_prog", dr.pid.map(lambda p: PAT[p]["E2"]).values.astype(bool)), ("mae_stable", ~dr.pid.map(lambda p: PAT[p]["E2"]).values.astype(bool))]:
                    sa = np.bincount(pi[sel], e[sel], len(P)); na = np.bincount(pi[sel], None, len(P)).astype(float)
                    with np.errstate(all="ignore"):
                        res[name] = ((W @ sa) / (W @ na), ((ONE @ sa) / (ONE @ na))[0], int(sel.sum()))
            al = {}
            for p, q in g.groupby("pid"):
                q = q if meth != "kept" else q[q.kind == "kept"]
                q = q.sort_values("t"); v = q[col].values; h = np.where(v >= FIX_C)[0]
                al[p] = q.t.values[h[0]] if len(h) else np.inf
            Dd = _og_onset_rows(al, PAT, P); ob, op = _og_onset_metrics(Dd, W), _og_onset_metrics(Dd, ONE)
            for k in ob:
                res[k] = (ob[k], op[k][0], int(sum(PAT[p]["E2"] for p in P)))
            B[(m, meth, sd)] = {k: v[0] for k, v in res.items()}
            rows.append(dict(model=m, method=meth, seed=sd, **{k: v[1] for k, v in res.items()}, **{f"n_{k}": v[2] for k, v in res.items() if k.startswith("mae")}))
    R = pd.DataFrame(rows); R.to_csv(OFFGRID_OUT / "unseen_per_seed.csv", index=False)
    others = [("no", "direct"), ("rnn", "interp"), ("lstm", "interp")] + ([("tlstm", "direct")] if len(tl_seeds) == 3 else [])
    T = []
    for met in ("mae_drop", "late"):
        a = np.mean([B[("lnn", "direct", s)][met] for s in SEEDS], 0)
        for m, meth in others:
            b = np.mean([B[(m, meth, s)][met] for s in SEEDS], 0); d = a - b
            va = R[(R.model == "lnn") & (R.method == "direct")][met].mean(); vb = R[(R.model == m) & (R.method == meth)][met].mean()
            T.append(dict(metric=met, other=m, other_method=meth, glance=va, other_value=vb, diff=va - vb, lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5), p=paired_p(d)))
    T = pd.DataFrame(T); T["p_holm"] = holm(T.p.values); T.to_csv(OFFGRID_OUT / "unseen_tests.csv", index=False)
    json.dump(dict(models=models, tlstm_seeds=tl_seeds, seconds=time.time() - t0), open(OFFGRID_OUT / "unseen_info.json", "w"), indent=1)
    print(R.groupby(["model", "method"])[["mae_drop", "late", "ontime"]].mean().round(3).to_string()); print(T.round(4).to_string())


def offgrid_annual_grid():
    OFFGRID_OUT.mkdir(parents=True, exist_ok=True)
    PAT, P = _og_cohort(); ONE = np.ones((1, len(P))); rows = []
    for m in ("rnn", "lstm"):
        for sd in SEEDS:
            if all((annual_rdir(m, k, sd) / "done.json").exists() for k in FOLDS):
                A = pd.concat([pd.read_csv(annual_rdir(m, k, sd) / "predictions_heldout.csv") for k in FOLDS]); F = {}
                for p, q in A.groupby("pid"):
                    if p in PAT:
                        gt, gp = q.t.values, q.y_pred.values; assert np.allclose(gt, np.arange(len(gt)))
                        F[p] = (gt, gp, PAT[p]["t"], np.interp(PAT[p]["t"], gt, gp))
                o, _ = _og_grid_metrics(F, PAT, P, ONE)
                ep = [json.load(open(annual_rdir(m, k, sd) / "done.json"))["n_epochs_best"] for k in FOLDS]
                cf = [json.load(open(annual_rdir(m, k, sd) / "config.json")) for k in FOLDS]
                rows.append(dict(model=m, grid="annual", seed=sd, **{k: v[0] for k, v in o.items()}, best_epochs=";".join(map(str, ep)),
                                 train_targets_all=sum(c["train_future_visits_all"] for c in cf), train_targets_integer=sum(c["train_future_visits_integer"] for c in cf)))
            Fs = {}
            for k in FOLDS:
                f = forecasts_from_csv(RESULTS_CV / "runs" / run_name(m, 1, "V4", k, sd) / "predictions_heldout.csv", PAT)
                Fs.update({p: (v["gt"], v["gp"], v["vt"], v["vp"]) for p, v in f.items()})
            o, _ = _og_grid_metrics(Fs, PAT, P, ONE)
            rows.append(dict(model=m, grid="6-month", seed=sd, **{k: v[0] for k, v in o.items()}))
    R = pd.DataFrame(rows); R.to_csv(OFFGRID_OUT / "annual_per_seed.csv", index=False)
    M = pd.read_csv(RESULTS_EVAL / "cv" / "metrics_cv.csv"); chk = {}
    for m in ("rnn", "lstm"):
        for mine, theirs, sc in (("mae", "mae", 1), ("auc5", "auc5_E2", 1), ("down", "down", 100), ("ontime", "ontime_E2", 100), ("fa", "fa_E2", 100)):
            ref = np.array([float(x) for x in M[(M.model == f"{m}_grade_V4") & (M.metric == theirs)].per_seed.iloc[0].split(";")]) * sc
            chk[f"{m}_{mine}"] = float(np.max(np.abs(R[(R.model == m) & (R.grid == "6-month")].sort_values("seed")[mine].values - ref)))
    json.dump(chk, open(OFFGRID_OUT / "annual_check.json", "w"), indent=1); print("6-month check vs evaluation/cv/metrics_cv.csv:", chk)
    print(R.groupby(["model", "grid"]).mean(numeric_only=True).round(3).to_string())


def offgrid_monthly():
    OFFGRID_OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); PAT, P = _og_cohort(); Ps = set(P); W = boot_weights(len(P)); ONE = np.ones((1, len(P)))
    rows, B, chk = [], {}, []
    for m in ("lnn", "no"):
        for sd in SEEDS:
            Fm, F6 = {}, {}
            for k in FOLDS:
                rd = RESULTS_CV / "runs" / run_name(m, 1, "V4", k, sd); D = round_data(k)
                z = np.load(rd / "scaler.npz"); assert np.allclose(z["data_min"], D["scaler"].min) and np.allclose(z["scale"], D["scaler"].scale)
                H = D["heldout_first"]; H = H[H.pid.isin(Ps)].reset_index(drop=True); ds = V2Dataset(H, D["X"], D["scaler"], 1)
                mod = build(m, 1025); mod.load_state_dict(torch.load(rd / "best.pt", map_location="cpu")["model_state_dict"]); mod.eval()
                with torch.no_grad():
                    p = mod(ds.x, torch.tensor(OG_MONTH, dtype=torch.float32)[None].expand(len(H), -1)); p = (p[0] if isinstance(p, tuple) else p).numpy()
                f6 = forecasts_from_csv(rd / "predictions_heldout.csv", PAT, set(H.pid))
                for i, r in H.iterrows():
                    f = f6[r.pid]
                    vt = PAT[r.pid]["t"]; vm = np.interp(vt, OG_MONTH, p[i])
                    Fm[r.pid] = (OG_MONTH, p[i], vt, np.where(vt <= 10, vm, f["vp"]))
                    F6[r.pid] = (f["gt"], f["gp"], f["vt"], f["vp"])
                    j = np.isin(np.round(OG_MONTH, 4), np.round(G, 4)); chk.append(dict(model=m, seed=sd, pid=r.pid, max_abs_6m_points=float(np.max(np.abs(p[i][j] - at(f, G))))))
            for res, F in (("monthly", Fm), ("6-month", F6)):
                ob, al = _og_grid_metrics(F, PAT, P, W); op, _ = _og_grid_metrics(F, PAT, P, ONE)
                rr = []
                for p in P:
                    d = PAT[p]
                    if d["E2"] and np.isfinite(al[p]) and al[p] > 0:
                        gt, gp = F[p][0], F[p][1]; k = int(np.argmin(np.abs(gt - al[p])))
                        yE = d["y"][int(np.argmin(np.abs(d["t"] - d["E2_T"])))]
                        rr.append(dict(pid=p, pred=(gp[k] - gp[0]) / al[p], obs=(yE - d["y"][0]) / d["E2_T"]))
                rr = pd.DataFrame(rr); ae = (rr.pred - rr.obs).abs().values; pi = rr.pid.map({p: i for i, p in enumerate(P)}).values
                ob["rate_abserr"] = np.array([wmedian(ae, w[pi]) for w in W])
                B[(m, res, sd)] = ob
                rows.append(dict(model=m, resolution=res, seed=sd, **{k: v[0] for k, v in op.items()}, rate_pred_median=float(rr.pred.median()),
                                 rate_obs_median=float(rr.obs.median()), rate_abserr_median=float(np.median(ae)), n_rate=len(rr)))
            print(f"offgrid_monthly {m} seed {sd} ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OFFGRID_OUT / "monthly_per_seed.csv", index=False)
    Cc = pd.DataFrame(chk); Cc.groupby("model").max_abs_6m_points.describe().to_csv(OFFGRID_OUT / "monthly_check_6m_points.csv")
    T = []
    for met in ("det", "ontime", "d", "late", "fa", "rate_abserr"):
        a = np.mean([B[("lnn", "monthly", s)][met] for s in SEEDS], 0); b = np.mean([B[("no", "monthly", s)][met] for s in SEEDS], 0); d = a - b
        va = R[(R.model == "lnn") & (R.resolution == "monthly")][met if met != "rate_abserr" else "rate_abserr_median"].mean()
        vb = R[(R.model == "no") & (R.resolution == "monthly")][met if met != "rate_abserr" else "rate_abserr_median"].mean()
        T.append(dict(metric=met, glance=va, no=vb, diff=va - vb, lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5), p=paired_p(d)))
    T = pd.DataFrame(T); T["p_holm"] = holm(T.p.values); T.to_csv(OFFGRID_OUT / "monthly_tests.csv", index=False)
    print(R.groupby(["model", "resolution"]).mean(numeric_only=True).round(3).to_string()); print(T.round(4).to_string())
    print("6-month points of the monthly query vs saved 6-month forecasts (max |diff|):", Cc.groupby("model").max_abs_6m_points.max().to_dict())


PM_MODELS = ["fg", "rnn", "lstm", "no", "lnn"]
PM_HB = (("0-2 y", 0, 2), ("2-5 y", 2, 5), ("5-10 y", 5, 10))
PM_NEVER = 10.5


def _pm_theil_sen(t, y):
    i, j = np.triu_indices(len(t), 1)
    return float(np.median((y[j] - y[i]) / (t[j] - t[i])))


def _pm_per_patient(F, PAT, P):
    R = []
    for p in P:
        d = PAT[p]; f = F[p]; t, y, yh = d["t"], d["y"], f["vp"]
        assert np.allclose(f["vt"], t)
        r = dict(E=d["E2"], T=d["E2_T"], b=_pm_theil_sen(t, y), bh=_pm_theil_sen(t, yh))
        k = t > 0; y0 = y[0]; ch = k & (y != y0)
        r["sabs_ch"], r["n_ch"] = np.abs(yh[ch] - y[ch]).sum(), ch.sum()
        obs, pred = (y >= y0 + 1), (yh >= y0 + 0.5)
        for name, sel in [("all", k)] + [(b, k & (t > lo) & (t <= hi)) for b, lo, hi in PM_HB]:
            r[f"tp_{name}"] = int((sel & obs & pred).sum()); r[f"fn_{name}"] = int((sel & obs & ~pred).sum())
            r[f"tn_{name}"] = int((sel & ~obs & ~pred).sum()); r[f"fp_{name}"] = int((sel & ~obs & pred).sum())
        g = at(f, G); h = np.where(g >= FIX_C)[0]; r["That"] = G[h[0]] if len(h) else PM_NEVER
        R.append(r)
    return pd.DataFrame(R)


def _pm_metrics(D, W):
    E = D.E.values.astype(bool); err = D.bh.values - D.b.values; out = {}
    with np.errstate(all="ignore"):
        for g, sel in (("all", np.ones(len(D), bool)), ("prog", E), ("stable", ~E)):
            out[f"m1_abs_{g}"] = (W[:, sel] @ np.abs(err[sel])) / W[:, sel].sum(1)
            out[f"m1_bias_{g}"] = (W[:, sel] @ err[sel]) / W[:, sel].sum(1)
        out["m2a"] = (W @ D.sabs_ch.values) / (W @ D.n_ch.values)
        for h in ["all"] + [b for b, _, _ in PM_HB]:
            tp, fn, tn, fp = (W @ D[f"{c}_{h}"].values for c in ("tp", "fn", "tn", "fp"))
            sens, spec = tp / (tp + fn), tn / (tn + fp)
            out[f"m2b_sens_{h}"], out[f"m2b_spec_{h}"], out[f"m2b_ba_{h}"] = sens, spec, (sens + spec) / 2
    out["m3_c"] = cindex_w(D["T"].values, E, -D.That.values, W)
    return out


def _pm_forecasts(m, part, PAT, P):
    if m == "fg":
        f = {p: dict(vt=PAT[p]["t"], vp=np.full(len(PAT[p]["t"]), PAT[p]["g0"]), gt=G, gp=np.full(len(G), PAT[p]["g0"])) for p in P}
        return {0: f}
    if part == "cv":
        F, _ = cv_forecasts(f"{m}_grade_V4", PAT); return {s: {p: F[s][p] for p in P} for s in SEEDS}
    return {s: forecasts_from_csv(RESULTS_FINAL / "runs" / run_name(m, 1, "V4", 0, s) / "predictions_test.csv", PAT, set(P)) for s in SEEDS}


def _pm_run(part, PAT, P):
    W = boot_weights(len(P)); ONE = np.ones((1, len(P))); rows, B = [], {}
    for m in PM_MODELS:
        Fs = _pm_forecasts(m, part, PAT, P); bs = []
        for s, F in Fs.items():
            D = _pm_per_patient(F, PAT, P); pt = _pm_metrics(D, ONE); bs.append(_pm_metrics(D, W))
            rows.append(dict(part=part, model=m, seed=s, n_patients=len(P), n_prog=int(D.E.sum()), n_changed_visits=int(D.n_ch.sum()),
                             n_increase_visits=int(D.tp_all.sum() + D.fn_all.sum()), **{k: v[0] for k, v in pt.items()}))
        B[m] = {k: np.nanmean([b[k] for b in bs], 0) for k in bs[0]}
    R = pd.DataFrame(rows); R.to_csv(RESULTS_POSTHOC / f"progression_per_seed_{part}.csv", index=False)
    T = []
    for met in ("m1_abs_prog", "m2a", "m2b_ba_all", "m3_c"):
        for o in ("rnn", "lstm", "no"):
            d = B["lnn"][met] - B[o][met]
            va, vb = R[R.model == "lnn"][met].mean(), R[R.model == o][met].mean()
            T.append(dict(part=part, metric=met, other=o, glance=va, other_value=vb, diff=va - vb, lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5), p=paired_p(d)))
    T = pd.DataFrame(T); T["p_holm"] = holm(T.p.values); T.to_csv(RESULTS_POSTHOC / f"progression_tests_{part}.csv", index=False)
    print(f"progression_metrics {part}: {len(R)} rows, {len(T)} tests")


def progression_metrics():
    PAT = patients()
    Pcv = sorted(p for p in rounds()[0][0] if not PAT[p]["late0"]); Pte = sorted(p for p in test_pids() if not PAT[p]["late0"])
    for part, P in (("cv", Pcv), ("test", Pte)):
        _pm_run(part, PAT, P)
    print("progression_metrics done")


CONT_OUT = RESULTS_POSTHOC / "continuous"
CONT_SCHED = {"1/12": 1 / 12, "1/4": 0.25, "1/2": 0.5, "1": 1.0}
CONT_SNAMES = list(CONT_SCHED) + ["visits"]
CONT_LAB = {"rs": "GLANCE-RS-V4", "con": "GLANCE-CON-V4", "ltc": "GLANCE-LTC-V4", "lnn": "GLANCE-V4", "no": "NO-V4"}
CONT_MEAN_MAX = (0.05, 0.30)


def _cont_ckpt(key, fold, seed):
    return RESULTS_CV / "runs" / run_name(key, 1, "V4", fold, seed) if key in ("lnn", "no") else continuous_rdir(key, fold, seed)


def _cont_load(key, d, dev):
    mod = (build(key, 1025) if key in ("lnn", "no") else continuous_build(key, 1025)).to(dev)
    mod.load_state_dict(torch.load(d / "best.pt", map_location=dev)["model_state_dict"]); mod.eval()
    return mod


def _cont_schedules(mod, ds):
    n = len(ds); out = {p: {} for p in ds.s.pid}
    for nm, st in CONT_SCHED.items():
        grid = continuous_pure_grid(st); P = _og_batched(mod, ds.x, [grid] * n)
        for p, v in zip(ds.s.pid, P):
            out[p][nm] = (grid, v)
    vt = [np.asarray(t, np.float32) for t in ds.s.t]; P = _og_batched(mod, ds.x, vt)
    for p, t, v in zip(ds.s.pid, vt, P):
        out[p]["visits"] = (np.round(t.astype(float), 6), v)
    return out


def _cont_pair_diffs(F, pids, a, b):
    dd = []
    for p in pids:
        ta, va = F[p][a]; tb, vb = F[p][b]
        ka = np.round(ta, 6) <= 10 + 1e-9; kb = np.round(tb, 6) <= 10 + 1e-9
        sa = {round(float(t), 5): float(v) for t, v in zip(ta[ka], va[ka])}; sb = {round(float(t), 5): float(v) for t, v in zip(tb[kb], vb[kb])}
        sh = sorted(set(sa) & set(sb)); dd.append(np.abs(np.array([sa[t] for t in sh]) - np.array([sb[t] for t in sh])))
    dd = np.concatenate(dd)
    return float(dd.mean()), float(dd.max()), len(dd), int((dd > 0.30).sum())


def continuous_consistency():
    CONT_OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu"); PAT, P = _og_cohort(); S = load_samples(); X = load_X(); rows = []
    for key in ("rs", "con", "ltc", "lnn", "no"):
        for seed in SEEDS:
            F = {}
            for k in FOLDS:
                d = _cont_ckpt(key, k, seed)
                if not (d / "done.json").exists():
                    continue
                mod = _cont_load(key, d, dev)
                Fk = S[S.pid.isin(set(rounds()[k][1]) & set(P)) & S["first"]].reset_index(drop=True)
                z = np.load(d / "scaler.npz"); sc = MinMaxNorm(np.zeros(1), np.ones(1)); sc.min, sc.scale = z["data_min"], z["scale"]
                F.update(_cont_schedules(mod, V2Dataset(Fk, X, sc, 1)))
            if not F:
                continue
            pids = [p for p in P if p in F]
            for i, a in enumerate(CONT_SNAMES):
                for b in CONT_SNAMES[i + 1:]:
                    mu, mx, n, n30 = _cont_pair_diffs(F, pids, a, b)
                    rows.append(dict(id=CONT_LAB[key], seed=seed, a=a, b=b, mean_absdiff=mu, max_absdiff=mx, n_points=n, n_gt030=n30))
    C = pd.DataFrame(rows); C.to_csv(CONT_OUT / "consistency.csv", index=False)
    verdict = {}
    for key in ("rs", "con", "ltc"):
        g = C[C.id == CONT_LAB[key]]
        if not len(g):
            verdict[CONT_LAB[key]] = "not trained"; continue
        res = {}
        for o in ("1/12", "1/4", "1", "visits"):
            r = g[((g.a == o) & (g.b == "1/2")) | ((g.a == "1/2") & (g.b == o))]
            res[o] = dict(mean=float(r.groupby("seed").mean_absdiff.first().mean()), max=float(r.max_absdiff.max()))
        verdict[CONT_LAB[key]] = all(v["mean"] <= CONT_MEAN_MAX[0] and v["max"] <= CONT_MEAN_MAX[1] for v in res.values())
    json.dump(verdict, open(CONT_OUT / "consistency_verdict.json", "w"), indent=1)
    print("continuous_consistency:", json.dumps(verdict, indent=1))
    print(C.groupby("id")[["mean_absdiff", "max_absdiff"]].mean().round(3).to_string())


def continuous_unseen_visits():
    CONT_OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu"); t0 = time.time(); PAT, P = _og_cohort(); Ps = set(P); pts = []
    for v in ("rs", "con", "ltc"):
        if not all((continuous_rdir(v, k, s) / "done.json").exists() for k in FOLDS for s in SEEDS):
            print(f"continuous_unseen_visits: {v} not fully trained yet, skipped"); continue
        for sd in SEEDS:
            for k in FOLDS:
                D = round_data(k); full = D["heldout_first"]; full = full[full.pid.isin(Ps)].reset_index(drop=True); kept = irreg_thin(full, 0.5, seed=sd)
                sv = pd.read_csv(continuous_rdir(v, k, sd) / "predictions_heldout.csv"); sv = sv[sv.is_visit]
                sv = {p: q.sort_values("t") for p, q in sv.groupby("pid")}
                ds = V2Dataset(kept, D["X"], D["scaler"], 1)
                mod = continuous_build(v, 1025); mod.load_state_dict(torch.load(continuous_rdir(v, k, sd) / "best.pt", map_location=dev)["model_state_dict"]); mod.eval()
                xs, qs, meta = [], [], []
                for i, r in kept.iterrows():
                    ft, fy = full.loc[i, "t"], full.loc[i, "y"]; kt = r.t
                    q = sv[r.pid]; assert np.allclose(q.t.values, kt), (v, sd, r.pid)
                    kp = q.y_pred.values
                    for tt, pp, yy in zip(kt, kp, r.y):
                        pts.append(dict(model=v, seed=sd, pid=r.pid, t=float(tt), y_true=float(yy), kind="kept", direct=pp))
                    drop = [(tt, yy) for tt, yy in zip(ft, fy) if not np.any(np.isclose(kt, tt))]
                    for tt, yy in drop:
                        pts.append(dict(model=v, seed=sd, pid=r.pid, t=float(tt), y_true=float(yy), kind="dropped", direct=np.nan))
                        xs.append(ds.x[i]); qs.append(np.sort(np.r_[kt, tt]).astype(np.float32)); meta.append((len(pts) - 1, tt))
                if qs:
                    outs = _og_batched(mod, torch.stack(xs), qs)
                    for (j, tt), o, q in zip(meta, outs, qs):
                        pts[j]["direct"] = float(o[int(np.argmin(np.abs(q - tt)))])
            print(f"continuous_unseen_visits {v} seed {sd} done ({time.time() - t0:.0f}s)", flush=True)
    if not pts:
        print("continuous_unseen_visits: nothing trained yet"); return
    X = pd.DataFrame(pts); X.to_csv(CONT_OUT / "unseen_points.csv.gz", index=False)
    W = boot_weights(len(P)); ONE = np.ones((1, len(P))); idx = {p: i for i, p in enumerate(P)}
    rows = []
    for (m, sd), g in X.groupby(["model", "seed"]):
        dr = g[g.kind == "dropped"]; e = np.abs(dr.direct.values - dr.y_true.values); pi = dr.pid.map(idx).values
        sa = np.bincount(pi, e, len(P)); na = np.bincount(pi, None, len(P)).astype(float)
        with np.errstate(all="ignore"):
            mae_drop_pt = ((ONE @ sa) / (ONE @ na))[0]
        al = {}
        for p, q in g.groupby("pid"):
            q = q.sort_values("t"); yp = q.direct.values; h = np.where(yp >= FIX_C)[0]; al[p] = q.t.values[h[0]] if len(h) else np.inf
        Dd = _og_onset_rows(al, PAT, P); ob = _og_onset_metrics(Dd, ONE)
        rows.append(dict(model=m, seed=sd, mae_drop=float(mae_drop_pt), n_dropped=int(len(dr)), **{k: float(v2[0]) for k, v2 in ob.items()}))
    R = pd.DataFrame(rows); R.to_csv(CONT_OUT / "unseen_per_seed.csv", index=False)
    print(R.groupby("model")[["mae_drop", "det", "ontime", "late"]].mean().round(3).to_string())


CRIT_PERF = [("auc5", "AUC5y", +1), ("c", "C", +1), ("mae", "MAE", -1), ("det", "%det", +1), ("ontime", "%on time", +1),
             ("late", "%late", -1), ("m3", "M3", +1)]
CRIT_COMP = ("G6-GLANCE-V4", "G6-NO-V4")


def _crit_boot(PP, PAT, P, pairs, mets, W):
    rows = []
    for a, b in pairs:
        for met in mets:
            def ms(mid, WW):
                return np.mean([ar_agg(D, WW, PAT, P).get(met, np.full(WW.shape[0], np.nan)) for D in PP[mid].values()], 0)
            ONE = np.ones((1, len(P))); va, vb = ms(a, ONE)[0], ms(b, ONE)[0]; d = ms(a, W) - ms(b, W)
            rows.append(dict(a=a, b=b, metric=met, value_a=va, value_b=vb, diff=va - vb, lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5), p=paired_p(d)))
    T = pd.DataFrame(rows); T["p_holm"] = holm(T.p.values)
    return T


def continuous_criteria():
    CONT_OUT.mkdir(parents=True, exist_ok=True)
    cv_path = ALLRESULTS_OUT / "cv_matrix_long.csv"
    if not cv_path.exists():
        print("continuous_criteria: run scripts.allresults first"); return
    CV = pd.read_csv(cv_path); PK = pickle.load(open(ALLRESULTS_OUT / "per_patient.pkl", "rb")); PP, Pc = PK["PPc"], PK["Pc"]
    PAT = patients(); W = boot_weights(len(Pc))
    get = lambda mid, met: CV[(CV.id == mid) & (CV.metric == met)]
    fams = {}
    for v in ("rs", "con", "ltc"):
        mid = f"G6-{CONT_CNAME[v]}-V4"
        if not len(get(mid, "mae")):
            print(f"continuous_criteria: {mid} not trained yet, skipped"); continue
        rows = []
        for met, lab, sgn in CRIT_PERF:
            a, b = get(mid, met).iloc[0], get("G6-GLANCE-V4", met).iloc[0]
            sa, sb = (0 if not np.isfinite(x) else x for x in (a["sd"], b["sd"])); pooled = np.sqrt((sa ** 2 + sb ** 2) / 2)
            worse_by = -sgn * (a["mean"] - b["mean"]); worse = bool(worse_by > 2 * pooled) if pooled > 0 else bool(worse_by > 0)
            better = bool(-worse_by > 2 * pooled) if pooled > 0 else bool(-worse_by > 0)
            rows.append(dict(endpoint=lab, variant=a["mean"], glance=b["mean"], diff=a["mean"] - b["mean"], two_pooled_sd=2 * pooled,
                             verdict="worse" if worse else ("better" if better else "equal")))
        P = pd.DataFrame(rows); P.to_csv(CONT_OUT / f"performance_criterion_{v}.csv", index=False)
        T = _crit_boot(PP, PAT, Pc, [(mid, b) for b in CRIT_COMP], [m for m, _, _ in CRIT_PERF], W)
        T["family"] = f"{mid} vs GLANCE V4 / NO V4"; fams[v] = T
        nw = int((P.verdict == "worse").sum())
        print(f"{mid}: {'PASS' if nw == 0 else 'FAIL'} ({nw} of {len(CRIT_PERF)} endpoints worse beyond seed noise)")
    if fams:
        pd.concat(fams.values()).to_csv(CONT_OUT / "tests_criteria.csv", index=False)
    print("continuous_criteria done:", list(fams))


def main():
    cmd = sys.argv[1]
    {"onset": onset, "onset_class": onset_class, "late_per_seed": late_per_seed, "onset_by_grade": onset_by_grade,
     "figs": figs, "bygrade": bygrade, "ablation_extra": ablation_extra, "explain59": explain59,
     "irregular": irregular, "irregular_3seeds": irregular_3seeds, "querystep": querystep,
     "offgrid_step0": offgrid_step0, "offgrid_unseen_visits": offgrid_unseen_visits,
     "offgrid_annual_grid": offgrid_annual_grid, "offgrid_monthly": offgrid_monthly,
     "progression_metrics": progression_metrics, "continuous_consistency": continuous_consistency,
     "continuous_unseen_visits": continuous_unseen_visits, "continuous_criteria": continuous_criteria}[cmd]()


if __name__ == "__main__":
    main()
