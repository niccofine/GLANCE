"""Extra posthoc metrics.
  compute --set cv|test   per model x seed: existing + new metrics, one paired-bootstrap test family (GLANCE V4 vs RNN / LSTM / NO V4 on 4 new endpoints, Holm over 12) -> results/posthoc/extrametrics/{cv,test}_long.csv, f5_{cv,test}_per_seed.csv, tests_{cv,test}.csv, run_info_{cv,test}.json
  table1                  NG-GLANCE-V4 vs G6-RNN-V4, G6-LSTM-V4 (fixed grid) and NG-NO-V4 (no grid), CV only -> results/posthoc/extrametrics/tests_table1.csv (--nboot N for a different resample count)
Run: python -m scripts.extrametrics compute --set cv
     python -m scripts.extrametrics compute --set test
     python -m scripts.extrametrics table1 [--nboot 10000]
"""
import json
import pickle
import sys
import time
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from glance.config import RESULTS_FINAL, RESULTS_EVAL, RESULTS_POSTHOC
from glance.data import SEEDS, FOLDS, rounds, test_pids, load_samples, load_X, V2Dataset
from glance.evaluate import patients, case_ctrl, auc_w, boot_weights, paired_p, holm, cv_seeds, _cv_run_dir
from glance.models import run_name
from glance.irregular import SETTINGS as IRREG_SETTINGS, build as irreg_build, thin as irreg_thin, rdir as irreg_rdir
from glance.annual import rdir as annual_rdir
from glance.continuous import rdir as continuous_rdir
from scripts.allresults import OUT as ALLRESULTS_OUT, registry as ar_registry, fixed_recs, annual_recs, nogrid_recs, fg_recs, cox_recs, per_patient, agg, mean_sd, scaler_of, five, HB, NAME, CNAME
from scripts.finaltraining import extra_rdir

OUT = RESULTS_POSTHOC / "extrametrics"
EXIST = ["auc5", "c", "mae", "m2a", "down", "det", "ontime", "early", "late", "missed", "fa", "m3"]
NEW = ["bal_mae"] + [f"mae_{h[0]}" for h in HB] + [f"f2_{a}_{g}" for a in ("abs", "bias") for g in ("all", "prog", "stable")] + \
      [f"f3_sp_{g}" for g in ("all", "prog")] + ["f4_auc"]
TESTED = ["bal_mae", "f2_abs_prog", "f3_sp_prog", "f4_auc"]
GL, OTHERS = "G6-GLANCE-V4", ("G6-RNN-V4", "G6-LSTM-V4", "G6-NO-V4")
EXCLUDE = {"G6-GLANCE-CON-V4", "G6-GLANCE-LTC-V4"}


def reg():
    R = {r["id"]: dict(r) for r in ar_registry() if r["id"] not in EXCLUDE}
    for m in ("rnn", "lstm", "no"):
        R[f"G6-{NAME[m]}-V0-3S"] = dict(id=f"G6-{NAME[m]}-V0-3S", kind="fixed", key=f"{m}_grade_V0", force3=True, test=True)
    for k in ("G12-RNN-V4", "G12-LSTM-V4") + tuple(f"NG-{NAME[m]}-V4" for m in ("rnn", "lstm", "no", "lnn", "tlstm")):
        R[k]["test"] = True
    groups = [("References", ["FG", "COX", "LR", "SVM"]),
              ("Fixed 6-month grid", [f"G6-{b}-{v}" for b in ("RNN", "LSTM", "NO") for v in ("V0", "V0-3S", "V1", "V4")] +
               ["G6-GLANCE-V0", "G6-GLANCE-V1", "G6-GLANCE-V4", "G6-GLANCE-V2", "G6-GLANCE-V3"]),
              ("GLANCE image only", ["G6-GLANCE-IMG-V1"]), ("Annual grid", ["G12-RNN-V4", "G12-LSTM-V4"]),
              ("No grid", [f"NG-{NAME[m]}-V4" for m in ("rnn", "lstm", "no", "lnn", "tlstm")]),
              ("No grid −50% visits", [f"NG50-{NAME[m]}-V4" for m in ("rnn", "lstm", "no", "lnn", "tlstm")]),
              ("V-RS (random query schedules)", [f"G6-{CNAME['rs']}-V4"])]
    assert sorted(i for _, g in groups for i in g) == sorted(R), "summary must contain every (non-excluded) model of allresults' registry"
    return R, groups


def _nogrid_test(m, s, PAT, Pt, S, X):
    d = extra_rdir("nogrid", m, s)
    if not (d / "predictions_test.csv").exists():
        return None
    df = pd.read_csv(d / "predictions_test.csv")
    f5 = None
    if m in ("lnn", "no", "tlstm"):
        Ft = S[S.pid.isin(Pt) & S["first"]].reset_index(drop=True)
        ds = V2Dataset(irreg_thin(Ft, IRREG_SETTINGS["S0"], seed=s), X, scaler_of(d / "scaler.npz"), 1)
        mod = irreg_build(m, 1025); mod.load_state_dict(torch.load(d / "best.pt", map_location="cpu")["model_state_dict"]); mod.eval()
        f5 = five(mod, ds)
    return nogrid_recs(df, PAT, set(Pt), f5)


def records(r, setname, PAT, P, PK, S=None, X=None):
    mid, kind, Ps = r["id"], r["kind"], set(P)
    if kind == "fg":
        return {0: fg_recs(PAT, P)}
    if setname == "cv":
        if kind == "fixed":
            m, g, v = r["key"].split("_")
            return {s: {p: x for k in FOLDS for p, x in fixed_recs(_cv_run_dir(run_name(m, int(g == "grade"), v, k, s)) / "predictions_heldout.csv", PAT, Ps).items()}
                    for s in (SEEDS if r.get("force3") else cv_seeds(r["key"]))}
        if kind == "continuous":
            return {s: {p: x for k in FOLDS for p, x in fixed_recs(continuous_rdir(r["key"], k, s) / "predictions_heldout.csv", PAT, Ps).items()}
                    for s in SEEDS if all((continuous_rdir(r["key"], k, s) / "done.json").exists() for k in FOLDS)}
        if kind == "annual":
            return {s: {p: x for k in FOLDS for p, x in annual_recs(annual_rdir(r["key"], k, s) / "predictions_heldout.csv", PAT, Ps).items()}
                    for s in SEEDS if all((annual_rdir(r["key"], k, s) / "done.json").exists() for k in FOLDS)}
        if kind == "nogrid":
            out = {}
            for s in SEEDS:
                if not all((irreg_rdir(r["key"], r["setting"], k, s) / "done.json").exists() for k in FOLDS):
                    continue
                df = pd.concat([pd.read_csv(irreg_rdir(r["key"], r["setting"], k, s) / "predictions_heldout.csv") for k in FOLDS])
                f5 = dict(zip(PK[mid][s].pid, PK[mid][s].f5)) if r["key"] in ("lnn", "no", "tlstm") and mid in PK and s in PK[mid] else None
                out[s] = nogrid_recs(df, PAT, Ps, f5)
            return out
    else:
        if kind == "fixed":
            m, g, v = r["key"].split("_")
            return {s: fixed_recs(RESULTS_FINAL / "runs" / run_name(m, int(g == "grade"), v, 0, s) / "predictions_test.csv", PAT, Ps)
                    for s in SEEDS if (RESULTS_FINAL / "runs" / run_name(m, int(g == "grade"), v, 0, s) / "predictions_test.csv").exists()}
        if kind == "continuous":
            return {s: fixed_recs(continuous_rdir(r["key"], 0, s) / "predictions_test.csv", PAT, Ps)
                    for s in SEEDS if (continuous_rdir(r["key"], 0, s) / "predictions_test.csv").exists()}
        if kind == "annual":
            return {s: annual_recs(extra_rdir("annual", r["key"], s) / "predictions_test.csv", PAT, Ps)
                    for s in SEEDS if (extra_rdir("annual", r["key"], s) / "predictions_test.csv").exists()}
        if kind == "nogrid":
            out = {}
            for s in SEEDS:
                rec = _nogrid_test(r["key"], s, PAT, sorted(Ps), S, X)
                if rec is not None:
                    out[s] = rec
            return out
    raise ValueError(mid)


def ols(t, y):
    return float(np.polyfit(t, y, 1)[0]) if len(np.unique(t)) >= 2 else np.nan


def new_pp(R, D, PAT, P):
    det = dict(zip(D.pid, D.det.fillna(False).astype(bool))); rows = []
    for p in P:
        r, d = R[p], PAT[p]; t, y, yp = r["vt"], r["vy"], r["vp"]; fut = t > 0
        x = dict(pid=p, E=bool(d["E2"]))
        for hn, a, b in HB:
            k = fut & (t > a) & (t <= b); x[f"s_{hn}"] = float(np.abs(yp[k] - y[k]).sum()); x[f"n_{hn}"] = int(k.sum())
        x["ob"], x["pb"] = ols(t, y), ols(t, yp)
        i0 = int(np.argmin(t)); assert t[i0] == 0
        x["rise_last"] = float(yp[int(np.argmax(t))] - yp[i0])
        x["worse"] = bool(d["y"][int(np.argmax(d["t"]))] >= d["g0"] + 1)
        k10 = r["st"] <= 10 + 1e-9; mx = float(np.max(r["sp"][k10])); y0 = float(r["sp"][int(np.argmin(r["st"]))])
        x.update(maxf=mx, rise_max=mx - y0, missed=bool(d["E2"] and not det[p]))
        rows.append(x)
    return pd.DataFrame(rows)


def wspear(a, b, W):
    out = np.full(W.shape[0], np.nan)
    for i, w in enumerate(W):
        idx = np.repeat(np.arange(len(a)), w.astype(int))
        if len(idx) > 2 and np.ptp(a[idx]) > 0 and np.ptp(b[idx]) > 0:
            out[i] = spearmanr(a[idx], b[idx])[0]
    return out


def new_agg(N, W, which=NEW):
    o = {}; E = N.E.values.astype(bool)
    with np.errstate(all="ignore"):
        for hn, _, _ in HB:
            o[f"mae_{hn}"] = (W @ N[f"s_{hn}"].values) / (W @ N[f"n_{hn}"].values)
        o["bal_mae"] = np.mean([o[f"mae_{h[0]}"] for h in HB], 0)
        ob, pb = N.ob.values, N.pb.values; ok = np.isfinite(ob) & np.isfinite(pb); e, bias = np.abs(pb - ob), pb - ob
        for g, sel in (("all", ok), ("prog", ok & E), ("stable", ok & ~E)):
            o[f"f2_abs_{g}"] = (W[:, sel] @ e[sel]) / W[:, sel].sum(1); o[f"f2_bias_{g}"] = (W[:, sel] @ bias[sel]) / W[:, sel].sum(1)
            if g != "stable" and f"f3_sp_{g}" in which:
                o[f"f3_sp_{g}"] = wspear(pb[sel], ob[sel], W[:, sel])
        wo = N.worse.values.astype(bool); o["f4_auc"] = auc_w(N.rise_last.values, wo, ~wo, W)
    return {k: v for k, v in o.items() if k in which}


def f5_seed(N):
    M = N[N.missed]; n = len(M)
    if n == 0:
        return dict(n_missed=0, n_conv=int(N.E.sum()), pct_ge30=np.nan, pct_ge325=np.nan, pct_rise05=np.nan, med_dist=np.nan, med_max=np.nan)
    return dict(n_missed=n, n_conv=int(N.E.sum()), pct_ge30=100 * float((M.maxf >= 3.0).mean()), pct_ge325=100 * float((M.maxf >= 3.25).mean()),
                pct_rise05=100 * float((M.rise_max >= 0.5).mean()), med_dist=float(np.median(3.5 - M.maxf)), med_max=float(np.median(M.maxf)))


def compute(setname):
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    PAT = patients()
    P = sorted(p for p in rounds()[0][0] if not PAT[p]["late0"]) if setname == "cv" else sorted(p for p in test_pids() if not PAT[p]["late0"])
    ONE = np.ones((1, len(P))); R_, groups = reg()
    PK = pickle.load(open(ALLRESULTS_OUT / "per_patient.pkl", "rb"))["PPc"]
    S, X = (load_samples(), load_X()) if setname == "test" else (None, None)
    if setname == "test":
        bt_path = RESULTS_EVAL / "test" / "baseline_predictions_test.pkl"
        Bt = pickle.load(open(bt_path, "rb")) if bt_path.exists() else {}
    else:
        base = [pickle.load(open(RESULTS_FINAL / "baselines" / "cv" / f"fold{k}.pkl", "rb"))["pred"] for k in FOLDS]
        Bt = {"cox": {}, "logreg": {("E2", 5.0): {}}, "svm": {("E2", 5.0): {}}}
        for b in base:
            Bt["cox"].update(b["cox"]); Bt["logreg"][("E2", 5.0)].update(b["logreg"][("E2", 5.0)]); Bt["svm"][("E2", 5.0)].update(b["svm"][("E2", 5.0)])
    long, f5rows, NN = [], [], {}
    for mid, r in R_.items():
        if setname == "test" and not r.get("test"):
            continue
        per = []
        if r["kind"] in ("lr", "svm"):
            src = Bt.get("logreg" if r["kind"] == "lr" else "svm", {}).get(("E2", 5.0), {})
            if src:
                case, ctrl = case_ctrl(PAT, P, "E2", 5.0); per = [dict(auc5=auc_w(np.array([src[p] for p in P]), case, ctrl, ONE)[0])]
        elif r["kind"] == "cox":
            if Bt.get("cox"):
                R = cox_recs(Bt["cox"], PAT, P); D = per_patient(R, PAT, P); per = [{k: v[0] for k, v in agg(D, ONE, PAT, P).items()}]
        else:
            RR = records(r, setname, PAT, P, PK, S, X); NN[mid] = {}
            for s, R in RR.items():
                D = per_patient(R, PAT, P); m = {k: v[0] for k, v in agg(D, ONE, PAT, P).items()}
                if r.get("kind") == "nogrid" and r["key"] in ("rnn", "lstm"):
                    m["auc5"] = np.nan
                N = new_pp(R, D, PAT, P); NN[mid][s] = N
                m.update({k: v[0] for k, v in new_agg(N, ONE).items()})
                if mid == "FG":
                    m["f3_sp_all"] = m["f3_sp_prog"] = np.nan
                per.append(m); f5rows.append(dict(id=mid, seed=s, **f5_seed(N)))
        for k in EXIST + NEW:
            mu, sd, n = mean_sd([x.get(k, np.nan) for x in per])
            long.append(dict(id=mid, metric=k, mean=mu, sd=sd, n_seeds=n, per_seed=";".join(f"{x.get(k, np.nan):.4f}" for x in per)))
        print(f"{setname} {mid} ({time.time() - t0:.0f}s)", flush=True)
    L = pd.DataFrame(long); L.to_csv(OUT / f"{setname}_long.csv", index=False)
    pd.DataFrame(f5rows).to_csv(OUT / f"f5_{setname}_per_seed.csv", index=False)
    W = boot_weights(len(P)); rows = []
    have = lambda mid: mid in NN and NN[mid]
    val = {mid: {s: new_agg(NN[mid][s], W, TESTED) for s in NN[mid]} for mid in (GL,) + OTHERS if have(mid)}
    pt = {mid: {s: new_agg(NN[mid][s], ONE, TESTED) for s in NN[mid]} for mid in (GL,) + OTHERS if have(mid)}
    for b in OTHERS:
        if GL not in val or b not in val:
            continue
        for k in TESTED:
            ba, bb = np.mean([val[GL][s][k] for s in val[GL]], 0), np.mean([val[b][s][k] for s in val[b]], 0)
            va, vb = np.mean([pt[GL][s][k][0] for s in pt[GL]]), np.mean([pt[b][s][k][0] for s in pt[b]])
            d = ba - bb; lo, hi = np.nanpercentile(d, [2.5, 97.5])
            rows.append(dict(set=setname.upper(), a=GL, b=b, metric=k, value_a=va, value_b=vb, diff=va - vb, lo=lo, hi=hi, p=paired_p(d)))
    T = pd.DataFrame(rows)
    if len(T):
        T["p_holm"] = holm(T.p.values)
    T.to_csv(OUT / f"tests_{setname}.csv", index=False)
    json.dump(dict(seconds=time.time() - t0, n_patients=len(P)), open(OUT / f"run_info_{setname}.json", "w"), indent=1)
    print(T.round(4).to_string() if len(T) else "no tests (one of GLANCE V4 / RNN / LSTM / NO V4 not yet scored)")


A = "NG-GLANCE-V4"; B = ("G6-RNN-V4", "G6-LSTM-V4", "NG-NO-V4")
EXISTM = ("auc5", "c", "down"); NEWM = ("f4_auc", "bal_mae", "f2_abs_prog", "f2_abs_stable", "f2_bias_all", "f3_sp_all")
ORDER = ["auc5", "c", "f4_auc", "bal_mae", "down", "f2_abs_prog", "f2_abs_stable", "f2_bias_all", "f3_sp_all"]


def metric_sets(mid, PAT, P, PK, W, R_):
    """seed -> {metric: (B,) array} under weights W."""
    out = {}
    for s, R in records(R_[mid], "cv", PAT, P, PK).items():
        D = per_patient(R, PAT, P); m = {k: v for k, v in agg(D, W, PAT, P).items() if k in EXISTM}
        m.update(new_agg(new_pp(R, D, PAT, P), W, NEWM)); out[s] = m
    return out


def table1(nboot=1000):
    OUT.mkdir(parents=True, exist_ok=True)
    PAT = patients(); P = sorted(p for p in rounds()[0][0] if not PAT[p]["late0"]); R_, _ = reg()
    PK = pickle.load(open(ALLRESULTS_OUT / "per_patient.pkl", "rb"))["PPc"]
    W = boot_weights(len(P), nboot); ONE = np.ones((1, len(P)))
    bt = {m: metric_sets(m, PAT, P, PK, W, R_) for m in (A,) + B}
    pt = {m: metric_sets(m, PAT, P, PK, ONE, R_) for m in (A,) + B}
    sm = lambda d, k: np.mean([d[s][k] for s in d], 0)
    rows = []
    for k in ORDER:
        for b in (("NG-NO-V4",) if k == "auc5" else B):
            d = sm(bt[A], k) - sm(bt[b], k); va, vb = float(sm(pt[A], k)[0]), float(sm(pt[b], k)[0])
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            rows.append(dict(metric=k, a=A, b=b, value_a=va, value_b=vb, diff=va - vb, lo=lo, hi=hi, p=paired_p(d)))
    T = pd.DataFrame(rows); T["p_holm"] = holm(T.p.values)
    suf = "" if nboot == 1000 else f"_{nboot // 1000}k"
    T.to_csv(OUT / f"tests_table1{suf}.csv", index=False)
    cvp = OUT / "cv_long.csv"
    if cvp.exists():
        L = pd.read_csv(cvp)
        for x in T.itertuples():
            for mid, v in ((x.a, x.value_a), (x.b, x.value_b)):
                q = L[(L.id == mid) & (L.metric == x.metric)]
                if len(q):
                    assert abs(q["mean"].iloc[0] - v) < 1e-9, (mid, x.metric, v, q["mean"].iloc[0])
    print(T.round(4).to_string())
    return T


def main():
    cmd = sys.argv[1]
    if cmd == "compute":
        compute(sys.argv[sys.argv.index("--set") + 1])
    elif cmd == "table1":
        table1(int(sys.argv[sys.argv.index("--nboot") + 1]) if "--nboot" in sys.argv else 1000)


if __name__ == "__main__":
    main()
