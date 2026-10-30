"""Consolidated per-patient scoring across the whole model registry.
   allresults: saving all results -> results/posthoc/allresults/{cv_matrix_long.csv, test_matrix_long.csv, monthly_per_seed.csv,
   per_patient.pkl, run_info.json}
Run: python -m scripts.allresults
"""
import json
import pickle
import time
import numpy as np, pandas as pd, torch
from glance.config import DROP_TOL, RESULTS_CV, RESULTS_FINAL, RESULTS_EVAL, RESULTS_POSTHOC
from glance.data import MinMaxNorm, SEEDS, FOLDS, rounds, round_data, load_samples, load_X, test_pids, V2Dataset
from glance.evaluate import patients, case_ctrl, auc_w, cindex_w, wmedian, cv_seeds, at, G, FIX_C, FIX_P, _cv_run_dir
from glance.models import build, run_name
from glance.irregular import SETTINGS as IRREG_SETTINGS, build as irreg_build, thin as irreg_thin, rdir as irreg_rdir
from glance.annual import rdir as annual_rdir
from glance.continuous import rdir as continuous_rdir

OUT = RESULTS_POSTHOC / "allresults"; OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GM = np.round(np.arange(0, 120 + 1) / 12, 6)
HB = (("0–2 y", 0, 2), ("2–5 y", 2, 5), ("5–10 y", 5, 10))
NAME = {"lnn": "GLANCE", "rnn": "RNN", "lstm": "LSTM", "no": "NO", "tlstm": "T-LSTM"}
CNAME = {"rs": "GLANCE-RS", "con": "GLANCE-CON", "ltc": "GLANCE-LTC"}


def rec_points(vt, vy, vp, st, sp, f5, d, thr=FIX_C, grade=True):
    last = d["last"]; allt = np.r_[vt, st]; allp = np.r_[vp, sp] if grade else sp
    k = allt <= last + 1e-9
    return dict(vt=vt, vy=vy, vp=vp, st=st, sp=sp, f5=f5, thr=thr, grade=grade,
                cmax=float(np.max(allp[k[:len(allp)]])) if grade else float(np.interp(last, st, sp)),
                clast=float(vp[np.argmax(vt)]) if grade else float(np.interp(last, st, sp)))


def fixed_recs(path, PAT, pids):
    P = pd.read_csv(path); R = {}
    for p, q in P.groupby("pid"):
        if p not in pids:
            continue
        q = q.sort_values("t"); v = q[q.is_visit]; g = q[np.isin(np.round(q.t, 4), np.round(G, 4))].drop_duplicates("t")
        assert len(g) == len(G)
        allq = q.drop_duplicates("t"); d = PAT[p]
        r = rec_points(v.t.values, v.y_true.values, v.y_pred.values, g.t.values, g.y_pred.values, float(g.y_pred.values[10]), d)
        k = allq.t.values <= d["last"] + 1e-9; r["cmax"] = float(allq.y_pred.values[k].max())
        R[p] = r
    return R


def annual_recs(path, PAT, pids):
    P = pd.read_csv(path); R = {}
    for p, q in P.groupby("pid"):
        if p not in pids:
            continue
        q = q.sort_values("t"); d = PAT[p]; ta, ya = q.t.values, q.y_pred.values
        vp = np.interp(d["t"], ta, ya); k10 = ta <= 10 + 1e-9
        r = rec_points(d["t"], d["y"], vp, ta[k10], ya[k10], float(ya[np.isclose(ta, 5)][0]), d)
        kk = ta <= d["last"] + 1e-9; r["cmax"] = float(max(ya[kk].max(), vp[d["t"] <= d["last"] + 1e-9].max()))
        R[p] = r
    return R


def nogrid_recs(df, PAT, pids, f5=None):
    R = {}
    for p, q in df.groupby("pid"):
        if p not in pids:
            continue
        q = q.sort_values("t"); t, y, yp = q.t.values, q.y_true.values, q.y_pred.values
        r = rec_points(t, y, yp, t, yp, (f5 or {}).get(p, np.nan), PAT[p]); r["cmax"] = float(yp.max()); R[p] = r
    return R


def fg_recs(PAT, P):
    return {p: rec_points(PAT[p]["t"], PAT[p]["y"], np.full(len(PAT[p]["t"]), PAT[p]["g0"]), G, np.full(len(G), PAT[p]["g0"]), PAT[p]["g0"], PAT[p]) for p in P}


def cox_recs(cox, PAT, P):
    R = {}
    for p in P:
        f = cox[p]; sp = at(f, G)
        R[p] = rec_points(np.array([0.0]), np.array([PAT[p]["g0"]]), np.array([np.nan]), G, sp, float(at(f, 5.0)[0]), PAT[p], thr=FIX_P, grade=False)
        R[p]["cmax"] = R[p]["clast"] = float(at(f, PAT[p]["last"])[0])
    return R


def slope_all(t, y):
    """Theil-Sen slope over the given visits, t = 0 included (the M1 'pace' definition)."""
    if len(t) < 2:
        return np.nan
    i, j = np.triu_indices(len(t), 1)
    return float(np.median((y[j] - y[i]) / (t[j] - t[i])))


def per_patient(R, PAT, P):
    rows = []
    for p in P:
        r = R[p]; d = PAT[p]; t, y, yp = r["vt"], r["vy"], r["vp"]; fut = t > 0; y0 = d["g0"]
        on = r["st"][np.argmax(r["sp"] >= r["thr"])] if (r["sp"] >= r["thr"]).any() else np.inf
        x = dict(pid=p, g0=y0, E=d["E2"], T=d["E2_T"], lo=d["E2_lo"], last=d["last"], up=d["traj"] == "Only up", onset=on,
                 f5=r["f5"], cmax=r["cmax"], clast=r["clast"], grade=r["grade"])
        if r["grade"]:
            e = np.abs(yp[fut] - y[fut]); x.update(sabs=e.sum(), nv=fut.sum(), down=float(np.any(np.diff(r["sp"]) < -DROP_TOL)))
            ch = fut & (y != y0); x.update(m2a_s=np.abs(yp[ch] - y[ch]).sum(), m2a_n=ch.sum())
            ob, pr = y >= y0 + 1, yp >= y0 + 0.5
            for hn, a, b in (("all", 0, np.inf),) + HB:
                k = fut & (t > a) & (t <= b)
                x.update({f"tp_{hn}": (k & ob & pr).sum(), f"fn_{hn}": (k & ob & ~pr).sum(), f"tn_{hn}": (k & ~ob & ~pr).sum(), f"fp_{hn}": (k & ~ob & pr).sum()})
            x["b_obs"] = slope_all(t, y); x["b_pred"] = slope_all(t, yp)
        if d["E2"]:
            det = np.isfinite(on); x.update(det=det, ontime=det and d["E2_lo"] - 1 <= on <= d["E2_T"] + 1, early=det and on < d["E2_lo"] - 1,
                                         late=det and on > d["E2_T"] + 1, dist=max(d["E2_lo"] - on, 0, on - d["E2_T"]) if det else np.nan)
        else:
            x.update(fa=bool(np.isfinite(on) and on <= d["last"] + 1e-9))
        rows.append(x)
    return pd.DataFrame(rows)


def agg(D, W, PAT, P, which="all"):
    """metric -> (B,) array under patient weights W."""
    o = {}; E = D.E.values.astype(bool); ws = lambda m: W[:, m].sum(1)
    case, ctrl = case_ctrl(PAT, P, "E2", 5.0)
    f5 = D.f5.values.astype(float)
    o["auc5"] = auc_w(f5, case, ctrl, W) if np.isfinite(f5).all() else np.full(W.shape[0], np.nan)
    if D.cmax.notna().all():
        o["c"] = cindex_w(D["T"].values, E, D.cmax.values, W); o["c_last"] = cindex_w(D["T"].values, E, D.clast.values, W)
    if which == "auc":
        return o
    det = D.det.fillna(False).values.astype(bool); ont = D.ontime.fillna(False).values.astype(bool); late = D.late.fillna(False).values.astype(bool)
    early = D.early.fillna(False).values.astype(bool); fa = D.fa.fillna(False).values.astype(bool); dist = D.dist.values.astype(float)
    with np.errstate(all="ignore"):
        o["det"] = 100 * (W @ det) / (W @ E); o["ontime"] = 100 * (W @ ont) / (W @ E); o["late"] = 100 * (W @ late) / (W @ E)
        o["early"] = 100 * (W @ early) / (W @ E); o["missed"] = 100 * (W @ (E & ~det)) / (W @ E); o["fa"] = 100 * (W @ fa) / (W @ ~E)
        o["d"] = np.array([wmedian(dist[det], w[det]) for w in W])
        Th = np.where(np.isfinite(D.onset.values), D.onset.values, 10.5); o["m3"] = cindex_w(D["T"].values, E, -Th, W)
        for g, sel in (("g3", D.g0.values == 3), ("g12", np.isin(D.g0.values, [1, 2]))):
            pe = E & sel
            o[f"{g}_ontime"] = 100 * (W @ (ont & sel)) / (W @ pe); o[f"{g}_missed"] = 100 * (W @ (pe & ~det)) / (W @ pe)
            o[f"{g}_late"] = 100 * (W @ (late & sel)) / (W @ pe); o[f"{g}_early"] = 100 * (W @ (early & sel)) / (W @ pe)
            o[f"{g}_fa"] = 100 * (W @ (fa & sel)) / (W @ (~E & sel))
        if D.grade.all():
            o["mae"] = (W @ D.sabs.values) / (W @ D.nv.values); up = D.up.values.astype(bool)
            o["mae_up"] = (W[:, up] @ D.sabs.values[up]) / (W[:, up] @ D.nv.values[up]); o["down"] = 100 * (W @ D.down.values) / W.sum(1)
            o["m2a"] = (W @ D.m2a_s.values) / (W @ D.m2a_n.values)
            for hn in ("all",) + tuple(h[0] for h in HB):
                tp, fn, tn, fp = (W @ D[f"{k}_{hn}"].values.astype(float) for k in ("tp", "fn", "tn", "fp"))
                o[f"m2b_{hn}"] = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
            e = np.abs(D.b_pred.values - D.b_obs.values); bias = D.b_pred.values - D.b_obs.values
            for g, sel in (("all", np.ones(len(D), bool)), ("prog", E), ("stable", ~E)):
                o[f"m1_abs_{g}"] = (W[:, sel] @ e[sel]) / ws(sel); o[f"m1_bias_{g}"] = (W[:, sel] @ bias[sel]) / ws(sel)
    return o


def scaler_of(path):
    z = np.load(path); sc = MinMaxNorm(np.zeros(1), np.ones(1)); sc.min, sc.scale = z["data_min"], z["scale"]; return sc


@torch.no_grad()
def monthly(mod, x):
    q = torch.tensor(GM, dtype=torch.float32, device=dev)[None].expand(x.size(0), -1)
    out = mod(x, q)
    return (out[0] if isinstance(out, tuple) else out).cpu().numpy()


@torch.no_grad()
def five(mod, ds):
    o = {}
    for i in range(len(ds)):
        x, t, _, _ = ds[i]; q = torch.unique(torch.cat([t, torch.tensor([5.0])]), sorted=True)
        p = mod(x[None].to(dev), q[None].to(dev)); p = p[0] if isinstance(p, tuple) else p
        o[ds.s.pid.iloc[i]] = float(p[0, int(torch.nonzero(torch.isclose(q, torch.tensor(5.0)))[0])])
    return o


def monthly_metrics(M, own, PAT, P):
    """M: pid -> monthly forecast (121); own: pid -> (times, values) of the model's own forecast; returns per-seed metrics."""
    diffs = []
    for p in P:
        t, v = own[p]; k = t <= 10 + 1e-9; idx = np.round(t[k] * 12).astype(int); diffs.append(np.abs(M[p][idx] - v[k]))
    dd = np.concatenate(diffs)
    R = {p: dict(st=GM, sp=M[p], thr=FIX_C) for p in P}
    D = per_patient_onset(R, PAT, P); ONE = np.ones((1, len(P)))
    o = {k: v[0] for k, v in agg_onset(D, ONE).items()}
    o.update(mean_absdiff=float(dd.mean()), max_absdiff=float(dd.max()))
    return o, D


def per_patient_onset(R, PAT, P):
    rows = []
    for p in P:
        r = R[p]; d = PAT[p]; on = r["st"][np.argmax(r["sp"] >= r["thr"])] if (r["sp"] >= r["thr"]).any() else np.inf
        x = dict(pid=p, E=d["E2"], onset=on)
        if d["E2"]:
            det = np.isfinite(on); x.update(det=det, ontime=det and d["E2_lo"] - 1 <= on <= d["E2_T"] + 1, late=det and on > d["E2_T"] + 1,
                                         dist=max(d["E2_lo"] - on, 0, on - d["E2_T"]) if det else np.nan)
        else:
            x.update(fa=bool(np.isfinite(on) and on <= d["last"] + 1e-9))
        rows.append(x)
    return pd.DataFrame(rows)


def agg_onset(D, W):
    E = D.E.values.astype(bool); det = D.det.fillna(False).values.astype(bool); dist = D.dist.values.astype(float)
    with np.errstate(all="ignore"):
        return dict(det=100 * (W @ det) / (W @ E), ontime=100 * (W @ D.ontime.fillna(False).values.astype(bool)) / (W @ E),
                    late=100 * (W @ D.late.fillna(False).values.astype(bool)) / (W @ E), fa=100 * (W @ D.fa.fillna(False).values.astype(bool)) / (W @ ~E),
                    d=np.array([wmedian(dist[det], w[det]) for w in W]))


def registry():
    """Every scored model: id, name, kind (how to read its forecasts), key / seeds / folder metadata."""
    R = []
    add = lambda **k: R.append(k)
    add(id="FG", name="First grade alone (grade stays the same)", backbone="–", input="first-visit grade", train_sched="–", eval_sched="any time (flat)",
        loss="–", seeds="deterministic", folder="(computed)", params=0, test=True, kind="fg")
    add(id="COX", name="Cox PH", backbone="lifelines CoxPHFitter (penalizer 0.1)", input="first grade + k PCA of LE‖RE (k by inner CV)", train_sched="E2 time",
        loss="partial likelihood", eval_sched="0.5-y grid (1 − S(t))", seeds="deterministic", folder="finaltraining/baselines", params=np.nan, test=True, kind="cox")
    add(id="LR", name="Logistic regression, late by 2 / 5 y", backbone="sklearn LogisticRegression", input="first grade + 10 PCA", train_sched="–",
        loss="log loss", eval_sched="2 y, 5 y", seeds="deterministic", folder="finaltraining/baselines", params=11, test=True, kind="lr")
    add(id="SVM", name="Linear SVM, late by 2 / 5 y (calibrated)", backbone="sklearn SVC linear + sigmoid calibration", input="LE‖RE (1024)",
        train_sched="all starting visits", loss="hinge", eval_sched="2 y, 5 y", seeds="deterministic", folder="finaltraining/baselines", params=1025, test=True, kind="svm")
    for m in ("rnn", "lstm", "no", "lnn"):
        for v in ("V0", "V1", "V4"):
            add(id=f"G6-{NAME[m]}-{v}", name=f"{NAME[m]} + grade, fixed 6-month grid, {v}", backbone=m.upper() if m != "lnn" else "LNN (CfC)",
                input="LE‖RE + grade/5", train_sched="0.5-y grid 0–10 y ∪ visits", eval_sched="0.5-y grid ∪ visits", loss=v,
                seeds=f"CV {len(cv_seeds(f'{m}_grade_{v}'))}, TEST 3", folder=f"crossvalidation|lossselection|finaltraining/runs/{m}_grade_{v}_f*_s*",
                key=f"{m}_grade_{v}", kind="fixed", test=True)
    for v in ("V2", "V3"):
        add(id=f"G6-GLANCE-{v}", name=f"GLANCE (LNN + grade), fixed grid, {v} (ablation)", backbone="LNN (CfC)", input="LE‖RE + grade/5",
            train_sched="0.5-y grid ∪ visits", eval_sched="0.5-y grid ∪ visits", loss=v, seeds="CV 3 (no final model)",
            folder=f"crossvalidation/runs/lnn_grade_{v}_f*_s*", key=f"lnn_grade_{v}", kind="fixed", test=False)
    add(id="G6-GLANCE-IMG-V1", name="LNN image only, fixed grid, V1", backbone="LNN (CfC)", input="LE‖RE only", train_sched="0.5-y grid ∪ visits",
        eval_sched="0.5-y grid ∪ visits", loss="V1", seeds="CV 3, TEST 3", folder="crossvalidation|finaltraining/runs/lnn_img_V1_f*_s*", key="lnn_img_V1", kind="fixed", test=True)
    for m in ("rnn", "lstm"):
        add(id=f"G12-{NAME[m]}-V4", name=f"{NAME[m]} + grade, annual grid, V4 (posthoc)", backbone=m.upper(), input="LE‖RE + grade/5",
            train_sched="integer-year targets, 1-y step", eval_sched="annual points (half-years interpolated)", loss="V4", seeds="CV 3",
            folder=f"posthoc/annual/runs/{m}_f*_s*", key=m, kind="annual", test=False)
    for st, lab in (("S0", "NG"), ("S50", "NG50")):
        for m in ("rnn", "lstm", "no", "lnn", "tlstm"):
            add(id=f"{lab}-{NAME[m]}-V4", name=f"{NAME[m]} + grade, no grid{' −50% visits' if st == 'S50' else ''}, V4", backbone={"lnn": "LNN (CfC)", "tlstm": "time-aware LSTM"}.get(m, m.upper()),
                input="LE‖RE + grade/5", train_sched=f"own visit times{' (50% future visits dropped)' if st == 'S50' else ''}", eval_sched="own (kept) visit times",
                loss="V4", seeds="CV 3", folder=f"posthoc/irregular/runs/{m}_{st}_f*_s*", key=m, setting=st, kind="nogrid", test=False)
    for v in ("rs", "con", "ltc"):
        add(id=f"G6-{CNAME[v]}-V4", name=f"GLANCE continuous-time variant '{v}', fixed-schedule evaluation, V4 (posthoc)",
            backbone="LNN (LTC)" if v == "ltc" else "LNN (CfC)", input="LE‖RE + grade/5", train_sched="per variant (see glance.continuous)",
            eval_sched="0.5-y grid ∪ visits", loss="V4", seeds="CV 3, TEST 3 (rs only)", folder=f"posthoc/continuous/runs/{v}_f*_s*",
            key=v, kind="continuous", test=v == "rs")
    return R


def mean_sd(vals):
    v = np.array([x for x in vals if x is not None], float)
    if not np.isfinite(v).any():
        return np.nan, np.nan, 0
    v = v[np.isfinite(v)]
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else np.nan), len(v)


def main():
    t0 = time.time(); PAT = patients(); S = load_samples(); X = load_X()
    Pc = sorted(p for p in rounds()[0][0] if not PAT[p]["late0"]); Pt = sorted(p for p in test_pids() if not PAT[p]["late0"])
    ONEc, ONEt = np.ones((1, len(Pc))), np.ones((1, len(Pt))); REG = registry()
    cvrows, terows, PPc, PPt, monthly_rows, monthD = [], [], {}, {}, [], {}
    base_cv = [pickle.load(open(RESULTS_FINAL / "baselines" / "cv" / f"fold{k}.pkl", "rb"))["pred"] for k in FOLDS]
    cox_cv, lr_cv, svm_cv = {}, {}, {}
    for b in base_cv:
        cox_cv.update(b["cox"]); lr_cv.update(b["logreg"][("E2", 5.0)]); svm_cv.update(b["svm"][("E2", 5.0)])
    bt_path = RESULTS_EVAL / "test" / "baseline_predictions_test.pkl"
    Bt = pickle.load(open(bt_path, "rb")) if bt_path.exists() else None

    def score(R, P, ONE):
        D = per_patient(R, PAT, P); return D, {k: v[0] for k, v in agg(D, ONE, PAT, P).items()}

    for r in REG:
        mid = r["id"]; per_cv, per_te, Dcv, Dte = [], [], {}, {}
        if r["kind"] == "fg":
            D, m = score(fg_recs(PAT, Pc), Pc, ONEc); per_cv = [m]; Dcv[0] = D
            D, m = score(fg_recs(PAT, Pt), Pt, ONEt); per_te = [m]; Dte[0] = D
        elif r["kind"] == "cox":
            D, m = score(cox_recs(cox_cv, PAT, Pc), Pc, ONEc); per_cv = [m]; Dcv[0] = D
            if Bt:
                D, m = score(cox_recs(Bt["cox"], PAT, Pt), Pt, ONEt); per_te = [m]; Dte[0] = D
        elif r["kind"] in ("lr", "svm"):
            src_cv = lr_cv if r["kind"] == "lr" else svm_cv
            case, ctrl = case_ctrl(PAT, Pc, "E2", 5.0); per_cv.append(dict(auc5=auc_w(np.array([src_cv[p] for p in Pc]), case, ctrl, ONEc)[0]))
            if Bt:
                src_te = Bt["logreg"][("E2", 5.0)] if r["kind"] == "lr" else Bt["svm"][("E2", 5.0)]
                case, ctrl = case_ctrl(PAT, Pt, "E2", 5.0); per_te.append(dict(auc5=auc_w(np.array([src_te[p] for p in Pt]), case, ctrl, ONEt)[0]))
        elif r["kind"] == "fixed":
            m, g, v = r["key"].split("_")
            for s in cv_seeds(r["key"]):
                R = {}
                for k in FOLDS:
                    R.update(fixed_recs(_cv_run_dir(run_name(m, int(g == "grade"), v, k, s)) / "predictions_heldout.csv", PAT, set(Pc)))
                D, mm = score(R, Pc, ONEc); per_cv.append(mm); Dcv[s] = D
            if r["test"]:
                for s in SEEDS:
                    d = RESULTS_FINAL / "runs" / run_name(m, int(g == "grade"), v, 0, s) / "predictions_test.csv"
                    if not d.exists():
                        continue
                    R = fixed_recs(d, PAT, set(Pt)); D, mm = score(R, Pt, ONEt); per_te.append(mm); Dte[s] = D
        elif r["kind"] == "continuous":
            v = r["key"]
            for s in SEEDS:
                R = {}
                for k in FOLDS:
                    d = continuous_rdir(v, k, s) / "predictions_heldout.csv"
                    if d.exists():
                        R.update(fixed_recs(d, PAT, set(Pc)))
                if not R:
                    continue
                D, mm = score(R, Pc, ONEc); per_cv.append(mm); Dcv[s] = D
            if r["test"]:
                for s in SEEDS:
                    d = continuous_rdir(v, 0, s) / "predictions_test.csv"
                    if not d.exists():
                        continue
                    R = fixed_recs(d, PAT, set(Pt)); D, mm = score(R, Pt, ONEt); per_te.append(mm); Dte[s] = D
        elif r["kind"] == "annual":
            for s in SEEDS:
                R = {}
                for k in FOLDS:
                    d = annual_rdir(r["key"], k, s) / "predictions_heldout.csv"
                    if d.exists():
                        R.update(annual_recs(d, PAT, set(Pc)))
                if not R:
                    continue
                D, mm = score(R, Pc, ONEc); per_cv.append(mm); Dcv[s] = D
        else:
            m, st = r["key"], r["setting"]
            for s in SEEDS:
                if not all((irreg_rdir(m, st, k, s) / "done.json").exists() for k in FOLDS):
                    continue
                df = pd.concat([pd.read_csv(irreg_rdir(m, st, k, s) / "predictions_heldout.csv") for k in FOLDS])
                f5, Mm = {}, {}
                if m in ("lnn", "no", "tlstm"):
                    for k in FOLDS:
                        Dk = round_data(k); ho = irreg_thin(Dk["heldout_first"], IRREG_SETTINGS[st], seed=s); ds = V2Dataset(ho, Dk["X"], Dk["scaler"], 1)
                        mod = irreg_build(m, 1025).to(dev); mod.load_state_dict(torch.load(irreg_rdir(m, st, k, s) / "best.pt", map_location=dev)["model_state_dict"]); mod.eval()
                        f5.update(five(mod, ds)); Y = monthly(mod, ds.x.to(dev)); Mm.update(dict(zip(ds.s.pid, Y)))
                R = nogrid_recs(df, PAT, set(Pc), f5 if f5 else None); D, mm = score(R, Pc, ONEc)
                if m in ("rnn", "lstm"):
                    mm["auc5"] = np.nan
                if Mm:
                    own = {p: (R[p]["vt"], R[p]["vp"]) for p in Pc}
                    mo, MD = monthly_metrics(Mm, own, PAT, Pc); monthly_rows.append(dict(id=mid, seed=s, set="CV", **mo)); monthD[(mid, s)] = MD
                per_cv.append(mm); Dcv[s] = D
        PPc[mid], PPt[mid] = Dcv, Dte
        for lst, rows in ((per_cv, cvrows), (per_te, terows)):
            if not lst:
                continue
            for k in sorted({k for x in lst for k in x}):
                mu, sd, n = mean_sd([x.get(k, np.nan) for x in lst]); rows.append(dict(id=mid, metric=k, mean=mu, sd=sd, n_seeds=n, per_seed=";".join(f"{x.get(k, np.nan):.4f}" for x in lst)))
        print(f"{mid} done ({time.time() - t0:.0f}s)", flush=True)

    for m in ("lnn", "no"):
        mid = f"G6-{NAME[m]}-V4"
        for setname, P, seeds_folds in (("CV", Pc, [(s, k) for s in SEEDS for k in FOLDS]), ("TEST", Pt, [(s, 0) for s in SEEDS])):
            for s in SEEDS:
                Mm, own = {}, {}
                for (ss, k) in [x for x in seeds_folds if x[0] == s]:
                    d = _cv_run_dir(run_name(m, 1, "V4", k, s)) if k else RESULTS_FINAL / "runs" / run_name(m, 1, "V4", 0, s)
                    pred = d / ("predictions_heldout.csv" if k else "predictions_test.csv")
                    if not pred.exists():
                        continue
                    pids = rounds()[k][1] if k else test_pids()
                    F = S[S.pid.isin(pids) & S["first"]].reset_index(drop=True); ds = V2Dataset(F, X, scaler_of(d / "scaler.npz"), 1)
                    mod = build(m, 1025).to(dev); mod.load_state_dict(torch.load(d / "best.pt", map_location=dev)["model_state_dict"]); mod.eval()
                    Mm.update(dict(zip(ds.s.pid, monthly(mod, ds.x.to(dev)))))
                    sv = pd.read_csv(pred)
                    for p, q in sv.groupby("pid"):
                        g = q[np.isin(np.round(q.t, 4), np.round(G, 4))].drop_duplicates("t").sort_values("t"); own[p] = (g.t.values, g.y_pred.values)
                if Mm:
                    mo, MD = monthly_metrics(Mm, own, PAT, P); monthly_rows.append(dict(id=mid, seed=s, set=setname, **mo)); monthD[(mid, s, setname)] = MD
        print(f"monthly {mid} done ({time.time() - t0:.0f}s)", flush=True)
    MO = pd.DataFrame(monthly_rows); MO.to_csv(OUT / "monthly_per_seed.csv", index=False)
    CV = pd.DataFrame(cvrows); TE = pd.DataFrame(terows); CV.to_csv(OUT / "cv_matrix_long.csv", index=False); TE.to_csv(OUT / "test_matrix_long.csv", index=False)
    pickle.dump(dict(PPc=PPc, PPt=PPt, monthD=monthD, Pc=Pc, Pt=Pt), open(OUT / "per_patient.pkl", "wb"))
    json.dump(dict(seconds=time.time() - t0), open(OUT / "run_info.json", "w"))
    print("saved", time.time() - t0)


if __name__ == "__main__":
    main()
