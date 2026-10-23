"""Evaluation functions for the forecasting pipeline."""
import pickle
import numpy as np, pandas as pd
from .config import LATE_THRESHOLD, N_BOOT, BOOT_SEED, DROP_TOL, RESULTS_CV, RESULTS_LOSSSEL, RESULTS_FINAL
from .data import load_cohort, load_samples, FOLDS, SEEDS
from .models import run_name, ONE_SEED_FLAG

G = np.round(np.arange(0, 10.0001, 0.5), 1)
TB = np.round(np.arange(1.0, 8.0001, 0.5), 1)
HS = (2.0, 5.0)
EVS = ("E1", "E2")
FIX_C, FIX_P = 3.5, 0.5
MIN_CASES_T = 5
MIN_EVENTS_FLAG = 10


def traj_group(y):
    d = np.diff(y); up, dn = (d > 0).any(), (d < 0).any()
    return "Up and down" if up and dn else "Only up" if up else "Only down" if dn else "Stable"


def event(t, y, kind):
    late = y >= LATE_THRESHOLD
    for k in range(1, len(t)):
        if late[k] and (kind == "E1" or k == len(t) - 1 or late[k + 1]):
            return k
    return None


def patients():
    S = load_samples(); F = S[S["first"]]
    PAT = {}
    for r in F.itertuples():
        t, y = r.t.astype(float), r.y.astype(float)
        d = dict(pid=r.pid, g0=float(r.grade), t=t, y=y, last=float(t.max()), feature_row=int(r.feature_row), visno=int(r.visno),
                 late0=bool(r.grade >= LATE_THRESHOLD), traj=traj_group(y))
        for e in EVS:
            k = event(t, y, e)
            d[e] = k is not None; d[e + "_T"] = float(t[k]) if k is not None else float(t.max())
            d[e + "_lo"] = float(t[k - 1]) if k is not None else np.nan
        PAT[r.pid] = d
    C = load_cohort()
    for p, d in PAT.items():
        assert d["g0"] == C.loc[p, "first_grade"]
        if d["late0"]:
            continue
        assert d["E2"] == bool(C.loc[p, "e2"]) and d["E1"] == bool(C.loc[p, "e1"])
    return PAT


def forecasts_from_csv(path, PAT, pids=None):
    P = pd.read_csv(path); out = {}
    for pid, q in P.groupby("pid", sort=False):
        if pids is not None and pid not in pids:
            continue
        q = q.sort_values("t"); v = q[q.is_visit]
        f = dict(vt=v.t.values.astype(float), vy=v.y_true.values.astype(float), vp=v.y_pred.values.astype(float),
                 gt=q.t.values.astype(float), gp=q.y_pred.values.astype(float))
        assert np.allclose(f["vt"], PAT[pid]["t"]) and np.allclose(f["vy"], PAT[pid]["y"]), pid
        assert np.all(np.isin(G, np.round(f["gt"], 4))), pid
        out[pid] = f
    return out


def seed_mean(Fs):
    a = {}
    for pid in Fs[0]:
        a[pid] = dict(Fs[0][pid]); a[pid]["vp"] = np.mean([F[pid]["vp"] for F in Fs], 0); a[pid]["gp"] = np.mean([F[pid]["gp"] for F in Fs], 0)
    return a


def at(f, times):
    times = np.atleast_1d(np.asarray(times, float))
    assert np.all(np.isin(np.round(times, 4), np.round(f["gt"], 4))), "time not a query point"
    return np.interp(times, f["gt"], f["gp"])


class Scorer:
    def __init__(self, kind, f=None, extra=None):
        self.kind, self.f, self.x = kind, f, extra

    def score_at(self, p, t, ev):
        if self.kind in ("grade", "cox"):
            return float(at(self.f[p], t)[0])
        return self.x.get((ev, t), {}).get(p, np.nan)

    def risk_fu(self, p, ev, PAT):
        d = PAT[p]
        if self.kind == "grade":
            f = self.f[p]; tt = f["gt"][f["gt"] <= d["last"] + 1e-9]
            return float(np.max(at(f, tt)))
        if self.kind == "cox":
            return float(at(self.f[p], d["last"])[0])
        return np.nan


def boot_weights(n, nb=N_BOOT):
    rng = np.random.default_rng(BOOT_SEED)
    return np.stack([np.bincount(rng.choice(n, n, replace=True), minlength=n) for _ in range(nb)]).astype(float)


def auc_w(s, case, ctrl, W):
    ci, ki = np.where(case)[0], np.where(ctrl)[0]
    if len(ci) == 0 or len(ki) == 0:
        return np.full(W.shape[0], np.nan)
    M = (s[ci][:, None] > s[ki][None, :]) + 0.5 * (s[ci][:, None] == s[ki][None, :])
    num = np.einsum("bi,ij,bj->b", W[:, ci], M, W[:, ki]); den = W[:, ci].sum(1) * W[:, ki].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def cindex_w(T, E, risk, W):
    T, E, risk = np.asarray(T, float), np.asarray(E, bool), np.asarray(risk, float)
    comp = E[:, None] & ((T[None, :] > T[:, None]) | ((T[None, :] == T[:, None]) & ~E[None, :]))
    conc = comp * ((risk[:, None] > risk[None, :]) + 0.5 * (risk[:, None] == risk[None, :]))
    num = np.einsum("bi,ij,bj->b", W, conc, W); den = np.einsum("bi,ij,bj->b", W, comp.astype(float), W)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def wmedian(x, w):
    k = w > 0
    return float(np.median(np.repeat(x[k], w[k].astype(int)))) if k.any() else np.nan


def first_cross(vals, c):
    h = np.where(vals >= c)[0]
    return G[h[0]] if len(h) else np.inf


def case_ctrl(PAT, P, ev, h):
    case = np.array([PAT[p][ev] and PAT[p][ev + "_T"] <= h for p in P])
    ctrl = np.array([(not PAT[p][ev] or PAT[p][ev + "_T"] > h) and
                     np.any((PAT[p]["t"] >= h) & (PAT[p]["y"] < LATE_THRESHOLD) & (PAT[p]["t"] < (PAT[p][ev + "_T"] if PAT[p][ev] else np.inf)))
                     for p in P])
    assert not np.any(case & ctrl)
    return case, ctrl


def valid_times(PAT, P):
    out = {}
    for ev in EVS:
        ts, mc = [], []
        for t in TB:
            case, ctrl = case_ctrl(PAT, P, ev, t)
            if case.sum() >= MIN_CASES_T and ctrl.sum() > 0:
                ts.append(t); mc.append(int(case.sum()))
        out[ev] = ts; out[ev + "_min_cases"] = min(mc) if mc else 0
    return out


def metrics(sc, P, W, PAT, tv):
    n = len(P); out = {}
    if sc.kind == "grade":
        sabs, ssq, nf, down = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        for i, p in enumerate(P):
            f = sc.f[p]; k = f["vt"] > 0; e = f["vp"][k] - f["vy"][k]
            sabs[i], ssq[i], nf[i] = np.abs(e).sum(), (e ** 2).sum(), k.sum()
            down[i] = float(np.any(np.diff(at(f, G)) < -DROP_TOL))
        with np.errstate(invalid="ignore", divide="ignore"):
            out["mae"] = (W @ sabs) / (W @ nf); out["rmse"] = np.sqrt((W @ ssq) / (W @ nf))
            up = np.array([PAT[p]["traj"] == "Only up" for p in P])
            out["mae_up"] = (W[:, up] @ sabs[up]) / (W[:, up] @ nf[up]) if up.any() else np.full(W.shape[0], np.nan)
            out["down"] = (W @ down) / W.sum(1)
    for ev in EVS:
        for H in HS:
            case, ctrl = case_ctrl(PAT, P, ev, H)
            s = np.array([sc.score_at(p, H, ev) for p in P])
            out[f"auc{H:g}_{ev}"] = auc_w(s, case, ctrl, W) if np.isfinite(s).all() else np.full(W.shape[0], np.nan)
        T = np.array([PAT[p][ev + "_T"] for p in P]); E = np.array([PAT[p][ev] for p in P])
        if sc.kind in ("grade", "cox"):
            r = np.array([sc.risk_fu(p, ev, PAT) for p in P])
            out[f"c_{ev}"] = cindex_w(T, E, r, W)
            A = []
            for t in tv[ev]:
                case, ctrl = case_ctrl(PAT, P, ev, t)
                A.append(auc_w(np.array([sc.score_at(p, t, ev) for p in P]), case, ctrl, W))
            out[f"iauc_{ev}"] = np.nanmean(np.array(A), 0) if A else np.full(W.shape[0], np.nan)
            for t, a in zip(tv[ev], A):
                out[f"auct{t:g}_{ev}"] = a
            c = FIX_P if sc.kind == "cox" else FIX_C
            pr = np.array([first_cross(at(sc.f[p], G), c) for p in P])
            lo = np.array([PAT[p][ev + "_lo"] for p in P]); last = np.array([PAT[p]["last"] for p in P])
            det = E & np.isfinite(pr); ontime = det & (pr >= lo - 1) & (pr <= T + 1)
            err = np.where(pr < lo, pr - lo, np.where(pr > T, pr - T, 0.0)); err = np.where(det, err, np.nan)
            fa = (~E) & (pr <= last)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"det_{ev}"] = (W @ det) / (W @ E); out[f"ontime_{ev}"] = (W @ ontime) / (W @ E); out[f"fa_{ev}"] = (W @ fa) / (W @ ~E)
            out[f"medabs_{ev}"] = np.array([wmedian(np.abs(err[det]), w[det]) for w in W])
    return out


def n_events(metric, PAT, P, tv):
    ev = metric.split("_")[-1]
    if ev not in EVS:
        return np.nan
    if metric.startswith("auct"):
        return int(case_ctrl(PAT, P, ev, float(metric[4:].split("_")[0]))[0].sum())
    if metric.startswith("auc"):
        return int(case_ctrl(PAT, P, ev, float(metric[3:].split("_")[0]))[0].sum())
    if metric.startswith("iauc"):
        return int(tv[ev + "_min_cases"])
    return int(sum(PAT[p][ev] for p in P))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - rank) * p[i])); adj[i] = run
    return adj


def paired_p(d):
    d = d[np.isfinite(d)]
    if not len(d):
        return np.nan
    return min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (len(d) + 1))


ARCH = {"lnn": "LNN", "rnn": "RNN", "lstm": "LSTM", "no": "NO"}
BASE_LABEL = {"first": "First grade alone", "cox": "Cox (first grade + PCA)", "svm": "Linear SVM (calibrated)",
              "logreg": "Logistic regression (first grade + 10 PCA)"}


def label(key):
    if key in BASE_LABEL:
        return BASE_LABEL[key]
    m, g, v = key.split("_")
    return f"{ARCH[m]}{' + grade' if g == 'grade' else ' (image only)'} {v}"


def split_key(key):
    m, g, v = key.split("_"); return m, int(g == "grade"), v


def cv_seeds(key):
    m, g, v = split_key(key)
    return (0,) if (v == "V0" and m in ("rnn", "lstm", "no") and ONE_SEED_FLAG.exists()) else SEEDS


def _cv_run_dir(name):
    for root in (RESULTS_CV, RESULTS_LOSSSEL):
        d = root / "runs" / name
        if d.exists():
            return d
    raise FileNotFoundError(f"CV run not found in crossvalidation/ or lossselection/: {name}")


def cv_forecasts(key, PAT):
    m, g, v = split_key(key); F, Ff = {}, {}
    for s in cv_seeds(key):
        F[s] = {}
        for k in FOLDS:
            d = _cv_run_dir(run_name(m, g, v, k, s))
            f = forecasts_from_csv(d / "predictions_heldout.csv", PAT)
            Ff[(k, s)] = f; F[s].update(f)
    return F, Ff


def first_scorer(PAT, P):
    f = {}
    for p in P:
        d = PAT[p]; gt = np.unique(np.round(np.concatenate([G, d["t"]]), 4))
        f[p] = dict(vt=d["t"], vy=d["y"], vp=np.full(len(d["t"]), d["g0"]), gt=gt, gp=np.full(len(gt), d["g0"]))
    return Scorer("grade", f)


def cv_baselines(PAT):
    cox, svm, lr, per = {}, {}, {}, {}
    for k in FOLDS:
        M = pickle.load(open(RESULTS_FINAL / "baselines" / "cv" / f"fold{k}.pkl", "rb")); pr = M["pred"]
        cox.update(pr["cox"])
        for key, src in (("svm", svm), ("logreg", lr)):
            for eh, dct in pr[key].items():
                src.setdefault(eh, {}).update(dct)
        per[k] = dict(cox=Scorer("cox", pr["cox"]), svm=Scorer("prob", extra=pr["svm"]), logreg=Scorer("prob", extra=pr["logreg"]))
    return dict(cox=Scorer("cox", cox), svm=Scorer("prob", extra=svm), logreg=Scorer("prob", extra=lr)), per


def summarise(SC, P, PAT, boot=True):
    tv = valid_times(PAT, P); n = len(P); ONE = np.ones((1, n)); W = boot_weights(n) if boot else None
    rows, B = [], {}
    for key, seeds in SC.items():
        sl = [s for s in seeds if s != "a"]
        pts = [metrics(seeds[s], P, ONE, PAT, tv) for s in sl]
        pa = metrics(seeds["a"], P, ONE, PAT, tv)
        if boot:
            bts = [metrics(seeds[s], P, W, PAT, tv) for s in sl]; B[key] = metrics(seeds["a"], P, W, PAT, tv)
        for met in pts[0]:
            v = np.array([x[met][0] for x in pts], float)
            r = dict(model=key, label=label(key), metric=met, n=n, n_events=n_events(met, PAT, P, tv), n_seeds=len(sl),
                     mean=float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                     sd=float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan,
                     per_seed=";".join(f"{x:.4f}" for x in v), seedmean=float(pa[met][0]))
            if boot:
                b = np.nanmean(np.array([x[met] for x in bts], float), 0) if len(sl) > 1 else bts[0][met]; ok = np.isfinite(b)
                r["lo"], r["hi"] = (float(np.percentile(b[ok], 2.5)), float(np.percentile(b[ok], 97.5))) if ok.any() else (np.nan, np.nan)
            r["flag_lt10"] = bool(np.isfinite(r["n_events"]) and r["n_events"] < MIN_EVENTS_FLAG)
            rows.append(r)
    return pd.DataFrame(rows), B, tv


def tests_list(vstar):
    T = [(f"lnn_grade_{vstar}", f"{m}_grade_{vstar}", "iauc_E2", 1) for m in ("rnn", "lstm", "no")]
    T += [(f"lnn_grade_{vstar}", "first", "iauc_E2", 2)]
    T += [("lnn_grade_V1", "lnn_grade_V0", "down", 3), ("lnn_grade_V1", "lnn_grade_V0", "mae", 3)]
    T += [(f"lnn_grade_{vstar}", "lnn_grade_V1", "iauc_E2", 4), (f"lnn_grade_{vstar}", "lnn_grade_V1", "medabs_E2", 4)]
    T += [(f"lnn_grade_{vstar}", b, m, 5) for b in ("cox", "logreg") for m in ("auc2_E2", "auc5_E2")]
    T += [("lnn_grade_V1", "lnn_img_V1", "mae", 6)]
    return T


def run_tests(SC, B, P, PAT, tv, vstar):
    ONE = np.ones((1, len(P))); rows = []
    for a, b, met, item in tests_list(vstar):
        if a == b:
            rows.append(dict(item=item, model_a=a, model_b=b, metric=met, note="n/a (V* = V1)")); continue
        va = metrics(SC[a]["a"], P, ONE, PAT, tv)[met][0]; vb = metrics(SC[b]["a"], P, ONE, PAT, tv)[met][0]
        d = B[a][met] - B[b][met]; dd = d[np.isfinite(d)]
        rows.append(dict(item=item, model_a=a, model_b=b, metric=met, value_a=va, value_b=vb, diff=va - vb,
                         lo=np.percentile(dd, 2.5) if len(dd) else np.nan, hi=np.percentile(dd, 97.5) if len(dd) else np.nan,
                         p=paired_p(d), n_boot_valid=len(dd), n_events=n_events(met, PAT, P, tv)))
    T = pd.DataFrame(rows)
    k = T.p.notna() if "p" in T else pd.Series(False, index=T.index)
    T.loc[k, "p_holm"] = holm(T.loc[k, "p"].values)
    return T
