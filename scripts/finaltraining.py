"""Stage 3: final models.
- epochs: final epochs per model/variant = median best epoch from the cross-validation runs (crossvalidation/ and, for the chosen V*, lossselection/) -> results/finaltraining/final_epochs.json
- train (worker): retrain on all of construction set (3 seeds; fixed epochs): LNN V0, V1, V* (+ grade), LNN image only V1; RNN, LSTM, NO V0, V1, V* -> 39 runs
- baselines (cv / final): first grade alone; Cox (first grade + RET-CLIP PCA, k chosen by inner cross-validation on construction, max 20); linear SVM for +2y / +5y (calibrated by cross-validation); logistic regression for late by 2y / 5y (first grade + 10 PCA). 'cv' fits per cross-validation round on the round's training patients (used by evaluation.py for the CV metrics); 'final' fits on all CONSTR (used for the TEST predictions). Never reads test set.

Can be run on its own or through scripts/driver.sh:
  python -m scripts.finaltraining epochs
  python -m scripts.finaltraining worker --gpu 0
  python -m scripts.finaltraining worker --gpu 1
  python -m scripts.finaltraining check
  python -m scripts.finaltraining baselines cv
  python -m scripts.finaltraining baselines final
"""
import argparse
import json
import pickle
import sys
import time
import warnings
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from glance.config import LATE_THRESHOLD, RESULTS_LOSSSEL, RESULTS_FINAL
from glance.data import MinMaxNorm, rounds, load_samples, load_X
from glance.evaluate import patients, case_ctrl, cv_seeds, G, HS, EVS, _cv_run_dir
from glance.models import jobs, rdir, run_name, worker
STAGE = "finaltraining"


def epochs():
    RESULTS_FINAL.mkdir(parents=True, exist_ok=True)
    vs = json.load(open(RESULTS_LOSSSEL / "selection.json"))["V_star"]
    keys = ["lnn|1|V0", "lnn|1|V1", f"lnn|1|{vs}", "lnn|0|V1"] + [f"{m}|1|{v}" for m in ("rnn", "lstm", "no") for v in ("V0", "V1", vs)]
    keys = list(dict.fromkeys(keys)); out, info = {}, {}
    for k in keys:
        m, g, v = k.split("|")
        ss = cv_seeds(f"{m}_{'grade' if int(g) else 'img'}_{v}")
        e = [json.load(open(_cv_run_dir(run_name(m, int(g), v, f, s)) / "done.json"))["n_epochs_best"] for f in (1, 2, 3, 4, 5) for s in ss]
        out[k] = int(np.median(e)); info[k] = e
    json.dump(out, open(RESULTS_FINAL / "final_epochs.json", "w"), indent=1)
    json.dump(dict(epochs=out, all_best_epochs=info), open(RESULTS_FINAL / "final_epochs_detail.json", "w"), indent=1)
    print(out)


def check():
    miss = [j for j in jobs(STAGE) if not (rdir(STAGE, j) / "done.json").exists()]
    print(len(miss), "missing of", len(jobs(STAGE)))
    for j in miss:
        print(" ", run_name(j["model"], j["grade"], j["variant"], j["fold"], j["seed"]))
    return len(miss) == 0


BASE_OUT = RESULTS_FINAL / "baselines"; KMAX, PEN, NPCA_LR, SEED = 20, 0.1, 10, 2027


def feats(PAT, P, X, sc):
    return sc(X[np.array([PAT[p]["feature_row"] for p in P])])


def cox_df(PAT, P, Z, pca, k):
    d = pd.DataFrame({"g0": [PAT[p]["g0"] for p in P]})
    if k:
        Zp = pca.transform(Z)[:, :k]
        for i in range(k):
            d[f"pc{i + 1}"] = Zp[:, i]
    return d


def fit_cox(PAT, P, Z, k):
    pca = PCA(n_components=KMAX, random_state=0).fit(Z) if k else None
    d = cox_df(PAT, P, Z, pca, k); d["T"] = [PAT[p]["E2_T"] for p in P]; d["E"] = [int(PAT[p]["E2"]) for p in P]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter(penalizer=PEN).fit(d, "T", "E")
    return dict(cph=cph, pca=pca, k=k)


def cox_curve(m, PAT, P, Z, times):
    d = cox_df(PAT, P, Z, m["pca"], m["k"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SF = m["cph"].predict_survival_function(d, times=times).values
    return {p: dict(gt=times, gp=1.0 - SF[:, j]) for j, p in enumerate(P)}


def choose_k(PAT, P, Z):
    E = np.array([int(PAT[p]["E2"]) for p in P]); res = {k: [] for k in range(KMAX + 1)}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Z, E):
        Pa, Pb = [P[i] for i in a], [P[i] for i in b]
        for k in range(KMAX + 1):
            m = fit_cox(PAT, Pa, Z[a], k)
            r = m["cph"].predict_partial_hazard(cox_df(PAT, Pb, Z[b], m["pca"], k)).values
            res[k].append(concordance_index([PAT[p]["E2_T"] for p in Pb], -r, [PAT[p]["E2"] for p in Pb]))
    mean = {k: float(np.mean(v)) for k, v in res.items()}
    best = max(mean.values()); k = min(k for k, v in mean.items() if v == best)
    return k, mean


def svm_samples(S, pids, H):
    rows = []
    for r in S[S.pid.isin(pids)].itertuples():
        if r.grade >= LATE_THRESHOLD:
            continue
        d = np.abs(r.t - H); ok = np.where(d <= 0.5)[0]
        if not len(ok):
            continue
        j = ok[np.argmin(d[ok])]
        rows.append((r.pid, r.feature_row, int(r.y[j] >= LATE_THRESHOLD)))
    return pd.DataFrame(rows, columns=["pid", "feature_row", "label"])


def fit_all(trp, PAT, S, X, sc):
    tr = [p for p in trp if not PAT[p]["late0"]]; Z = feats(PAT, tr, X, sc)
    k, kmean = choose_k(PAT, tr, Z)
    cox = fit_cox(PAT, tr, Z, k)
    svm, svm_info = {}, {}
    for H in HS:
        D = svm_samples(S, trp, H); Xs = sc(X[D.feature_row.values]); y = D.label.values
        splits = list(GroupKFold(5).split(Xs, y, groups=D.pid.values))
        svm[H] = CalibratedClassifierCV(SVC(kernel="linear"), method="sigmoid", cv=splits).fit(Xs, y)
        svm_info[H] = dict(n=len(D), n_pos=int(y.sum()), n_patients=int(D.pid.nunique()))
    pca = PCA(n_components=NPCA_LR, random_state=0).fit(Z)
    Ftr = np.column_stack([[PAT[p]["g0"] for p in tr], pca.transform(Z)]); st = StandardScaler().fit(Ftr)
    lr, lr_info = {}, {}
    for ev in EVS:
        for H in HS:
            case, ctrl = case_ctrl(PAT, tr, ev, H); use = case | ctrl
            lr[(ev, H)] = LogisticRegression(max_iter=5000).fit(st.transform(Ftr[use]), case[use].astype(int))
            lr_info[f"{ev}_{H:g}"] = dict(n=int(use.sum()), cases=int(case[use].sum()))
    return dict(scaler_min=sc.min, scaler_scale=sc.scale, cox=cox, cox_k=k, cox_k_cv=kmean, svm=svm, svm_info=svm_info,
                lr_pca=pca, lr_scaler=st, lr=lr, lr_info=lr_info, n_train_notlate=len(tr), train_pids=list(trp))


def predict_all(M, PAT, P, X):
    sc = MinMaxNorm(np.zeros(1), np.ones(1)); sc.min, sc.scale = M["scaler_min"], M["scaler_scale"]
    Z = feats(PAT, P, X, sc)
    times = np.unique(np.round(np.concatenate([G, np.concatenate([PAT[p]["t"] for p in P])]), 4))
    cox = cox_curve(M["cox"], PAT, P, Z, times)
    prob = {}
    for H in HS:
        s = M["svm"][H].predict_proba(Z)[:, 1]
        for ev in EVS:
            prob.setdefault("svm", {})[(ev, H)] = dict(zip(P, s))
    F = np.column_stack([[PAT[p]["g0"] for p in P], M["lr_pca"].transform(Z)])
    for (ev, H), m in M["lr"].items():
        prob.setdefault("logreg", {})[(ev, H)] = dict(zip(P, m.predict_proba(M["lr_scaler"].transform(F))[:, 1]))
    return dict(cox=cox, svm=prob["svm"], logreg=prob["logreg"])


def baselines(mode):
    PAT = patients(); S = load_samples(); X = load_X(); R = rounds()
    (BASE_OUT / mode).mkdir(parents=True, exist_ok=True)
    keys = [1, 2, 3, 4, 5] if mode == "cv" else [0]
    for k in keys:
        f = BASE_OUT / mode / (f"fold{k}.pkl" if mode == "cv" else "models.pkl")
        if f.exists():
            print("exists:", f); continue
        t0 = time.time(); trp, hop = R[k]
        sc = MinMaxNorm.fit(X[S[S.pid.isin(trp)].feature_row.values])
        M = fit_all(trp, PAT, S, X, sc)
        if mode == "cv":
            M["pred"] = predict_all(M, PAT, hop, X); M["heldout_pids"] = list(hop)
        pickle.dump(M, open(f, "wb"))
        print(f"{mode} {k}: cox k = {M['cox_k']} | svm {M['svm_info']} | lr {M['lr_info']} | {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("epochs", "worker", "check", "list", "baselines"))
    ap.add_argument("mode", nargs="?", choices=("cv", "final"))
    ap.add_argument("--gpu", type=int)
    a = ap.parse_args()
    if a.cmd == "epochs":
        epochs()
    elif a.cmd == "worker":
        worker(STAGE, a.gpu)
    elif a.cmd == "check":
        sys.exit(0 if check() else 1)
    elif a.cmd == "list":
        for j in jobs(STAGE):
            print(run_name(j["model"], j["grade"], j["variant"], j["fold"], j["seed"]), (rdir(STAGE, j) / "done.json").exists())
    elif a.cmd == "baselines":
        assert a.mode, "baselines needs a mode: cv or final"
        baselines(a.mode)


if __name__ == "__main__":
    main()
