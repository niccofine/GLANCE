""" Evaluation (cross-validation out-of-fold and test set, the same code).
  cv         : pooled cross-validation out-of-fold metrics + paired tests (construction set, never touches test set) -> results/evaluation/cv/{metrics_cv.csv, metrics_cv_by_fold.csv, metrics_cv_folds.csv, tests_cv.csv, cv_info.json}
  test       : checks every finaltraining run and baseline model exists, predicts test set once, metrics + paired tests -> runs/<run>/predictions_test.csv, results/evaluation/test/{metrics_test.csv, tests_test.csv, test_info.json, TEST_OPENED.txt, baseline_predictions_test.pkl}
  test_extra : extra test for the posthoc final models -> extra_runs/<run>/predictions_test.csv,results/evaluation/test_extra/{TEST_OPENED.txt, test_extra_info.json}
  properties : parameters, seconds per epoch, test set seed SD (existing files only) -> results/evaluation/properties/model_properties.csv

Can be run on its own or through scripts/driver.sh:
  python -m scripts.evaluation cv
  python -m scripts.evaluation test
  python -m scripts.evaluation test_extra
  python -m scripts.evaluation properties
"""
import json
import pickle
import sys
from datetime import datetime
import numpy as np, pandas as pd, torch
from glance.config import RESULTS_LOSSSEL, RESULTS_FINAL, RESULTS_EVAL
from glance.data import MinMaxNorm, rounds, load_samples, load_X, test_pids, V2Dataset, SEEDS, FOLDS
from glance.evaluate import Scorer, cv_baselines, cv_forecasts, cv_seeds, first_scorer, forecasts_from_csv, label, metrics, patients, run_tests, seed_mean, split_key, summarise, valid_times, _cv_run_dir, at
from glance.models import build, predict, run_name
from glance.irregular import SETTINGS as IRREG_SETTINGS, thin as irreg_thin, build as irreg_build, predict as irreg_predict
from glance.annual import annualise, AnnualRNN, predict as annual_predict
from glance.continuous import build as continuous_build, rdir as continuous_rdir
from scripts.finaltraining import predict_all, extra_rdir

CV_OUT = RESULTS_EVAL / "cv"
TEST_OUT = RESULTS_EVAL / "test"
EXTRA_TEST_OUT = RESULTS_EVAL / "test_extra"
PROP_OUT = RESULTS_EVAL / "properties"
EXTRA_TEST_JOBS = [("nogrid", m) for m in ("rnn", "lstm", "no", "lnn", "tlstm")] + [("annual", m) for m in ("rnn", "lstm")] + [("rs", "lnn")]


def cohort(PAT):
    allp = rounds()[0][0]
    return [p for p in allp if not PAT[p]["late0"]]


def model_SC(keys, PAT):
    SC, SCf = {}, {}
    for k in keys:
        F, Ff = cv_forecasts(k, PAT); ss = cv_seeds(k)
        SC[k] = {s: Scorer("grade", F[s]) for s in ss}; SC[k]["a"] = Scorer("grade", seed_mean([F[s] for s in ss]))
        SCf[k] = {fs: Scorer("grade", f) for fs, f in Ff.items()}
    return SC, SCf


def cv():
    CV_OUT.mkdir(parents=True, exist_ok=True)
    PAT = patients(); P = cohort(PAT); vs = json.load(open(RESULTS_LOSSSEL / "selection.json"))["V_star"]
    keys = [f"lnn_grade_V{i}" for i in range(5)] + ["lnn_img_V1"] + [f"{m}_grade_{v}" for m in ("rnn", "lstm", "no") for v in dict.fromkeys(("V0", "V1", vs))]
    SC, SCf = model_SC(keys, PAT)
    base, base_f = cv_baselines(PAT)
    SC["first"] = {0: first_scorer(PAT, P)}; SC["first"]["a"] = SC["first"][0]
    for b in ("cox", "svm", "logreg"):
        SC[b] = {0: base[b], "a": base[b]}
    M, B, tv = summarise(SC, P, PAT, boot=True)
    M.to_csv(CV_OUT / "metrics_cv.csv", index=False)
    f_of = {p: k for k in FOLDS for p in rounds()[k][1]}; rows = []
    for k in FOLDS:
        Pk = [p for p in P if f_of[p] == k]; tvk = valid_times(PAT, Pk); ONE = np.ones((1, len(Pk)))
        for key in SC:
            if key in SCf:
                vals = [metrics(SCf[key][(k, s)], Pk, ONE, PAT, tvk) for s in cv_seeds(key)]
            elif key == "first":
                vals = [metrics(first_scorer(PAT, Pk), Pk, ONE, PAT, tvk)]
            else:
                vals = [metrics(base_f[k][key], Pk, ONE, PAT, tvk)]
            for met in vals[0]:
                rows.append(dict(model=key, fold=k, metric=met, value=float(np.nanmean([v[met][0] for v in vals]))))
    Fd = pd.DataFrame(rows); Fd.to_csv(CV_OUT / "metrics_cv_by_fold.csv", index=False)
    Fs = Fd.groupby(["model", "metric"]).value.agg(fold_mean="mean", fold_sd=lambda x: x.std(ddof=1), n_folds="count").reset_index()
    Fs.to_csv(CV_OUT / "metrics_cv_folds.csv", index=False)
    T = run_tests(SC, B, P, PAT, tv, vs); T.to_csv(CV_OUT / "tests_cv.csv", index=False)
    json.dump(dict(n_patients=len(P), events={e: int(sum(PAT[p][e] for p in P)) for e in ("E1", "E2")}, valid_times=tv, V_star=vs),
              open(CV_OUT / "cv_info.json", "w"), indent=1, default=float)
    print(M[M.metric.isin(["iauc_E2", "mae", "down"])].pivot(index="model", columns="metric", values="mean").round(3).to_string())
    print(T.to_string())


def test():
    TEST_OUT.mkdir(parents=True, exist_ok=True)
    vs = json.load(open(RESULTS_LOSSSEL / "selection.json"))["V_star"]
    E = json.load(open(RESULTS_FINAL / "final_epochs.json"))
    keys = [f"{m}_{'grade' if int(g) else 'img'}_{v}" for m, g, v in (k.split("|") for k in E)]
    for k in keys:
        m, g, v = split_key(k)
        for s in SEEDS:
            assert (RESULTS_FINAL / "runs" / run_name(m, g, v, 0, s) / "done.json").exists(), (k, s)
    assert (RESULTS_FINAL / "baselines" / "final" / "models.pkl").exists()
    stamp = datetime.now().isoformat(timespec="seconds")
    opened = TEST_OUT / "TEST_OPENED.txt"
    with open(opened, "a") as f:
        f.write(stamp + "\n")

    PAT = patients(); T = test_pids(); P = [p for p in T if not PAT[p]["late0"]]
    S = load_samples(); X = load_X(); Ft = S[S.pid.isin(T) & S["first"]]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SC = {}
    for k in keys:
        m, g, v = split_key(k); F = []
        for s in SEEDS:
            d = RESULTS_FINAL / "runs" / run_name(m, g, v, 0, s)
            z = np.load(d / "scaler.npz"); sc = MinMaxNorm(np.zeros(1), np.ones(1)); sc.min, sc.scale = z["data_min"], z["scale"]
            ds = V2Dataset(Ft, X, sc, g)
            mod = build(m, ds.x.shape[1]).to(dev)
            mod.load_state_dict(torch.load(d / "best.pt", map_location=dev)["model_state_dict"])
            pr = predict(mod, ds, dev); pr.to_csv(d / "predictions_test.csv", index=False)
            F.append(forecasts_from_csv(d / "predictions_test.csv", PAT))
        SC[k] = {s: Scorer("grade", F[i]) for i, s in enumerate(SEEDS)}; SC[k]["a"] = Scorer("grade", seed_mean(F))
        print("predicted", k, flush=True)
    M = pickle.load(open(RESULTS_FINAL / "baselines" / "final" / "models.pkl", "rb"))
    B = predict_all(M, PAT, T, X)
    pickle.dump(B, open(TEST_OUT / "baseline_predictions_test.pkl", "wb"))
    SC["first"] = {0: first_scorer(PAT, P)}; SC["first"]["a"] = SC["first"][0]
    SC["cox"] = {0: Scorer("cox", B["cox"])}; SC["svm"] = {0: Scorer("prob", extra=B["svm"])}; SC["logreg"] = {0: Scorer("prob", extra=B["logreg"])}
    for b in ("cox", "svm", "logreg"):
        SC[b]["a"] = SC[b][0]
    Mt, Bt, tv = summarise(SC, P, PAT, boot=True); Mt.to_csv(TEST_OUT / "metrics_test.csv", index=False)
    Tt = run_tests(SC, Bt, P, PAT, tv, vs); Tt.to_csv(TEST_OUT / "tests_test.csv", index=False)
    json.dump(dict(opened=stamp, n_test=len(T), n_eval=len(P), events={e: int(sum(PAT[p][e] for p in P)) for e in ("E1", "E2")},
                   valid_times=tv, V_star=vs, keys=keys, cox_k=M["cox_k"]), open(TEST_OUT / "test_info.json", "w"), indent=1, default=float)
    print(Mt[Mt.metric.isin(["iauc_E2", "mae", "down", "auc2_E2", "auc5_E2"])].pivot(index="model", columns="metric", values="mean").round(3).to_string())
    print(Tt.to_string())


def _extra_dir(kind, model, seed):
    return extra_rdir(kind, model, seed) if kind != "rs" else continuous_rdir("rs", 0, seed)


def test_extra():
    assert not info.exists(), "TEST already predicted once for the extra final models"
    for kind, m in EXTRA_TEST_JOBS:
        for s in SEEDS:
            assert (_extra_dir(kind, m, s) / "done.json").exists(), f"final run missing: {kind} {m} seed {s}"
    stamp = datetime.now().isoformat(timespec="seconds")
    with open(info, "a") as f:
        f.write(f"{stamp} opened for the extra final models (scripts.evaluation.test_extra)\n")
    S = load_samples(); X = load_X(); T = test_pids(); Ft = S[S.pid.isin(T) & S["first"]].reset_index(drop=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    done = []
    for kind, m in EXTRA_TEST_JOBS:
        for s in SEEDS:
            d = _extra_dir(kind, m, s); z = np.load(d / "scaler.npz"); sc = MinMaxNorm(np.zeros(1), np.ones(1)); sc.min, sc.scale = z["data_min"], z["scale"]
            sd = torch.load(d / "best.pt", map_location=dev)["model_state_dict"]
            if kind == "nogrid":
                mod = irreg_build(m, 1025).to(dev); mod.load_state_dict(sd); mod.eval()
                ds = V2Dataset(irreg_thin(Ft, IRREG_SETTINGS["S0"], seed=s), X, sc, 1)
                irreg_predict(mod, ds, dev).to_csv(d / "predictions_test.csv", index=False)
            elif kind == "annual":
                mod = AnnualRNN({"rnn": "RNN", "lstm": "LSTM"}[m], 1025).to(dev); mod.load_state_dict(sd); mod.eval()
                annual_predict(mod, V2Dataset(annualise(Ft), X, sc, 1), Ft, dev).to_csv(d / "predictions_test.csv", index=False)
            else:
                mod = continuous_build("rs", 1025).to(dev); mod.load_state_dict(sd); mod.eval()
                P = predict(mod, V2Dataset(Ft, X, sc, 1), dev); assert set(P.pid) == set(T); P.to_csv(d / "predictions_test.csv", index=False)
            done.append(str(d.relative_to(RESULTS_FINAL.parent))); print("predicted", d.name, flush=True)
    json.dump(dict(opened=stamp, n_test_patients=len(T), runs=done), open(EXTRA_TEST_OUT / "test_extra_info.json", "w"), indent=1)


def properties():
    PROP_OUT.mkdir(parents=True, exist_ok=True)
    E = json.load(open(RESULTS_FINAL / "final_epochs.json"))
    keys = [f"{m}_{'grade' if int(g) else 'img'}_{v}" for m, g, v in (k.split("|") for k in E)]
    Mt = pd.read_csv(TEST_OUT / "metrics_test.csv").set_index(["model", "metric"])
    PAT = patients(); P = [p for p in test_pids() if not PAT[p]["late0"]]
    rows = []
    for k in keys:
        m, g, v = split_key(k)
        cv_r = [json.load(open(_cv_run_dir(run_name(m, g, v, f, s)) / "done.json")) for f in FOLDS for s in cv_seeds(k)]
        fin = [json.load(open(RESULTS_FINAL / "runs" / run_name(m, g, v, 0, s) / "done.json")) for s in SEEDS]
        sl = []
        for s in SEEDS:
            F = forecasts_from_csv(RESULTS_FINAL / "runs" / run_name(m, g, v, 0, s) / "predictions_test.csv", PAT)
            sl.append(np.mean([(at(F[p], 10.0)[0] - at(F[p], 8.0)[0]) / 2 > 0.1 for p in P]))
        gpu = lambda R: "; ".join(f"{n} {sum(r['gpu'] == n for r in R)}" for n in sorted({r["gpu"] for r in R}))
        rows.append(dict(model=k, label=label(k), n_params=fin[0]["n_params"],
                         cv_sec_per_epoch=float(np.mean([r["sec_per_epoch"] for r in cv_r])), cv_runs=len(cv_r), cv_gpus=gpu(cv_r),
                         final_sec_per_epoch=float(np.mean([r["sec_per_epoch"] for r in fin])), final_epochs=fin[0]["epochs_run"], final_gpus=gpu(fin),
                         test_iauc_E2_mean=Mt.loc[(k, "iauc_E2"), "mean"], test_iauc_E2_seed_sd=Mt.loc[(k, "iauc_E2"), "sd"],
                         test_mae_mean=Mt.loc[(k, "mae"), "mean"], test_mae_seed_sd=Mt.loc[(k, "mae"), "sd"],
                         rising_8_10_mean=float(np.mean(sl)), rising_8_10_sd=float(np.std(sl, ddof=1)), rising_per_seed=";".join(f"{x:.3f}" for x in sl), n_test_eval=len(P)))
    D = pd.DataFrame(rows); D.to_csv(PROP_OUT / "model_properties.csv", index=False)
    print(D.to_string())


def main():
    cmd = sys.argv[1]
    {"cv": cv, "test": test, "test_extra": test_extra, "properties": properties}[cmd]()


if __name__ == "__main__":
    main()
