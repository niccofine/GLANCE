"""Choose the best LNN variant (rule fixed in advance), then retrain RNN, LSTM and NO + grade with that variant (5 folds x 3 seeds, 45 runs; none if V* = V1).
- Among V1-V4: highest out-of-fold integrated AUC (E2), mean over seeds.
- If two are within 0.005, take the one with the lower median absolute timing error.
- The chosen variant must have <= 5% forecasts going down (else move to the next by the same rule; none qualifies -> V1).

Can be run on its own or through scripts/driver.sh:
  python -m scripts.lossselection select
  python -m scripts.lossselection worker --gpu 0
  python -m scripts.lossselection worker --gpu 1
  python -m scripts.lossselection check
"""
import argparse
import json
import sys
from datetime import datetime
import numpy as np
from glance.config import RESULTS_LOSSSEL
from glance.data import rounds
from glance.evaluate import Scorer, cv_forecasts, cv_seeds, patients, seed_mean, summarise
from glance.models import jobs, rdir, run_name, worker

STAGE = "lossselection"
SEL = RESULTS_LOSSSEL / "selection.json"


def cohort(PAT):
    allp = rounds()[0][0]
    return [p for p in allp if not PAT[p]["late0"]]


def model_SC(keys, PAT):
    SC = {}
    for k in keys:
        F, _ = cv_forecasts(k, PAT); ss = cv_seeds(k)
        SC[k] = {s: Scorer("grade", F[s]) for s in ss}; SC[k]["a"] = Scorer("grade", seed_mean([F[s] for s in ss]))
    return SC


def select():
    RESULTS_LOSSSEL.mkdir(parents=True, exist_ok=True)
    PAT = patients(); P = cohort(PAT)
    keys = [f"lnn_grade_V{i}" for i in (1, 2, 3, 4)]
    SC = model_SC(keys, PAT)
    M, _, tv = summarise(SC, P, PAT, boot=False)
    piv = M[M.metric.isin(["iauc_E2", "medabs_E2", "down"])].pivot(index="model", columns="metric", values="mean")
    rows = [dict(variant=k.split("_")[-1], **piv.loc[k].to_dict()) for k in keys]
    remaining = list(rows); steps = []
    pick = None
    while remaining:
        best = max(r["iauc_E2"] for r in remaining)
        cand = [r for r in remaining if r["iauc_E2"] >= best - 0.005]
        pick = min(cand, key=lambda r: (r["medabs_E2"] if np.isfinite(r["medabs_E2"]) else np.inf, -r["iauc_E2"])) if len(cand) > 1 else cand[0]
        steps.append(dict(best_iauc=best, candidates=[c["variant"] for c in cand], picked=pick["variant"], down=pick["down"]))
        if pick["down"] <= 0.05:
            break
        remaining = [r for r in remaining if r is not pick]; pick = None
    vstar = pick["variant"] if pick else "V1"
    res = dict(V_star=vstar, rule="max mean-over-seeds OOF iAUC E2 among V1-V4; within 0.005 -> lower median |timing error| E2; "
               "must have <= 5% going down (else next by the same rule; none -> V1)", table=rows, steps=steps,
               n_patients=len(P), iauc_times=tv["E2"], decided=datetime.now().isoformat(timespec="seconds"))
    json.dump(res, open(SEL, "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))


def check():
    miss = [j for j in jobs(STAGE) if not (rdir(STAGE, j) / "done.json").exists()]
    print(len(miss), "missing of", len(jobs(STAGE)))
    for j in miss:
        print(" ", run_name(j["model"], j["grade"], j["variant"], j["fold"], j["seed"]))
    return len(miss) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("select", "worker", "check", "list"))
    ap.add_argument("--gpu", type=int)
    a = ap.parse_args()
    if a.cmd == "select":
        select()
    elif a.cmd == "worker":
        worker(STAGE, a.gpu)
    elif a.cmd == "check":
        sys.exit(0 if check() else 1)
    elif a.cmd == "list":
        for j in jobs(STAGE):
            print(run_name(j["model"], j["grade"], j["variant"], j["fold"], j["seed"]), (rdir(STAGE, j) / "done.json").exists())


if __name__ == "__main__":
    main()
