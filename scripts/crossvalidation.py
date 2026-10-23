""" Cross-validation on construction set (5 folds x 3 seeds).
- LNN + grade: V0, V1, V2, V3, V4
- LNN image only: V1
- RNN, LSTM, NO + grade: V0, V1

Can be run on its own (one worker per GPU) or through scripts/driver.sh.
  python -m scripts.crossvalidation worker --gpu 0
  python -m scripts.crossvalidation worker --gpu 1
  python -m scripts.crossvalidation check
"""
import argparse
import sys
from glance.models import jobs, rdir, run_name, worker

STAGE = "crossvalidation"

def check():
    miss = [j for j in jobs(STAGE) if not (rdir(STAGE, j) / "done.json").exists()]
    print(len(miss), "missing of", len(jobs(STAGE)))
    for j in miss:
        print(" ", run_name(j["model"], j["grade"], j["variant"], j["fold"], j["seed"]))
    return len(miss) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("worker", "check", "list"))
    ap.add_argument("--gpu", type=int)
    a = ap.parse_args()
    if a.cmd == "worker":
        worker(STAGE, a.gpu)
    elif a.cmd == "check":
        sys.exit(0 if check() else 1)
    elif a.cmd == "list":
        for j in jobs(STAGE):
            print(run_name(j["model"], j["grade"], j["variant"], j["fold"], j["seed"]), (rdir(STAGE, j) / "done.json").exists())


if __name__ == "__main__":
    main()
