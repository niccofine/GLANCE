"""Irregular-visit functions (posthoc experiment)"""
import argparse, json, os, subprocess, sys, time
from datetime import datetime
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from .config import ROOT, RESULTS_POSTHOC
from .data import round_data, horizon_weights, V2Dataset, collate_w, load_cohort, load_samples, FOLDS
from .models import LNN, NeuralOperator, head, loss, set_seed, CFG

OUT = RESULTS_POSTHOC / "irregular"; SEED = 0; VARIANT = "V4"
SETTINGS = {"S0": 0.0, "S50": 0.5}; MODELS = ("lnn", "rnn", "lstm", "tlstm", "no")


def thin(S, p, base=2027, seed=SEED):
    if p == 0:
        return S.copy()
    S = S.copy(); T, Y = [], []
    for r in S.itertuples():
        key = [base, int(r.pid[1:]), int(r.visno)] + ([seed] if seed else [])
        rng = np.random.default_rng(key)
        keep = rng.random(len(r.t) - 1) >= p
        if not keep.any():
            keep[rng.integers(len(keep))] = True
        k = np.r_[True, keep]; T.append(r.t[k]); Y.append(r.y[k])
    S["t"], S["y"] = T, Y
    return S


class StepRNN(nn.Module):
    kind = "continuous"; has_clean_output = True

    def __init__(self, mode, in_dim=1025, hidden=512):
        super().__init__()
        self.mode = mode
        self.input_projection = nn.Linear(in_dim, hidden)
        ni = 2 if mode == "TLSTM" else 1
        self.rnn_cell = nn.RNNCell(ni, hidden) if mode == "RNN" else nn.LSTMCell(ni, hidden)
        self.trajectory_projector = head(hidden)

    def project_clean(self, h):
        pr = self.trajectory_projector
        return pr[3](pr[1](pr[0](h)))

    def forward(self, x, q, return_clean=False):
        dt = torch.diff(q, dim=-1, prepend=q[:, :1])
        h = self.input_projection(x); c = torch.zeros_like(h) if self.mode != "RNN" else None
        cur = self.trajectory_projector(h); outs = [cur]; clean = [self.project_clean(h)]
        for k in range(1, q.size(1)):
            inp = torch.cat([cur, dt[:, k:k + 1]], 1) if self.mode == "TLSTM" else cur
            if self.mode == "RNN":
                h = self.rnn_cell(inp, h)
            else:
                h, c = self.rnn_cell(inp, (h, c))
            cur = self.trajectory_projector(h); outs.append(cur); clean.append(self.project_clean(h))
        p, pc = torch.cat(outs, 1), torch.cat(clean, 1)
        return (p, pc) if return_clean else p


def build(name, in_dim):
    if name == "no":
        return NeuralOperator(in_dim)
    return LNN(in_dim) if name == "lnn" else StepRNN({"rnn": "RNN", "lstm": "LSTM", "tlstm": "TLSTM"}[name], in_dim)


def run(model, x, t, m):
    if getattr(model, "has_clean_output", False):
        return model(x, t, return_clean=True)
    p = model(x, t)
    return p, p


def epoch(model, loader, dev, opt=None):
    tr = opt is not None; model.train(tr); tot = n = 0
    with torch.set_grad_enabled(tr):
        for x, t, y, m, idx, w, so in loader:
            x, t, y, m, w, so = (a.to(dev) for a in (x, t, y, m, w, so))
            p, pc = run(model, x, t, m)
            L, *_ = loss(VARIANT, p, y, m, w, t, pc, m, so)
            if tr:
                opt.zero_grad(); L.backward(); opt.step()
            tot += L.item() * x.size(0); n += x.size(0)
    return tot / n


@torch.no_grad()
def predict(model, ds, dev):
    model.eval(); rows = []
    for x, t, y, m, idx, w, so in DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_w(ds)):
        p, _ = run(model, x.to(dev), t.to(dev), m.to(dev)); p = p.cpu().numpy()
        for k, i in enumerate(idx.tolist()):
            r = ds.s.iloc[i]
            for j, (tt, yy) in enumerate(zip(r.t, r.y)):
                rows.append(dict(pid=r.pid, t=float(tt), y_true=float(yy), y_pred=float(p[k, j])))
    return pd.DataFrame(rows)


def rdir(model, setting, fold, seed=SEED):
    return OUT / "runs" / f"{model}_{setting}_f{fold}_s{seed}"


def train(model, setting, fold, seed=SEED):
    rd = rdir(model, setting, fold, seed); rd.mkdir(parents=True, exist_ok=True)
    if (rd / "done.json").exists():
        return
    set_seed(seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = load_cohort(); S = load_samples(); D = round_data(fold, C, S); p = SETTINGS[setting]
    tr, inner, ho, allt = (thin(D[k], p, seed=seed) for k in ("train", "inner", "heldout_first", "all_train"))
    bw, share = horizon_weights(allt)
    dtr, dva = V2Dataset(tr, D["X"], D["scaler"], 1, bw), V2Dataset(inner, D["X"], D["scaler"], 1, bw)
    g = torch.Generator(); g.manual_seed(seed)
    ltr = DataLoader(dtr, batch_size=1, shuffle=True, collate_fn=collate_w(dtr), generator=g)
    lva = DataLoader(dva, batch_size=64, shuffle=False, collate_fn=collate_w(dva))
    mod = build(model, dtr.x.shape[1]).to(dev)
    opt = torch.optim.AdamW(mod.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    log = open(rd / "train.log", "w"); best, best_ep, bad, hist, t0 = np.inf, -1, 0, [], time.time()
    cfg = dict(model=model, setting=setting, drop_p=p, fold=fold, seed=seed, variant=VARIANT, **CFG, bin_weights=bw.tolist(),
               n_train=len(dtr), n_inner=len(dva), n_params=sum(q.numel() for q in mod.parameters()),
               gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", started=datetime.now().isoformat(timespec="seconds"))
    json.dump(cfg, open(rd / "config.json", "w"), indent=1)
    for ep in range(CFG["max_epochs"]):
        te = time.time(); a = epoch(mod, ltr, dev, opt); b = epoch(mod, lva, dev); hist.append(dict(epoch=ep, train=a, val=b, sec=time.time() - te))
        if b < best:
            best, best_ep, bad = b, ep, 0; torch.save({"model_state_dict": mod.state_dict(), "epoch": ep}, rd / "best.pt")
        else:
            bad += 1
        log.write(f"epoch {ep} train {a:.4f} val {b:.4f} {hist[-1]['sec']:.1f}s\n"); log.flush()
        if bad >= CFG["patience"]:
            break
    pd.DataFrame(hist).to_csv(rd / "log.csv", index=False)
    mod.load_state_dict(torch.load(rd / "best.pt", map_location=dev)["model_state_dict"])
    predict(mod, V2Dataset(ho, D["X"], D["scaler"], 1, bw), dev).to_csv(rd / "predictions_heldout.csv", index=False)
    json.dump(dict(best_epoch=best_ep, epochs_run=len(hist), train_seconds=time.time() - t0, sec_per_epoch=float(np.mean([h["sec"] for h in hist])),
                   finished=datetime.now().isoformat(timespec="seconds")), open(rd / "done.json", "w"), indent=1)


def jobs(seeds=(SEED,)):
    return [(m, s, f, sd) for sd in seeds for s in SETTINGS for m in (MODELS if sd == 0 else ("lnn", "rnn", "lstm", "no")) for f in FOLDS]


def worker(gpu, seeds=(SEED,)):
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1")
    for attempt in (1, 2):
        for m, s, f, sd in jobs(seeds):
            rd = rdir(m, s, f, sd); rd.mkdir(parents=True, exist_ok=True); lk = rd / "claim.lock"
            if (rd / "done.json").exists():
                continue
            try:
                fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
            except FileExistsError:
                try:
                    os.kill(int(lk.read_text()), 0); continue
                except Exception:
                    lk.unlink(missing_ok=True); continue
            with open(rd / "stdout.log", "a") as o:
                subprocess.call([sys.executable, "-m", "glance.irregular", "train", "--model", m, "--setting", s, "--fold", str(f), "--seed", str(sd)],
                                 env=env, stdout=o, stderr=subprocess.STDOUT, cwd=str(ROOT))
            lk.unlink(missing_ok=True)
            done = sum((rdir(*j) / "done.json").exists() for j in jobs(seeds))
            (OUT / "STATUS.md").write_text(f"# irregular-visit experiment\n\nUpdated {datetime.now().isoformat(timespec='seconds')}: seeds {seeds}: {done} / {len(jobs(seeds))} runs done.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("train", "worker"))
    ap.add_argument("--model"); ap.add_argument("--setting"); ap.add_argument("--fold", type=int); ap.add_argument("--gpu", type=int)
    ap.add_argument("--seed", type=int, default=SEED); ap.add_argument("--seeds", default=str(SEED))
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.model, a.setting, a.fold, a.seed)
    else:
        worker(a.gpu, tuple(int(x) for x in a.seeds.split(",")))
