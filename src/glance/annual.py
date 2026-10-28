"""Annual-grid RNN / LSTM + grade posthoc"""
import argparse, json, os, subprocess, sys, time
from datetime import datetime
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from .config import ROOT, RESULTS_POSTHOC
from .data import round_data, horizon_weights, V2Dataset, collate_w, load_cohort, load_samples, FOLDS, SEEDS
from .models import RNNLSTMGrid, loss, set_seed, CFG

OUT = RESULTS_POSTHOC / "annual"; RUNS = OUT / "runs"; VARIANT = "V4"; STEP = 1.0; GRID_MAX = 10


def annualise(S):
    S = S.copy(); T, Y = [], []
    for t, y in zip(S.t, S.y):
        k = np.isclose(t, np.round(t)); T.append(t[k]); Y.append(y[k])
    S["t"], S["y"] = T, Y
    return S


class AnnualRNN(RNNLSTMGrid):

    def forward(self, x, q, return_clean=False):
        k = torch.round(q / STEP)
        assert torch.allclose(k * STEP, q, atol=1e-4), "query time not on the annual grid"
        n = int(k.max().item())
        h = self.input_projection(x); c = torch.zeros_like(h) if self.mode == "LSTM" else None
        cur = self.trajectory_projector(h); outs = [cur]; clean = [self.project_clean(h)] if return_clean else None
        for _ in range(n):
            if self.mode == "LSTM":
                h, c = self.rnn_cell(cur, (h, c))
            else:
                h = self.rnn_cell(cur, h)
            cur = self.trajectory_projector(h); outs.append(cur)
            if return_clean:
                clean.append(self.project_clean(h))
        idx = k.long(); p = torch.gather(torch.cat(outs, 1), 1, idx)
        return (p, torch.gather(torch.cat(clean, 1), 1, idx)) if return_clean else p


def annual_grid(dev):
    return torch.arange(GRID_MAX + 1, device=dev, dtype=torch.float32)


def run(model, x, t, m):
    qs, pv = [], []
    for i in range(x.size(0)):
        u = torch.unique(torch.cat([t[i][m[i]], annual_grid(x.device)]), sorted=True)
        qs.append(u); pv.append(torch.searchsorted(u, t[i].contiguous()))
    L = max(len(u) for u in qs)
    q = torch.stack([torch.cat([u, u[-1:].expand(L - len(u))]) for u in qs])
    qmask = torch.stack([torch.arange(L, device=x.device) < len(u) for u in qs])
    p, pc = model(x, q, return_clean=True)
    return torch.gather(p, 1, torch.stack(pv).clamp(max=L - 1)), pc, qmask, q, p


def epoch(model, loader, dev, opt=None):
    tr = opt is not None; model.train(tr); tot = n = 0
    with torch.set_grad_enabled(tr):
        for x, t, y, m, idx, w, so in loader:
            x, t, y, m, w, so = (a.to(dev) for a in (x, t, y, m, w, so))
            pv, pc, qm, q, _ = run(model, x, t, m)
            L, *_ = loss(VARIANT, pv, y, m, w, t, pc, qm, so)
            if tr:
                opt.zero_grad(); L.backward(); opt.step()
            tot += L.item() * x.size(0); n += x.size(0)
    return tot / n


def rdir(model, fold, seed):
    return RUNS / f"{model}_f{fold}_s{seed}"


@torch.no_grad()
def predict(model, ds, full, dev):
    model.eval(); rows = []; last = dict(zip(full.pid, [float(a.max()) for a in full.t]))
    for i in range(len(ds)):
        x, _, _, _ = ds[i]; pid = ds.s.pid.iloc[i]
        q = torch.arange(max(GRID_MAX, int(np.ceil(last[pid]))) + 1, dtype=torch.float32, device=dev)[None]
        p = model(x[None].to(dev), q)[0].cpu().numpy()
        rows += [dict(pid=pid, t=float(tt), y_pred=float(pp)) for tt, pp in zip(q[0].cpu().numpy(), p)]
    return pd.DataFrame(rows)


def train(model, fold, seed):
    rd = rdir(model, fold, seed); rd.mkdir(parents=True, exist_ok=True)
    if (rd / "done.json").exists():
        return
    set_seed(seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = load_cohort(); S = load_samples(); D = round_data(fold, C, S)
    tr, inner, allt = (annualise(D[k]) for k in ("train", "inner", "all_train"))
    n_vis = lambda X: int(sum((a > 0).sum() for a in X.t))
    excl = dict(train_future_visits_all=n_vis(D["train"]), train_future_visits_integer=n_vis(tr),
                inner_future_visits_all=n_vis(D["inner"]), inner_future_visits_integer=n_vis(inner))
    bw, share = horizon_weights(allt)
    dtr, dva = V2Dataset(tr, D["X"], D["scaler"], 1, bw), V2Dataset(inner, D["X"], D["scaler"], 1, bw)
    g = torch.Generator(); g.manual_seed(seed)
    ltr = DataLoader(dtr, batch_size=1, shuffle=True, collate_fn=collate_w(dtr), generator=g)
    lva = DataLoader(dva, batch_size=64, shuffle=False, collate_fn=collate_w(dva))
    mod = AnnualRNN({"rnn": "RNN", "lstm": "LSTM"}[model], dtr.x.shape[1]).to(dev)
    opt = torch.optim.AdamW(mod.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    log = open(rd / "train.log", "w"); best, best_ep, bad, hist, t0 = np.inf, -1, 0, [], time.time()
    json.dump(dict(model=model, grid="annual", fold=fold, seed=seed, variant=VARIANT, **CFG, bin_weights=bw.tolist(), **excl,
                   n_train=len(dtr), n_inner=len(dva), n_params=sum(q.numel() for q in mod.parameters()),
                   gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", started=datetime.now().isoformat(timespec="seconds")),
              open(rd / "config.json", "w"), indent=1)
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
    ho = D["heldout_first"]
    predict(mod, V2Dataset(annualise(ho), D["X"], D["scaler"], 1, bw), ho, dev).to_csv(rd / "predictions_heldout.csv", index=False)
    json.dump(dict(best_epoch=best_ep, n_epochs_best=best_ep + 1, epochs_run=len(hist), train_seconds=time.time() - t0,
                   sec_per_epoch=float(np.mean([h["sec"] for h in hist])), finished=datetime.now().isoformat(timespec="seconds")),
              open(rd / "done.json", "w"), indent=1)


def jobs():
    from .irregular import rdir as irdir
    J = [("tlstm", f, s, irdir("tlstm", "S50", f, s)) for s in (1, 2) for f in FOLDS]
    J += [(m, f, s, rdir(m, f, s)) for s in SEEDS for m in ("lstm", "rnn") for f in FOLDS]
    return J


def worker(gpu):
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1")
    qd = OUT / "queue"; qd.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        for m, f, s, rd in jobs():
            if (rd / "done.json").exists():
                continue
            lk = qd / f"{m}_f{f}_s{s}.lock"
            try:
                fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
            except FileExistsError:
                try:
                    os.kill(int(lk.read_text()), 0); continue
                except Exception:
                    lk.unlink(missing_ok=True); continue
            rd.mkdir(parents=True, exist_ok=True)
            cmd = ([sys.executable, "-m", "glance.irregular", "train", "--model", "tlstm", "--setting", "S50", "--fold", str(f), "--seed", str(s)] if m == "tlstm"
                   else [sys.executable, "-m", "glance.annual", "train", "--model", m, "--fold", str(f), "--seed", str(s)])
            with open(rd / "stdout.log", "a") as o:
                rc = subprocess.call(cmd, env=env, stdout=o, stderr=subprocess.STDOUT, cwd=str(ROOT))
            if rc != 0:
                (rd / "FAILED").write_text(f"exit {rc} at {datetime.now().isoformat(timespec='seconds')} (attempt {attempt})\n")
            lk.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--model"); ap.add_argument("--fold", type=int); ap.add_argument("--seed", type=int); ap.add_argument("--gpu", type=int)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.model, a.fold, a.seed)
    else:
        worker(a.gpu)
