"""Forecasting models, loss variants, training loop and GPU job queue"""
import argparse, json, os, random, socket, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
from ncps.torch import CfC
from torch.utils.data import DataLoader
from .config import RESULTS, RESULTS_CV, RESULTS_LOSSSEL, RESULTS_FINAL, RNN_NATIVE_YEARS, GRID_STEP_Y, GRID_MAX_Y
from .data import round_data, horizon_weights, V2Dataset, collate_w, load_cohort, load_samples

STAGE_ROOTS = {"crossvalidation": RESULTS_CV, "lossselection": RESULTS_LOSSSEL, "finaltraining": RESULTS_FINAL}


def head(h):
    return nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))


def lnn_timespans(q):
    return torch.diff(q, dim=-1, prepend=torch.zeros_like(q[..., :1]))


class LNN(nn.Module):
    kind = "continuous"
    has_clean_output = True

    def __init__(self, in_dim=1024, hidden_dim=256):
        super().__init__()
        self.lnn = CfC(input_size=in_dim, units=hidden_dim, return_sequences=True)
        self.trajectory_projector = head(hidden_dim)

    def project_clean(self, h):
        pr = self.trajectory_projector
        return pr[3](pr[1](pr[0](h))).squeeze(-1)

    def forward(self, x, q, return_clean=False):
        ts = lnn_timespans(q)
        h = torch.zeros(x.size(0), self.lnn.state_size, device=x.device)
        outs = []
        for k in range(q.size(1)):
            h_out, h = self.lnn.rnn_cell(x, h, ts[:, k:k + 1])
            outs.append(self.lnn.fc(h_out))
        hs = torch.stack(outs, 1)
        p = self.trajectory_projector(hs).squeeze(-1)
        if return_clean:
            return p, self.project_clean(hs)
        return p

    def forward_ncps(self, x, q):
        seq = x.unsqueeze(1).expand(-1, q.size(1), -1)
        out, _ = self.lnn(seq, timespans=lnn_timespans(q))
        return self.trajectory_projector(out).squeeze(-1)


class NeuralOperator(nn.Module):
    """DeepONet: branch(x) . trunk(t)."""
    kind = "continuous"

    def __init__(self, in_dim=1024):
        super().__init__()
        self.branch_net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        self.trunk_net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128))

    def forward(self, x, q):
        b = self.branch_net(x)
        tr = self.trunk_net(q.unsqueeze(-1))
        return torch.einsum("bh,blh->bl", b, tr)


class RNNLSTM(nn.Module):
    kind = "native"
    has_clean_output = True

    def __init__(self, mode="LSTM", in_dim=1024, hidden=512):
        super().__init__()
        self.mode = mode
        self.input_projection = nn.Linear(in_dim, hidden)
        self.rnn_cell = nn.LSTMCell(1, hidden) if mode == "LSTM" else nn.RNNCell(1, hidden)
        self.trajectory_projector = head(hidden)
        self.native_times = torch.tensor(RNN_NATIVE_YEARS)
        self.max_year = int(max(RNN_NATIVE_YEARS))

    def project_clean(self, h):
        pr = self.trajectory_projector
        return pr[3](pr[1](pr[0](h)))

    def forward(self, x, return_clean=False):
        h = self.input_projection(x)
        c = torch.zeros_like(h) if self.mode == "LSTM" else None
        cur = torch.zeros(x.size(0), 1, device=x.device)
        keep, keep_clean = {}, {}
        for year in range(1, self.max_year + 1):
            if self.mode == "LSTM":
                h, c = self.rnn_cell(cur, (h, c))
            else:
                h = self.rnn_cell(cur, h)
            cur = self.trajectory_projector(h)
            if float(year) in RNN_NATIVE_YEARS:
                keep[year] = cur.squeeze(-1)
                if return_clean:
                    keep_clean[year] = self.project_clean(h).squeeze(-1)
        p = torch.stack([keep[int(y)] for y in RNN_NATIVE_YEARS], dim=1)
        if return_clean:
            return p, torch.stack([keep_clean[int(y)] for y in RNN_NATIVE_YEARS], dim=1)
        return p


STEP = 0.5


class RNNLSTMGrid(nn.Module):
    kind = "continuous"
    has_clean_output = True

    def __init__(self, mode="LSTM", in_dim=1024, hidden=512):
        super().__init__()
        self.mode = mode
        self.input_projection = nn.Linear(in_dim, hidden)
        self.rnn_cell = nn.LSTMCell(1, hidden) if mode == "LSTM" else nn.RNNCell(1, hidden)
        self.trajectory_projector = head(hidden)

    def project_clean(self, h):
        pr = self.trajectory_projector
        return pr[3](pr[1](pr[0](h)))

    def forward(self, x, q, return_clean=False):
        k = torch.round(q / STEP)
        assert torch.allclose(k * STEP, q, atol=1e-4), "query time not on the 0.5-y grid"
        n = int(k.max().item())
        h = self.input_projection(x)
        c = torch.zeros_like(h) if self.mode == "LSTM" else None
        cur = self.trajectory_projector(h)
        outs = [cur]; clean = [self.project_clean(h)] if return_clean else None
        for _ in range(n):
            if self.mode == "LSTM":
                h, c = self.rnn_cell(cur, (h, c))
            else:
                h = self.rnn_cell(cur, h)
            cur = self.trajectory_projector(h)
            outs.append(cur)
            if return_clean:
                clean.append(self.project_clean(h))
        P = torch.cat(outs, 1)
        idx = k.long()
        p = torch.gather(P, 1, idx)
        if return_clean:
            return p, torch.gather(torch.cat(clean, 1), 1, idx)
        return p


def build(name, in_dim):
    name = name.lower()
    if name == "lnn": return LNN(in_dim)
    if name == "no": return NeuralOperator(in_dim)
    if name == "rnn": return RNNLSTMGrid("RNN", in_dim)
    if name == "lstm": return RNNLSTMGrid("LSTM", in_dim)
    raise ValueError(name)


def check_grid(device=None):
    n = int(round(GRID_MAX_Y / GRID_STEP_Y)) + 1
    return torch.arange(n, device=device, dtype=torch.float32) * GRID_STEP_Y


def query_points(tv, device=None):
    return torch.unique(torch.cat([tv, check_grid(device if device is not None else tv.device)]), sorted=True)


def monotonic_penalty(pred_grid, grid_mask=None):
    drop = torch.relu(pred_grid[:, :-1] - pred_grid[:, 1:])
    if grid_mask is not None:
        drop = drop * (grid_mask[:, :-1] & grid_mask[:, 1:])
    return drop.sum(1)


LAM, MU = 1.0, 1.0
VARIANTS = {"V0": dict(lam=0.0, hw=False, slope=False), "V1": dict(lam=LAM, hw=False, slope=False),
            "V2": dict(lam=LAM, hw=True, slope=False), "V3": dict(lam=LAM, hw=False, slope=True),
            "V4": dict(lam=LAM, hw=True, slope=True)}


def run(model, x, t, m):
    qs, pv = [], []
    for i in range(x.size(0)):
        u = query_points(t[i][m[i]])
        qs.append(u); pv.append(torch.searchsorted(u, t[i].contiguous()))
    L = max(len(u) for u in qs)
    q = torch.stack([torch.cat([u, u[-1:].expand(L - len(u))]) for u in qs])
    qmask = torch.stack([torch.arange(L, device=x.device) < len(u) for u in qs])
    if getattr(model, "has_clean_output", False):
        p, pc = model(x, q, return_clean=True)
    else:
        p = pc = model(x, q)
    pred_vis = torch.gather(p, 1, torch.stack(pv).clamp(max=L - 1))
    return pred_vis, pc, qmask, q, p


def loss(variant, pred_vis, y, m, w, t, pred_clean, qmask, slope_obs):
    v = VARIANTS[variant]
    mf = m.float(); ww = w * mf if v["hw"] else mf
    se = (pred_vis - y) ** 2 * ww
    mse = (se.sum(1) / mf.sum(1)).mean()
    pen = monotonic_penalty(pred_clean, qmask).mean()
    total = mse + v["lam"] * pen if v["lam"] > 0 else mse
    sl = pred_vis.sum() * 0
    if v["slope"]:
        last = m.sum(1) - 1
        tl = torch.gather(t, 1, last[:, None]).squeeze(1)
        ps = (torch.gather(pred_vis, 1, last[:, None]).squeeze(1) - pred_vis[:, 0]) / tl
        ok = torch.isfinite(slope_obs)
        if ok.any():
            sl = ((ps[ok] - slope_obs[ok]) ** 2).mean()
            total = total + MU * sl
    return total, mse.detach(), pen.detach(), sl.detach()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


CFG = dict(lr=1e-4, weight_decay=0.01, batch_size=1, max_epochs=40, patience=8)


def run_name(model, grade, variant, fold, seed):
    return f"{model}_{'grade' if grade else 'img'}_{variant}_f{fold}_s{seed}"


def rdir(stage, j):
    return STAGE_ROOTS[stage] / "runs" / run_name(j["model"], j["grade"], j["variant"], j["fold"], j["seed"])


def epoch(model, loader, device, variant, opt=None):
    train = opt is not None; model.train(train)
    tot = mse = pen = sl = 0.0; n = 0
    with torch.set_grad_enabled(train):
        for x, t, y, m, idx, w, so in loader:
            x, t, y, m, w, so = (a.to(device) for a in (x, t, y, m, w, so))
            pv, pc, qm, q, _ = run(model, x, t, m)
            L, a, b, c = loss(variant, pv, y, m, w, t, pc, qm, so)
            if train:
                opt.zero_grad(); L.backward(); opt.step()
            k = x.size(0); tot += L.item() * k; mse += a.item() * k; pen += b.item() * k; sl += c.item() * k; n += k
    return tot / n, mse / n, pen / n, sl / n


@torch.no_grad()
def predict(model, ds, device, bs=64):
    model.eval(); rows = []
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_w(ds))
    for x, t, y, m, idx, w, so in loader:
        x, t, m = x.to(device), t.to(device), m.to(device)
        pv, pc, qm, q, p = run(model, x, t, m)
        for k, i in enumerate(idx.tolist()):
            r = ds.s.iloc[i]; tv = r.t; yv = dict(zip(np.round(tv, 4), r.y))
            qq = q[k][qm[k]].cpu().numpy(); pp = p[k][qm[k]].cpu().numpy()
            for tt, yy in zip(qq, pp):
                key = round(float(tt), 4)
                rows.append(dict(pid=r.pid, visno=int(r.visno), t=float(tt), y_pred=float(yy), is_visit=key in yv, y_true=yv.get(key, np.nan)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "worker", "list"))
    ap.add_argument("--stage", choices=tuple(STAGE_ROOTS), required=True)
    ap.add_argument("--model", choices=("lnn", "rnn", "lstm", "no"))
    ap.add_argument("--grade", type=int, default=1)
    ap.add_argument("--variant", choices=tuple(VARIANTS))
    ap.add_argument("--fold", type=int, help="1-5 CV round; 0 = final model on all CONSTR")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int, default=None, help="final runs: fixed number of epochs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", type=int, help="worker: which GPU to run on")
    a = ap.parse_args()
    if a.cmd == "worker":
        worker(a.stage, a.gpu); return
    if a.cmd == "list":
        for j in jobs(a.stage):
            print(run_name(j["model"], j["grade"], j["variant"], j["fold"], j["seed"]), (rdir(a.stage, j) / "done.json").exists())
        return
    name = run_name(a.model, a.grade, a.variant, a.fold, a.seed)
    rd = Path(a.out) if a.out else rdir(a.stage, dict(model=a.model, grade=a.grade, variant=a.variant, fold=a.fold, seed=a.seed))
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "done.json").exists():
        print("done already:", rd); return
    assert a.fold != 0 or a.epochs, "final runs need --epochs"
    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    cfg = dict(**CFG, **VARIANTS[a.variant], model=a.model, grade=a.grade, variant=a.variant, fold=a.fold, seed=a.seed, name=name,
               fixed_epochs=a.epochs, gpu=gpu, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), host=socket.gethostname(),
               started=datetime.now().isoformat(timespec="seconds"))

    C = load_cohort(); S = load_samples()
    D = round_data(a.fold, C, S)
    bw, share = horizon_weights(D["all_train"])
    cfg.update(horizon_bin_weights=bw.tolist(), horizon_bin_shares=share.tolist(), n_train_patients=len(D["train_pids"]),
               n_inner_patients=len(D["inner_pids"]), n_heldout_patients=len(D["heldout_pids"]))
    json.dump(cfg, open(rd / "config.json", "w"), indent=1)
    np.savez(rd / "scaler.npz", data_min=D["scaler"].min, scale=D["scaler"].scale)
    dtr = V2Dataset(D["train"], D["X"], D["scaler"], a.grade, bw)
    g = torch.Generator(); g.manual_seed(a.seed)
    ltr = DataLoader(dtr, batch_size=1, shuffle=True, collate_fn=collate_w(dtr), generator=g)
    if a.fold:
        dva = V2Dataset(D["inner"], D["X"], D["scaler"], a.grade, bw)
        lva = DataLoader(dva, batch_size=64, shuffle=False, collate_fn=collate_w(dva))
    model = build(a.model, dtr.x.shape[1]).to(dev)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"run {name} on {gpu} | train {len(dtr)} samples / {len(D['train_pids'])} patients | inner val "
          f"{len(D['inner'])} / {len(D['inner_pids'])} | held-out {len(D['heldout_pids'])} | params {npar} | bin weights {np.round(bw, 3).tolist()}")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    best, best_ep, bad, hist = float("inf"), -1, 0, []
    t0 = time.time(); n_ep = a.epochs if a.fold == 0 else CFG["max_epochs"]
    for ep in range(n_ep):
        te = time.time()
        tr = epoch(model, ltr, dev, a.variant, opt)
        va = epoch(model, lva, dev, a.variant) if a.fold else (np.nan,) * 4
        hist.append(dict(epoch=ep, train_total=tr[0], train_mse=tr[1], train_pen=tr[2], train_slope=tr[3],
                         val_total=va[0], val_mse=va[1], val_pen=va[2], val_slope=va[3], sec=time.time() - te))
        flag = ""
        if a.fold and va[0] < best:
            best, best_ep, bad = va[0], ep, 0; flag = " *best"
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "config": cfg}, rd / "best.pt")
        elif a.fold:
            bad += 1
        print(f"epoch {ep:2d} train {tr[0]:.4f} (mse {tr[1]:.4f} pen {tr[2]:.4f} slope {tr[3]:.4f}) | inner val {va[0]:.4f} "
              f"(mse {va[1]:.4f} pen {va[2]:.4f} slope {va[3]:.4f}) | {hist[-1]['sec']:.1f}s{flag}")
        if a.fold and bad >= CFG["patience"]:
            print(f"early stop at epoch {ep}"); break
    if a.fold == 0:
        best_ep = n_ep - 1
        torch.save({"model_state_dict": model.state_dict(), "epoch": best_ep, "config": cfg}, rd / "best.pt")
    sec = time.time() - t0
    pd.DataFrame(hist).to_csv(rd / "log.csv", index=False)
    res = dict(name=name, best_epoch=best_ep, n_epochs_best=best_ep + 1, epochs_run=len(hist), best_inner_val=best, train_seconds=sec,
               sec_per_epoch=float(np.mean([h["sec"] for h in hist])), gpu=gpu, n_params=npar)
    if a.fold:
        model.load_state_dict(torch.load(rd / "best.pt", map_location=dev)["model_state_dict"])
        P = predict(model, V2Dataset(D["heldout_first"], D["X"], D["scaler"], a.grade, bw), dev)
        assert set(P.pid) == set(D["heldout_pids"])
        P.to_csv(rd / "predictions_heldout.csv", index=False)
    res["finished"] = datetime.now().isoformat(timespec="seconds")
    json.dump(res, open(rd / "done.json", "w"), indent=1)
    print(f"done: best epoch {best_ep} | {sec / 60:.1f} min")


CV_FOLDS, CV_SEEDS = (1, 2, 3, 4, 5), (0, 1, 2)
ONE_SEED_FLAG = RESULTS_CV / "stage1_v0_one_seed.flag"


def seeds_for(model, variant, fold=1):
    if fold and variant == "V0" and model in ("rnn", "lstm", "no") and ONE_SEED_FLAG.exists():
        return (0,)
    return CV_SEEDS


def cv_jobs(model, grade, variant):
    return [dict(model=model, grade=grade, variant=variant, fold=f, seed=s) for s in seeds_for(model, variant) for f in CV_FOLDS]


def jobs(stage):
    if stage == "crossvalidation":
        J = []
        for v in ("V1", "V2", "V3", "V4", "V0"):
            J += cv_jobs("lnn", 1, v)
        J += cv_jobs("lnn", 0, "V1")
        for m in ("lstm", "rnn", "no"):
            for v in ("V1", "V0"):
                J += cv_jobs(m, 1, v)
        return J
    if stage == "lossselection":
        vs = json.load(open(RESULTS_LOSSSEL / "selection.json"))["V_star"]
        return [] if vs == "V1" else sum((cv_jobs(m, 1, vs) for m in ("lstm", "rnn", "no")), [])
    if stage == "finaltraining":
        E = json.load(open(RESULTS_FINAL / "final_epochs.json")); J = []
        for key, e in E.items():
            model, grade, variant = key.split("|")
            J += [dict(model=model, grade=int(grade), variant=variant, fold=0, seed=s, epochs=int(e)) for s in CV_SEEDS]
        return J
    raise ValueError(stage)


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def claim(stage, name):
    d = STAGE_ROOTS[stage] / "queue"; d.mkdir(parents=True, exist_ok=True); f = d / f"{name}.lock"
    try:
        fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return f
    except FileExistsError:
        try:
            pid = int(f.read_text().strip() or 0)
        except Exception:
            pid = 0
        if pid and alive(pid):
            return None
        f.unlink(missing_ok=True)
        return claim(stage, name)


def worker(stage, gpu):
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1")
    for attempt in (1, 2):
        for j in jobs(stage):
            name = run_name(j["model"], j["grade"], j["variant"], j["fold"], j["seed"]); d = rdir(stage, j)
            if (d / "done.json").exists() or (attempt == 1 and (d / "FAILED").exists()):
                continue
            lk = claim(stage, name)
            if lk is None:
                continue
            if (d / "done.json").exists():
                lk.unlink(missing_ok=True); continue
            d.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "glance.models", "train", "--stage", stage, "--model", j["model"], "--grade", str(j["grade"]),
                   "--variant", j["variant"], "--fold", str(j["fold"]), "--seed", str(j["seed"])] + (["--epochs", str(j["epochs"])] if j.get("epochs") else [])
            rc = subprocess.call(cmd, env=env, cwd=str(RESULTS.parent))
            if rc != 0:
                (d / "FAILED").write_text(f"exit {rc} at {datetime.now().isoformat(timespec='seconds')} (attempt {attempt})\n")
            else:
                (d / "FAILED").unlink(missing_ok=True)
            lk.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
