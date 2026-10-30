"""Continuous-time LNN variants (posthoc).
Variants:
  rs   CfC, training schedule redrawn per sample (rs_schedule); inner validation on the GLANCE schedule (0.5-y grid 0-10 y union visits).
  con  CfC, GLANCE schedule + a consistency loss L_CON = mean over the coarse points of (y_hat_fine - y_hat_coarse)^2, weight 1, in training and inner validation.
  ltc  ncps LTC on the GLANCE schedule.
Usage: python -m glance.continuous train --variant rs|con|ltc --fold k --seed s [--epochs E]
       python -m glance.continuous worker --gpu G   (claims the runs with lock files)
"""
import argparse, json, os, socket, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
from ncps.torch import LTC
from torch.utils.data import DataLoader
from .config import ROOT, RESULTS_POSTHOC
from .data import round_data, horizon_weights, V2Dataset, collate_w, load_cohort, load_samples, FOLDS, SEEDS
from .models import LNN, head, lnn_timespans, loss as v4_loss, query_points, set_seed, CFG, predict

OUT = RESULTS_POSTHOC / "continuous"; RUNS = OUT / "runs"
VARIANTS = ("rs", "con", "ltc")
T0_ELAPSED = 1.0
RS_STEPS = (1 / 12, 1 / 4, 1 / 2, 1.0)
W_CON = 1.0
FINAL_SEEDS, FINAL_EPOCHS = (0, 1, 2), 31


class LNNLTC(nn.Module):
    kind = "continuous"
    has_clean_output = True

    def __init__(self, in_dim=1025, hidden_dim=256):
        super().__init__()
        self.lnn = LTC(in_dim, hidden_dim, return_sequences=True)
        self.trajectory_projector = head(hidden_dim)

    def project_clean(self, h):
        pr = self.trajectory_projector
        return pr[3](pr[1](pr[0](h))).squeeze(-1)

    def sensory(self, x):
        c = self.lnn.rnn_cell; P = c._params; pos = c.make_positive_fn
        u = c._map_inputs(x)
        a = pos(P["sensory_w"]) * c._sigmoid(u, P["sensory_mu"], P["sensory_sigma"]) * P["sensory_sparsity_mask"]
        return torch.sum(a * P["sensory_erev"], dim=1), torch.sum(a, dim=1)

    def solve(self, sn, sd, v, elapsed):
        c = self.lnn.rnn_cell; P = c._params; pos = c.make_positive_fn
        cm_t = pos(P["cm"]) / (elapsed / c._ode_unfolds); w = pos(P["w"]); gl = pos(P["gleak"])
        for _ in range(c._ode_unfolds):
            wa = w * c._sigmoid(v, P["mu"], P["sigma"]) * P["sparsity_mask"]
            num = cm_t * v + gl * P["vleak"] + torch.sum(wa * P["erev"], dim=1) + sn
            den = cm_t + gl + torch.sum(wa, dim=1) + sd
            v = num / (den + c._epsilon)
        return v

    def forward(self, x, q, return_clean=False):
        ts = lnn_timespans(q)
        sn, sd = self.sensory(x)
        h = torch.zeros(x.size(0), self.lnn.state_size, device=x.device)
        outs = []
        for k in range(q.size(1)):
            dt = torch.full_like(ts[:, :1], T0_ELAPSED) if k == 0 else ts[:, k:k + 1]
            pos = dt > 0
            hn = self.solve(sn, sd, h, torch.where(pos, dt, torch.ones_like(dt)))
            h = torch.where(pos, hn, h)
            outs.append(self.lnn.rnn_cell._map_outputs(h))
        hs = torch.stack(outs, 1)
        p = self.trajectory_projector(hs).squeeze(-1)
        if return_clean:
            return p, self.project_clean(hs)
        return p


def build(variant, in_dim=1025):
    if variant == "ltc":
        return LNNLTC(in_dim)
    if variant in ("rs", "con"):
        return LNN(in_dim)
    raise ValueError(variant)


def pure_grid(step, tmax=10.0):
    n = int(round(tmax / step))
    return np.round(np.arange(n + 1) * step, 6)


def rs_schedule(tv, rng):
    s = RS_STEPS[rng.integers(len(RS_STEPS))]
    n = int(round(10.0 / s)); k = np.arange(1, n + 1)
    g = np.minimum(k * s + rng.uniform(-s / 2, s / 2, n), 10.0)
    return np.unique(np.r_[0.0, g, tv]).astype(np.float32), s


def fine_schedule(tv):
    return np.unique(np.r_[pure_grid(1 / 12), tv]).astype(np.float32)


def run_name(variant, fold, seed):
    return f"{variant}_f{fold}_s{seed}"


def rdir(variant, fold, seed):
    return RUNS / run_name(variant, fold, seed)


def run_sched(model, x, t, m, qlist):
    pv = [torch.searchsorted(u, t[i].contiguous()) for i, u in enumerate(qlist)]
    L = max(len(u) for u in qlist)
    q = torch.stack([torch.cat([u, u[-1:].expand(L - len(u))]) for u in qlist])
    qmask = torch.stack([torch.arange(L, device=x.device) < len(u) for u in qlist])
    p, pc = model(x, q, return_clean=True)
    pred_vis = torch.gather(p, 1, torch.stack(pv).clamp(max=L - 1))
    return pred_vis, pc, qmask, q, p


def glance_q(t, m):
    return [query_points(t[i][m[i]]) for i in range(t.size(0))]


def batch_loss(variant, model, x, t, y, m, w, so, train, rng):
    if variant == "rs" and train:
        ql = [torch.from_numpy(rs_schedule(t[i][m[i]].cpu().numpy(), rng)[0]).to(x.device) for i in range(x.size(0))]
    else:
        ql = glance_q(t, m)
    pv, pc, qm, q, _ = run_sched(model, x, t, m, ql)
    L, a, b, c = v4_loss("V4", pv, y, m, w, t, pc, qm, so)
    con = L.detach() * 0
    if variant == "con":
        qf = [torch.from_numpy(fine_schedule(t[i][m[i]].cpu().numpy())).to(x.device) for i in range(x.size(0))]
        _, pcf, qmf, _, _ = run_sched(model, x, t, m, qf)
        terms = []
        for i in range(x.size(0)):
            qc = ql[i]; idx = torch.searchsorted(qf[i], qc)
            assert torch.allclose(qf[i][idx], qc), "coarse point missing from the fine schedule"
            terms.append(((pcf[i, idx] - pc[i, :len(qc)]) ** 2).mean())
        con = torch.stack(terms).mean()
        L = L + W_CON * con
    return L, a, b, c, con.detach()


def epoch(variant, model, loader, device, opt=None, rng=None):
    train = opt is not None; model.train(train)
    acc = np.zeros(5); n = 0
    with torch.set_grad_enabled(train):
        for x, t, y, m, idx, w, so in loader:
            x, t, y, m, w, so = (a.to(device) for a in (x, t, y, m, w, so))
            L, a, b, c, d = batch_loss(variant, model, x, t, y, m, w, so, train, rng)
            if train:
                opt.zero_grad(); L.backward(); opt.step()
            k = x.size(0); acc += np.array([L.item(), a.item(), b.item(), c.item(), float(d)]) * k; n += k
    return acc / n


def train(variant, fold, seed, epochs=None, max_epochs=None, out=None):
    rd = Path(out) if out else rdir(variant, fold, seed); rd.mkdir(parents=True, exist_ok=True)
    if (rd / "done.json").exists():
        print("done already:", rd); return
    assert fold != 0 or epochs, "final runs need --epochs"
    set_seed(seed); rng = np.random.default_rng([2027, seed, fold])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    name = run_name(variant, fold, seed)
    cfg = dict(**CFG, variant=variant, loss="V4", w_con=W_CON if variant == "con" else None, fold=fold, seed=seed, name=name,
               fixed_epochs=epochs, gpu=gpu, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), host=socket.gethostname(),
               started=datetime.now().isoformat(timespec="seconds"))
    log = open(rd / "train.log", "w")

    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()

    C = load_cohort(); S = load_samples(); D = round_data(fold, C, S)
    bw, share = horizon_weights(D["all_train"])
    cfg.update(horizon_bin_weights=bw.tolist(), n_train_patients=len(D["train_pids"]), n_inner_patients=len(D["inner_pids"]),
               n_heldout_patients=len(D["heldout_pids"]))
    json.dump(cfg, open(rd / "config.json", "w"), indent=1)
    np.savez(rd / "scaler.npz", data_min=D["scaler"].min, scale=D["scaler"].scale)
    dtr = V2Dataset(D["train"], D["X"], D["scaler"], 1, bw)
    g = torch.Generator(); g.manual_seed(seed)
    ltr = DataLoader(dtr, batch_size=1, shuffle=True, collate_fn=collate_w(dtr), generator=g)
    if fold:
        dva = V2Dataset(D["inner"], D["X"], D["scaler"], 1, bw)
        lva = DataLoader(dva, batch_size=64, shuffle=False, collate_fn=collate_w(dva))
    model = build(variant, dtr.x.shape[1]).to(dev)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    say(f"run {name} on {gpu} | train {len(dtr)} samples | inner val {len(D['inner'])} | held-out {len(D['heldout_pids'])} | params {npar}")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    best, best_ep, bad, hist = float("inf"), -1, 0, []
    t0 = time.time(); n_ep = epochs if fold == 0 else (max_epochs or CFG["max_epochs"])
    for ep in range(n_ep):
        te = time.time()
        tr = epoch(variant, model, ltr, dev, opt, rng)
        ttr = time.time() - te
        va = epoch(variant, model, lva, dev) if fold else (np.nan,) * 5
        if not np.isfinite(tr[0]):
            say(f"non-finite training loss at epoch {ep}"); raise SystemExit(2)
        hist.append(dict(epoch=ep, train_total=tr[0], train_mse=tr[1], train_pen=tr[2], train_slope=tr[3], train_con=tr[4],
                         val_total=va[0], val_mse=va[1], val_pen=va[2], val_slope=va[3], val_con=va[4], sec=time.time() - te, sec_train=ttr))
        flag = ""
        if fold and va[0] < best:
            best, best_ep, bad = va[0], ep, 0; flag = " *best"
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "config": cfg}, rd / "best.pt")
        elif fold:
            bad += 1
        say(f"epoch {ep:2d} train {tr[0]:.4f} (mse {tr[1]:.4f} pen {tr[2]:.4f} slope {tr[3]:.4f} con {tr[4]:.4f}) | inner val {va[0]:.4f} "
            f"(mse {va[1]:.4f} con {va[4]:.4f}) | {hist[-1]['sec']:.1f}s{flag}")
        if fold and bad >= CFG["patience"]:
            say(f"early stop at epoch {ep}"); break
    if fold == 0:
        best_ep = n_ep - 1
        torch.save({"model_state_dict": model.state_dict(), "epoch": best_ep, "config": cfg}, rd / "best.pt")
    sec = time.time() - t0
    pd.DataFrame(hist).to_csv(rd / "log.csv", index=False)
    res = dict(name=name, best_epoch=best_ep, n_epochs_best=best_ep + 1, epochs_run=len(hist), best_inner_val=best, train_seconds=sec,
               sec_per_epoch=float(np.mean([h["sec"] for h in hist])), gpu=gpu, n_params=npar)
    if fold:
        model.load_state_dict(torch.load(rd / "best.pt", map_location=dev)["model_state_dict"])
        P = predict(model, V2Dataset(D["heldout_first"], D["X"], D["scaler"], 1, bw), dev)
        assert set(P.pid) == set(D["heldout_pids"])
        P.to_csv(rd / "predictions_heldout.csv", index=False)
    res["finished"] = datetime.now().isoformat(timespec="seconds")
    json.dump(res, open(rd / "done.json", "w"), indent=1)
    say(f"done: best epoch {best_ep} | {sec / 60:.1f} min")


def jobs():
    J = [(v, f, s, None) for v in VARIANTS for s in SEEDS for f in FOLDS]
    J += [("rs", 0, s, FINAL_EPOCHS) for s in FINAL_SEEDS]
    return J


def worker(gpu):
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1")
    for attempt in (1, 2):
        for v, f, s, e in jobs():
            rd = rdir(v, f, s); rd.mkdir(parents=True, exist_ok=True); lk = rd / "claim.lock"
            if (rd / "done.json").exists():
                continue
            try:
                fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
            except FileExistsError:
                try:
                    os.kill(int(lk.read_text()), 0); continue
                except Exception:
                    lk.unlink(missing_ok=True); continue
            cmd = [sys.executable, "-m", "glance.continuous", "train", "--variant", v, "--fold", str(f), "--seed", str(s)] + \
                (["--epochs", str(e)] if e else [])
            with open(rd / "stdout.log", "a") as o:
                subprocess.call(cmd, env=env, stdout=o, stderr=subprocess.STDOUT, cwd=str(ROOT))
            lk.unlink(missing_ok=True)
            done = sum((rdir(*j[:3]) / "done.json").exists() for j in jobs())
            (OUT / "STATUS.md").write_text(f"# continuous-time variants (rs / con / ltc)\n\nUpdated {datetime.now().isoformat(timespec='seconds')}: {done} / {len(jobs())} runs done.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("train", "worker"))
    ap.add_argument("--variant", choices=VARIANTS); ap.add_argument("--fold", type=int); ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--max_epochs", type=int, default=None, help="timing / smoke tests only")
    ap.add_argument("--out", default=None); ap.add_argument("--gpu", type=int)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.variant, a.fold, a.seed, a.epochs, a.max_epochs, a.out)
    else:
        worker(a.gpu)
