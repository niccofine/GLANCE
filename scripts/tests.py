"""Unit tests for the pipeline's own code (CPU, no training run saved).
Run: python -m scripts.tests
"""
import sys
import numpy as np, pandas as pd, torch
from glance.data import load_cohort, load_samples, rounds, inner_split, horizon_weights, theil_sen, V2Dataset, collate_w, round_data, DUP_PAIRS, SPLIT_TEST, bin_index
from glance.models import RNNLSTMGrid, RNNLSTM, build, run, loss

torch.manual_seed(0)

ok = lambda c, msg: print(("PASS " if c else "FAIL ") + msg) or (c or sys.exit(1))

# 1. Theil-Sen
ok(np.isnan(theil_sen(np.array([0, 1.]), np.array([2, 3.]))), "1 future visit -> no slope")
ok(abs(theil_sen(np.array([0, 1, 3.]), np.array([0, 2, 3.])) - 0.5) < 1e-9, "2 future visits -> simple difference (3-2)/(3-1)")
ok(abs(theil_sen(np.array([0, 1, 2, 4.]), np.array([9, 1, 3, 1.])) - 0.0) < 1e-9, "Theil-Sen median of pairwise slopes (2, 0, -1) -> 0")

# 2. rounds, TEST, inner split
C = load_cohort(); S = load_samples(); R = rounds()
test = set(pd.read_csv(SPLIT_TEST).query("part == 'test'").patient_id)
ok(len(test) == 76 and all(not (set(a) | set(b)) & test for a, b in R.values()), "no TEST patient in any round (76 locked)")
ok(sorted(sum((R[k][1] for k in range(1, 6)), [])) == sorted(R[0][0]) and len(R[0][0]) == 302, "folds partition the 302 CONSTR patients")
for k in range(1, 6):
    tr, va = inner_split(R[k][0], C, k)
    ok(abs(len(va) - round(0.15 * len(R[k][0]))) <= 1 and not set(tr) & set(va) and sorted(tr + va) == sorted(R[k][0]),
       f"fold {k}: inner val {len(va)} of {len(R[k][0])} training patients")
    for a, b in DUP_PAIRS:
        if a in R[k][0] and b in R[k][0]:
            ok((a in va) == (b in va), f"fold {k}: duplicate partners {a}/{b} on the same side")
    ok(inner_split(R[k][0], C, k) == (tr, va), f"fold {k}: inner split deterministic")

# 3. horizon weights
D = round_data(1, C, S)
bw, share = horizon_weights(D["all_train"])
t = np.concatenate([a[a > 0] for a in D["all_train"].t]); wt = bw[bin_index(t)]
ok(abs(wt.mean() - 1) < 1e-5 and abs(share.sum() - 1) < 1e-9, f"horizon weights mean 1 over future targets: {np.round(bw, 3)} shares {np.round(share, 3)}")
ds = V2Dataset(D["train"], D["X"], D["scaler"], 1, bw)
ok(all(float(w[0]) == 1.0 for w in ds.w), "t = 0 target weight 1")
ok(ds.x.shape[1] == 1025 and float(ds.x[:, -1].max()) <= 1.0, "grade input = grade / 5 as last column")
ok(abs(float(ds.x[0, -1]) * 5 - D["train"].grade.iloc[0]) < 1e-6, "grade input = grade at the sample's own starting visit")

# 4. RNN / LSTM grid model
for mode in ("RNN", "LSTM"):
    m = RNNLSTMGrid(mode, 1025).eval()
    x = torch.rand(2, 1025); q = torch.stack([torch.arange(0, 10.01, 0.5), torch.arange(0, 10.01, 0.5)])
    full = m(x, q)
    q2 = torch.tensor([[0, 1.0, 3.5, 7.0], [0.5, 2.0, 2.5, 10.0]])
    ok(torch.allclose(m(x, q2), torch.gather(full, 1, torch.round(q2 / 0.5).long())), f"{mode}: outputs at arbitrary 0.5-y query points = full grid outputs")
    p, pc = m(x, q, return_clean=True); ok(torch.allclose(p, pc), f"{mode}: eval mode clean == normal")
    m.train(); p, pc = m(x, q, return_clean=True); ok(not torch.allclose(p, pc), f"{mode}: train mode dropout acts on the MSE path only")
    try:
        m(x, torch.tensor([[0, 0.25], [0, 0.5]])); ok(False, "off-grid time must fail")
    except AssertionError:
        ok(True, f"{mode}: off-grid query time rejected")
    old = RNNLSTM(mode, 1025)
    ok(sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in m.parameters()), f"{mode}: same parameter count as v1 ({sum(p.numel() for p in m.parameters())})")

# 5. loss variants on a real batch
b = collate_w(ds)([ds[i] for i in range(4)])
x, t, y, m, idx, w, so = b
for name in ("lnn", "no", "rnn", "lstm"):
    mod = build(name, 1025).train()
    pv, pc, qm, q, p = run(mod, x, t, m)
    L0 = loss("V0", pv, y, m, w, t, pc, qm, so); L1 = loss("V1", pv, y, m, w, t, pc, qm, so)
    L2 = loss("V2", pv, y, m, w, t, pc, qm, so); L3 = loss("V3", pv, y, m, w, t, pc, qm, so); L4 = loss("V4", pv, y, m, w, t, pc, qm, so)
    ok(abs(L1[0].item() - (L0[0].item() + L1[2].item())) < 1e-4, f"{name}: V1 = MSE + penalty")
    ok(abs(L3[0].item() - (L1[0].item() + L3[3].item())) < 1e-4, f"{name}: V3 = V1 + slope")
    ok(abs(L4[0].item() - (L2[0].item() + L4[3].item())) < 1e-4, f"{name}: V4 = V2 + slope")
    ok(int(q[0][qm[0]].numel()) >= 21 and float(q[0][qm[0]].max()) >= 10, f"{name}: queried at the 0.5-y grid 0-10 y union visits")
    L4[0].backward()
i = 0; n = int(m[i].sum()); tt = t[i, :n]; mse_w = float((w[i, :n] * (pv[i, :n] - y[i, :n]) ** 2).sum() / n)
ok(abs(float(loss("V2", pv[:1], y[:1], m[:1], w[:1], t[:1], pc[:1], qm[:1], so[:1])[1]) - mse_w) < 1e-5, "V2 weighted MSE = sum w.se / n")
print("all tests passed")
