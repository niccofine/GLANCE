"""Samples, cohort, split and torch datasets for the forecasting pipeline."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .config import DATA, GRADES_XLSX, GRADES_SHEET

COHORT, SAMPLES, PAIRS = DATA / "cohort_v2.csv", DATA / "samples_v2.csv", DATA / "features_v2_pairs.npz"
SPLIT_TEST, SPLIT_CV = DATA / "split_v2_test.csv", DATA / "split_v2_cv5.csv"
SEEDS = (0, 1, 2)
FOLDS = (1, 2, 3, 4, 5)
INNER_FRAC, INNER_SEED = 0.15, 2027
DUP_PAIRS = (("G175", "G456"), ("G273", "G474"), ("G336", "G397"), ("G351", "G577"))   # old duplicate partners in cohort
BINS = ((0, 2), (2, 4), (4, 6), (6, 8), (8, np.inf))


class MinMaxNorm:
    def __init__(self, data_min, data_max):
        self.min = data_min.astype(np.float32)
        rng = (data_max - data_min).astype(np.float32)
        self.scale = np.where(rng > 0, rng, 1.0).astype(np.float32)

    @classmethod
    def fit(cls, X):
        return cls(X.min(0), X.max(0))

    def __call__(self, X):
        return (X - self.min) / self.scale


def collate(batch):
    x = torch.stack([b[0] for b in batch])
    L = max(len(b[1]) for b in batch)
    t = torch.zeros(len(batch), L); y = torch.zeros(len(batch), L); m = torch.zeros(len(batch), L, dtype=torch.bool)
    for k, b in enumerate(batch):
        n = len(b[1]); t[k, :n] = b[1]; y[k, :n] = b[2]; m[k, :n] = True
        if n < L:
            t[k, n:] = b[1][-1]
    idx = torch.tensor([b[3] for b in batch])
    return x, t, y, m, idx


def load_grades():
    d = pd.read_excel(GRADES_XLSX, sheet_name=GRADES_SHEET)
    d.columns = d.columns.astype(str).str.strip()
    cols = [c for c in d.columns if c.startswith("SCALE_")]
    out = {}
    for _, r in d.iterrows():
        v = [(int(c.split("_")[1]), float(r[c])) for c in cols if pd.notna(r[c])]
        out[str(r["ID2"])] = sorted(v)
    return out


def parse(s):
    return np.array([float(v) for v in str(s).split()], dtype=np.float32)


def load_cohort():
    return pd.read_csv(COHORT).set_index("patient_id")


def load_samples():
    S = pd.read_csv(SAMPLES)
    S["t"] = S.times_y.map(parse); S["y"] = S.grades.map(parse)
    return S


def load_X():
    return np.load(PAIRS)["X"].astype(np.float32)


def constr_folds():
    return pd.read_csv(SPLIT_CV).set_index("patient_id")["fold"].to_dict()


def test_pids():
    T = pd.read_csv(SPLIT_TEST)
    return sorted(T[T.part == "test"].patient_id)


def inner_split(pids, cohort, fold_key):
    pids = sorted(pids); st = cohort.loc[pids, "stratum"]
    N = int(round(INNER_FRAC * len(pids))); cnt = st.value_counts().sort_index()
    q = {s: int(np.floor(INNER_FRAC * n)) for s, n in cnt.items()}; rem = {s: INNER_FRAC * n - q[s] for s, n in cnt.items()}
    for s in sorted(rem, key=lambda s: (-rem[s], s))[:N - sum(q.values())]:
        q[s] += 1
    partner = {}
    for a, b in DUP_PAIRS:
        if a in pids and b in pids:
            partner[a] = b; partner[b] = a
    rng = np.random.default_rng(INNER_SEED + fold_key); used = {s: 0 for s in q}; val = set(); seen = set()
    for s in sorted(cnt.index):
        ids = sorted(st[st == s].index); rng.shuffle(ids)
        for p in ids:
            if p in seen:
                continue
            unit = [p] + ([partner[p]] if p in partner else [])
            need = pd.Series([st[u] for u in unit]).value_counts()
            fits = all(used[k] + v <= q[k] for k, v in need.items())
            for u in unit:
                seen.add(u)
                if fits:
                    val.add(u)
            if fits:
                for k, v in need.items():
                    used[k] += v
    return sorted(set(pids) - val), sorted(val)


def rounds():
    f = constr_folds(); allp = sorted(f)
    R = {k: ([p for p in allp if f[p] != k], [p for p in allp if f[p] == k]) for k in FOLDS}
    R[0] = (allp, [])
    return R


def horizon_weights(samples):
    t = np.concatenate([a[a > 0] for a in samples.t])
    share = np.array([((t > lo) & (t <= hi)).mean() for lo, hi in BINS])
    w = np.where(share > 0, 1.0 / np.sqrt(np.where(share > 0, share, 1)), 0.0)
    w = w / (share * w).sum()
    return w.astype(np.float32), share


def bin_index(t):
    t = np.asarray(t, float); out = np.full(t.shape, -1)
    for i, (lo, hi) in enumerate(BINS):
        out[(t > lo) & (t <= hi)] = i
    return out


def theil_sen(t, y):
    k = t > 0; tt, yy = t[k], y[k]
    if len(tt) < 2:
        return np.nan
    i, j = np.triu_indices(len(tt), 1)
    return float(np.median((yy[j] - yy[i]) / (tt[j] - tt[i])))


class V2Dataset(Dataset):
    def __init__(self, samples, X, scaler, use_grade, bin_w=None):
        self.s = samples.reset_index(drop=True)
        x = scaler(X[self.s.feature_row.values]).astype(np.float32)
        if use_grade:
            x = np.concatenate([x, (self.s.grade.values / 5.0).astype(np.float32)[:, None]], 1)
        self.x = torch.from_numpy(x)
        self.t = [torch.from_numpy(a) for a in self.s.t]
        self.y = [torch.from_numpy(a) for a in self.s.y]
        bw = np.ones(len(BINS), np.float32) if bin_w is None else bin_w
        self.w = [torch.from_numpy(np.where(bin_index(a) >= 0, bw[np.clip(bin_index(a), 0, None)], 1.0).astype(np.float32)) for a in self.s.t]
        self.slope = torch.tensor([theil_sen(a, b) for a, b in zip(self.s.t, self.s.y)], dtype=torch.float32)
        for a in self.s.t:
            assert np.allclose(a * 2, np.round(a * 2)) and a[0] == 0 and (np.diff(a) > 0).all()

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.x[i], self.t[i], self.y[i], i


def collate_w(ds):
    def f(batch):
        x, t, y, m, idx = collate(batch)
        w = torch.zeros_like(y)
        for k, i in enumerate(idx.tolist()):
            w[k, :len(ds.w[i])] = ds.w[i]
        return x, t, y, m, idx, w, ds.slope[idx]
    return f


def round_data(fold_key, cohort=None, S=None):
    cohort = load_cohort() if cohort is None else cohort; S = load_samples() if S is None else S
    trp, hop = rounds()[fold_key]
    assert not set(trp + hop) & set(pd.read_csv(SPLIT_TEST).query("part == 'test'").patient_id), "TEST patient in a round"
    if fold_key == 0:
        fit, inner = trp, []
    else:
        fit, inner = inner_split(trp, cohort, fold_key)
    X = load_X()
    scaler = MinMaxNorm.fit(X[S[S.pid.isin(trp)].feature_row.values])
    return dict(train=S[S.pid.isin(fit)], inner=S[S.pid.isin(inner)], heldout_first=S[S.pid.isin(hop) & S["first"]],
                all_train=S[S.pid.isin(trp)], scaler=scaler, X=X, train_pids=fit, inner_pids=inner, heldout_pids=hop)
