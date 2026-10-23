"""Explainability: after test is opened, on test set, final LNN + grade V* (3 seeds).
 A  feature attribution;
 B  concept projection, per-eye RET-CLIP heads;
 C  pixel IG through the frozen RET-CLIP encoder.
--dry: code test only on a cross-validation model (lnn_grade_V1 fold 1 seed 0) and its held-out construction patients, writing results/explainability_dry/.

Can be run on its own or through scripts/driver.sh:
  python -m scripts.explainability
  python -m scripts.explainability --dry
"""
import json, sys, traceback
import numpy as np, pandas as pd, torch
import torch.nn.functional as Fnn
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import roc_auc_score
from glance.config import DATA, EDO_DIR, RETCLIP_CODE, RETCLIP_CKPT, BOOT_SEED, N_BOOT, LATE_THRESHOLD, RESULTS, RESULTS_LOSSSEL, RESULTS_FINAL, RESULTS_EXPLAIN
from glance.data import MinMaxNorm, collate, load_samples, load_X, test_pids, rounds, V2Dataset, SEEDS
from glance.encoder import load_visual_encoder, clip_preprocess, CLIP_MEAN, CLIP_STD, _remap_state_dict
from glance.evaluate import patients
from glance.models import build, query_points, run_name

DRY = "--dry" in sys.argv
OUT = RESULTS / "explainability_dry" if DRY else RESULTS_EXPLAIN
OUT.mkdir(parents=True, exist_ok=True)
H = 5.0; IG_STEPS = 50; PIX_STEPS, CH = 32, 4
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SHORT = {"normal fundus": "normal", "drusen": "drusen", "large drusen": "large drusen",
         "pigmentary abnormality": "pigment", "geographic atrophy": "GA",
         "choroidal neovascularization": "CNV", "macular degeneration": "AMD"}


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def boot_mean_ci(v, n=N_BOOT, seed=BOOT_SEED):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


CONCEPTS = [("normal fundus", "正常眼底"), ("drusen", "玻璃膜疣"), ("large drusen", "大玻璃膜疣"),
            ("pigmentary abnormality", "色素异常"), ("geographic atrophy", "地图样萎缩"),
            ("choroidal neovascularization", "脉络膜新生血管"), ("macular degeneration", "黄斑变性")]
LATE_CONCEPTS = ("geographic atrophy", "choroidal neovascularization")


def load_full_clip(device="cpu"):
    """Whole RET-CLIP-PLUS model (visual + text towers), eval mode. Returns (model, missing text keys)."""
    if str(RETCLIP_CODE) not in sys.path:
        sys.path.insert(0, str(RETCLIP_CODE))
    from RET_CLIP_PLUS.clip.model import CLIP
    cfg = RETCLIP_CODE / "RET_CLIP_PLUS" / "clip" / "model_configs"
    info = json.load(open(cfg / "ViT-B-16.json", encoding="utf-8"))
    if isinstance(info["vision_layers"], str):
        info["vision_layers"] = eval(info["vision_layers"])
    info.update(json.load(open(cfg / "RoBERTa-wwm-ext-base-chinese.json", encoding="utf-8")))
    model = CLIP(**info).to("cpu")
    ckpt = torch.load(RETCLIP_CKPT, map_location="cpu")
    sd = _remap_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading RET-CLIP checkpoint: {unexpected}")
    miss_text = [k for k in missing if k.startswith(("bert.", "text_projection")) and "pooler" not in k]
    return model.to(device).eval(), miss_text


def _preprocess_text(text):
    return text.lower().replace("“", '"').replace("”", '"')


@torch.no_grad()
def build_text_features(model, text_classes, context_length=200, device="cpu", head=0):
    """head 0 = global text projection (the zero-shot head), 1 / 2 = left / right text projections
    (diagnostic only). Returns [512, n_classes]."""
    from RET_CLIP_PLUS.clip import tokenize
    weights = []
    for cname_list in text_classes:
        texts = tokenize([_preprocess_text(n) for n in cname_list], context_length=context_length).to(device)
        class_emb = model.encode_text(text=texts)[head]
        class_emb = class_emb / class_emb.norm(dim=-1, keepdim=True)
        class_emb = class_emb.mean(dim=0)
        weights.append(class_emb / class_emb.norm())
    return torch.stack(weights, dim=1)


def zero_shot_cos(feats, textfeat):
    return Fnn.normalize(torch.as_tensor(feats, dtype=torch.float32), dim=-1) @ textfeat

def setup():
    PAT = patients(); S = load_samples(); X = load_X()
    if DRY:
        vs = "V1"; dirs = [RESULTS / "crossvalidation" / "runs" / run_name("lnn", 1, "V1", 1, 0)]; pids = rounds()[1][1]
    else:
        vs = json.load(open(RESULTS_LOSSSEL / "selection.json"))["V_star"]
        dirs = [RESULTS_FINAL / "runs" / run_name("lnn", 1, vs, 0, s) for s in SEEDS]; pids = test_pids()
    F = S[S.pid.isin(pids) & S["first"]].sort_values("pid").reset_index(drop=True)
    models, scalers = [], []
    for d in dirs:
        z = np.load(d / "scaler.npz"); sc = MinMaxNorm(np.zeros(1), np.ones(1)); sc.min, sc.scale = z["data_min"], z["scale"]
        m = build("lnn", 1025).to(dev); m.load_state_dict(torch.load(d / "best.pt", map_location=dev)["model_state_dict"]); m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m); scalers.append(sc)
    for sc in scalers[1:]:
        assert np.allclose(sc.min, scalers[0].min) and np.allclose(sc.scale, scalers[0].scale)
    ds = V2Dataset(F, X, scalers[0], 1)
    grp = np.array([PAT[p]["traj"] for p in F.pid]); g0 = F.grade.values
    return dict(PAT=PAT, S=S, X=X, F=F, ds=ds, models=models, scaler=scalers[0], grp=grp, g0=g0, vs=vs, dirs=[str(d) for d in dirs])


def qidx(t):
    q = query_points(t).to(dev); i0 = int(torch.nonzero(q == 0)[0]); i5 = int(torch.nonzero(torch.isclose(q, torch.tensor(H, device=dev)))[0])
    return q, i0, i5


def ig_all(C):
    IG, GR, ERR = [], [], []
    al = (torch.arange(IG_STEPS, dtype=torch.float32, device=dev) + 0.5) / IG_STEPS
    for i in range(len(C["ds"])):
        x, t, y, m, _ = collate([C["ds"][i]]); x = x.to(dev); q, i0, i5 = qidx(t[0][m[0]])
        per, gro, err = [], [], []
        for mod in C["models"]:
            f = lambda z: (lambda o: o[:, i5] - o[:, i0])(mod(z, q[None].expand(z.size(0), -1)))
            xs = (al[:, None] * x).requires_grad_(True)
            out = f(xs); g, = torch.autograd.grad(out.sum(), xs)
            ig = (x * g.mean(0, keepdim=True))[0]
            with torch.no_grad():
                fx = float(f(x)); fb = float(f(torch.zeros_like(x)))
            per.append(ig.detach().cpu().numpy()); gro.append(fx); err.append(abs(float(ig.sum()) - (fx - fb)))
        IG.append(np.mean(per, 0)); GR.append(np.mean(gro)); ERR.append(max(err))
    return np.array(IG), np.array(GR), np.array(ERR)


def part_A(C, IG, GR, ERR):
    A = np.abs(IG); tot = A.sum(1)
    sh = pd.DataFrame(dict(pid=C["F"].pid, group=C["grp"], first_grade=C["g0"], growth=GR, completeness_err=ERR,
                           share_left=A[:, :512].sum(1) / tot, share_right=A[:, 512:1024].sum(1) / tot, share_grade=A[:, 1024] / tot))
    sh.to_csv(OUT / "A_shares_per_patient.csv", index=False)
    rows = []
    for eye, o in (("left", 0), ("right", 512)):
        msign = IG[:, o:o + 512].mean(0)
        for kind, idx in (("risk", np.argsort(-msign)[:15]), ("protective", np.argsort(msign)[:15])):
            for r, j in enumerate(idx):
                rows.append(dict(eye=eye, kind=kind, rank=r + 1, feature=int(o + j), mean_signed_ig=float(msign[j]), mean_abs_ig=float(A[:, o + j].mean())))
    pd.DataFrame(rows).to_csv(OUT / "A_top_features.csv", index=False)
    summ = dict(n=len(sh), share_left=float(sh.share_left.mean()), share_right=float(sh.share_right.mean()), share_grade=float(sh.share_grade.mean()),
                completeness_err_max=float(ERR.max()),
                by_group={g: dict(n=int((sh.group == g).sum()), share_grade=float(sh[sh.group == g].share_grade.mean()), growth=float(sh[sh.group == g].growth.mean()))
                          for g in ("Only up", "Stable", "Up and down", "Only down") if (sh.group == g).any()})
    return summ


def part_B(C, IG):
    model, miss = load_full_clip("cpu")
    assert not miss, miss[:5]
    prompts = [[zh] for _, zh in CONCEPTS]; names = [en for en, _ in CONCEPTS]
    Tl = build_text_features(model, prompts, head=1); Tr = build_text_features(model, prompts, head=2)
    ML, MR = model.left_feature_mapping, model.right_feature_mapping
    PR = pd.read_csv(DATA / "features_v2_pairs.csv"); PR = PR[PR.graded & PR.pid.isin(rounds()[0][0])].reset_index(drop=True)
    XV = torch.from_numpy(C["X"][PR.row.values])
    with torch.no_grad():
        sc = np.maximum(zero_shot_cos(ML(XV[:, :512]), Tl).numpy(), zero_shot_cos(MR(XV[:, 512:]), Tr).numpy())
    late = PR.grade.values >= LATE_THRESHOLD; rng = np.random.default_rng(BOOT_SEED); up = PR.pid.unique()
    groups = {p: np.where(PR.pid.values == p)[0] for p in up}; boots = [np.concatenate([groups[p] for p in rng.choice(up, len(up))]) for _ in range(N_BOOT)]
    b1 = []
    for c, nm in enumerate(names):
        auc = roc_auc_score(late, sc[:, c]); bs = [roc_auc_score(late[b], sc[b, c]) for b in boots]
        b1.append(dict(concept=nm, n_visits=len(PR), n_patients=len(up), auc_late=auc, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    B1 = pd.DataFrame(b1); B1.to_csv(OUT / "B1_sanity_constr.csv", index=False)
    passed = bool((B1[B1.concept.isin(LATE_CONCEPTS)].lo > 0.5).any())
    if not passed:
        return dict(sanity_passed=False), B1, None
    WL, WR = ML.weight.detach().numpy().astype(np.float64), MR.weight.detach().numpy().astype(np.float64)
    aL, aR = IG[:, :512] @ WL.T, IG[:, 512:1024] @ WR.T
    CA = ((aL / np.linalg.norm(aL, axis=1, keepdims=True)) @ Tl.numpy() + (aR / np.linalg.norm(aR, axis=1, keepdims=True)) @ Tr.numpy()) / 2
    pr, st = C["grp"] == "Only up", C["grp"] == "Stable"; rows = []
    for c, nm in enumerate(names):
        v = CA[:, c]; r = dict(concept=nm, mean_all=float(v.mean()), n_all=len(v))
        for g, msk in (("progressors", pr), ("stable", st)):
            m, lo, hi = boot_mean_ci(v[msk]) if msk.sum() > 1 else (np.nan, np.nan, np.nan); r.update({f"{g}_mean": m, f"{g}_lo": lo, f"{g}_hi": hi, f"{g}_n": int(msk.sum())})
        r["mwu_p"] = float(mannwhitneyu(v[pr], v[st]).pvalue) if pr.sum() and st.sum() else np.nan
        r["wilcoxon_vs0_p"] = float(wilcoxon(v).pvalue)
        rows.append(r)
    B2 = pd.DataFrame(rows); B2["mwu_q"] = bh(B2.mwu_p.fillna(1)); B2.to_csv(OUT / "B2_concepts.csv", index=False)
    pd.DataFrame(CA, columns=[SHORT[n] for n in names]).assign(pid=C["F"].pid.values, group=C["grp"]).to_csv(OUT / "B2_concepts_per_patient.csv", index=False)
    return dict(sanity_passed=True), B1, B2


def part_C(C, GR, pick=None, csv_name="C_macula.csv", maps_name=None):
    F = C["F"]; grp = C["grp"]
    pr = sorted([i for i in range(len(F)) if grp[i] == "Only up"], key=lambda i: -GR[i])[:2]
    sb = sorted([i for i in range(len(F)) if grp[i] == "Stable"], key=lambda i: GR[i])[:2]
    if pick is not None:
        pr, sb = list(pick), []
    v, _ = load_visual_encoder(dev)
    mn = torch.from_numpy(C["scaler"].min).to(dev); sc = torch.from_numpy(C["scaler"].scale).to(dev)
    mean = torch.tensor(CLIP_MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(CLIP_STD, device=dev).view(1, 3, 1, 1)
    rows, maps = [], {}
    for i in pr + sb:
        r = F.iloc[i]
        x_img = torch.stack([clip_preprocess(EDO_DIR / r.le_file), clip_preprocess(EDO_DIR / r.re_file)]).to(dev)
        base = (torch.zeros_like(x_img) - mean) / std
        g = torch.tensor([[float(r.grade) / 5.0]], dtype=torch.float32, device=dev)
        x, t, y, m, _ = collate([C["ds"][i]]); q, i0, i5 = qidx(t[0][m[0]])

        def f(imgs):
            B = imgs.shape[0]; feats = v(imgs.reshape(B * 2, 3, 224, 224)).reshape(B, 1024)
            xin = torch.cat([(feats - mn) / sc, g.expand(B, 1)], 1)
            return torch.stack([(lambda o: o[:, i5] - o[:, i0])(mod(xin, q[None].expand(B, -1))) for mod in C["models"]]).mean(0), feats
        with torch.no_grad():
            fx, feats = f(x_img[None]); fb, _ = f(base[None])
        stored = torch.from_numpy(C["X"][r.feature_row]).to(dev)
        enc_cos = float(Fnn.cosine_similarity(feats[0], stored, dim=0))
        grad = torch.zeros_like(x_img); al = (torch.arange(PIX_STEPS, dtype=torch.float32, device=dev) + 0.5) / PIX_STEPS
        for c0 in range(0, PIX_STEPS, CH):
            a = al[c0:c0 + CH].view(-1, 1, 1, 1, 1)
            xs = (base[None] + a * (x_img - base)[None]).requires_grad_(True)
            out, _ = f(xs); gg, = torch.autograd.grad(out.sum(), xs); grad += gg.sum(0)
        IGp = ((x_img - base) * grad / PIX_STEPS).detach()
        h = IGp.sum(1).cpu().numpy(); img = (x_img * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        fund = img.max(-1) > 0.04; yy, xx = np.mgrid[:224, :224]; centre = (yy - 111.5) ** 2 + (xx - 111.5) ** 2 <= (0.25 * 224) ** 2
        row = dict(pid=r.pid, group=grp[i], first_grade=r.grade, grades=r.grades, predicted_growth=float(fx), completeness_err=abs(float(IGp.sum()) - float(fx - fb)),
                   encoder_cos_vs_stored=enc_cos)
        for e, eye in enumerate(("LE", "RE")):
            Aa = np.abs(h[e]); row[f"{eye}_macula_share"] = float(Aa[centre & fund[e]].sum() / Aa.sum()); row[f"{eye}_macula_area"] = float((centre & fund[e]).sum() / fund[e].sum())
        rows.append(row); maps[r.pid] = dict(ig=h, img=img, fund=fund)
    CT = pd.DataFrame(rows); CT.to_csv(OUT / csv_name, index=False)
    if maps_name:
        np.savez_compressed(OUT / maps_name, pids=np.array(list(maps)), **{f"{p}_{k}": v for p, d in maps.items() for k, v in d.items()})
    return CT


def main():
    C = setup(); res = dict(dry=DRY, V_star=C["vs"], models=C["dirs"], n_patients=len(C["F"]), errors=[])
    IG, GR, ERR = ig_all(C); np.savez(OUT / "ig_seedmean.npz", ig=IG, growth=GR, pids=C["F"].pid.values.astype(str))
    for name, fn in (("A", lambda: part_A(C, IG, GR, ERR)), ("B", lambda: part_B(C, IG)), ("C", lambda: part_C(C, GR))):
        try:
            out = fn()
            if name == "A": res["A"] = out
            if name == "B": res["B"] = out[0]; res["B1"] = out[1].to_dict("records"); res["B2"] = None if out[2] is None else out[2].to_dict("records")
            if name == "C": res["C"] = out.to_dict("records")
        except Exception:
            res["errors"].append(f"{name}: {traceback.format_exc()}")
    json.dump(res, open(OUT / "explain_summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("B1", "B2")}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
