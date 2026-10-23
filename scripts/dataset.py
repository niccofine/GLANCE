"""Dataset build (locked once written).
Read-only on the raw grades spreadsheet and the fundus image folder.
Nothing here is re-run once data/*.csv and data/*.npz exist: every subcommand refuses to overwrite its final output.

  md5scan      : MD5 of every image file -> results/dataset/md5_all_images.csv
  exclusions   : write / apply data/exclusions_v2.csv; pair LE/RE per visit; stop on cross-patient duplicates -> results/dataset/final/pairing_final.csv
  features     : RET-CLIP vectors for images -> data/features_v2_added.npz, data/features_v2_pairs.{npz,csv}
  final        : final cohort, starting visits, test set 20% + 5-fold cross-validation on construction set -> data/{cohort_v2.csv, samples_v2.csv, split_v2_test.csv, split_v2_cv5.csv}
  update_check : re-verify the final cohort/split after a manual image correction; exits 2 if a rebuild is needed

Can be run on its own or through scripts/driver.sh:
  python -m scripts.dataset md5scan
  python -m scripts.dataset exclusions
  python -m scripts.dataset features
  python -m scripts.dataset final
  python -m scripts.dataset update_check
"""
import argparse
import hashlib
import sys
from pathlib import PurePosixPath
import numpy as np, pandas as pd
from glance.config import DATA, EDO_DIR, GRADES_XLSX, GRADES_SHEET, LATE_THRESHOLD as L, VISITS_PER_YEAR, RESULTS_DATASET, RETCLIP_FEATURES, RETCLIP_IMAGE_PATHS
from glance.data import load_grades
from glance.encoder import modeldef, extrvencoder, extrduallatent

FIN = RESULTS_DATASET / "final"
EXCL_CSV = DATA / "exclusions_v2.csv"
GG = ["0", "1", "2", "3", "4-5"]
gg = lambda g: "4-5" if g >= 4 else str(int(g))
SEED = 2027


def load_exclusions():
    return pd.read_csv(EXCL_CSV, dtype={"visno": "Int64"})


def inventory(excl=None, path=None):
    cur = pd.read_csv(path or RESULTS_DATASET / "md5_all_images.csv", dtype=str); cur["visno"] = cur.visit.astype(int)
    if excl is not None:
        cur = cur[~cur.pid.isin(excl[excl.kind == "patient"].pid)]
        cur = cur[~cur.path.isin(excl[excl.kind == "file"].path)]
    return cur.reset_index(drop=True)


def grades(excl=None):
    G = load_grades()
    if excl is not None:
        for p in excl[excl.kind == "patient"].pid: G.pop(p, None)
        for r in excl[excl.kind == "grade"].itertuples():
            G[r.pid] = [(v, g) for v, g in G[r.pid] if v != int(r.visno)]
    return G


def retclip_first():
    pairs = np.load(RETCLIP_IMAGE_PATHS, allow_pickle=True); rel = lambda s: "/".join(PurePosixPath(s).parts[-5:])
    first, row = {}, {}
    for i, (le, re_) in enumerate(pairs):
        a, b = rel(le), rel(re_); row[a] = (i, 0); row[b] = (i, 1)
        pp = PurePosixPath(le).parts; first.setdefault((pp[-5], pp[-4], int(pp[-2])), (i, a, b))
    return first, row


def pair_visits(cur, G):
    first, _ = retclip_first(); GD = {p: dict(v) for p, v in G.items()}
    J = cur[cur.ext == ".jpg"]; pr, multi = [], []
    for (p, v), d in J.groupby(["pid", "visno"]):
        pick = {}
        for (view, eye), e in d.groupby(["view", "eye"]):
            files = sorted(e.path); prev = first.get((p, view, v)); prev_f = (prev[1] if eye == "LE" else prev[2]) if prev else None
            if len(files) == 1: pick[(view, eye)] = files[0]; continue
            ch, rule = (prev_f, "previous pipeline file") if prev_f in files else (files[0], "first by file name")
            pick[(view, eye)] = ch; multi.append(dict(pid=p, visno=v, view=view, eye=eye, n_files=len(files), files=" | ".join(files), chosen=ch, rule=rule))
        sides = {eye: {vw for (vw, e) in pick if e == eye} for eye in ("LE", "RE")}
        if "F2_LS" in sides["LE"] & sides["RE"]: kind, lv, rv = "same side", "F2_LS", "F2_LS"
        elif "F2_RS" in sides["LE"] & sides["RE"]: kind, lv, rv = "same side", "F2_RS", "F2_RS"
        elif sides["LE"] and sides["RE"]: kind, lv, rv = "different side", sorted(sides["LE"])[0], sorted(sides["RE"])[0]
        else: kind, lv, rv = "unpaired", None, None
        g = GD.get(p, {}).get(v, np.nan)
        pr.append(dict(pid=p, visno=v, grade=g, graded=not np.isnan(g), pairing=kind, le_view=lv or "", re_view=rv or "",
                       le_file=pick.get((lv, "LE"), "") if lv else "", re_file=pick.get((rv, "RE"), "") if rv else "",
                       has_le=bool(sides["LE"]), has_re=bool(sides["RE"])))
    return pd.DataFrame(pr), pd.DataFrame(multi, columns=["pid", "visno", "view", "eye", "n_files", "files", "chosen", "rule"])


def starts(PR, G, need_feature=None):
    out = {}
    for r in PR[PR.graded & (PR.pairing != "unpaired")].itertuples():
        if need_feature is not None and (r.pid, r.visno) not in need_feature: continue
        if any(v > r.visno for v, _ in G[r.pid]): out.setdefault(r.pid, []).append(r.visno)
    return {p: sorted(v) for p, v in out.items()}


def traj_group(y):
    d = np.diff(y); up, dn = (d > 0).any(), (d < 0).any()
    return "Up and down" if up and dn else "Only up" if up else "Only down" if dn else "Stable"


def e1(t, y):
    i = np.where(y[1:] >= L)[0]; return float(t[1:][i[0]]) if len(i) else np.nan


def e2(t, y):
    for i in range(1, len(y)):
        if y[i] >= L and (i == len(y) - 1 or y[i + 1] >= L): return float(t[i])
    return np.nan


def patient_table(ST, G, PR):
    rows, samp = [], []
    prk = PR.set_index(["pid", "visno"])
    for p in sorted(ST):
        for k, v0 in enumerate(ST[p]):
            vv = [(v, g) for v, g in G[p] if v >= v0]
            t = np.array([(v - v0) / VISITS_PER_YEAR for v, _ in vv]); y = np.array([g for _, g in vv], float)
            r0 = prk.loc[(p, v0)]
            samp.append(dict(pid=p, visno=v0, first=k == 0, grade=y[0], pairing=r0.pairing, le_file=r0.le_file, re_file=r0.re_file,
                             times_y=" ".join(f"{x:g}" for x in t), grades=" ".join(f"{x:g}" for x in y)))
            if k: continue
            g0 = int(y[0]); late0 = g0 >= L; fut = y[1:]; below = fut < L; conf_rev = bool((below[:-1] & below[1:]).any())
            rows.append(dict(patient_id=p, first_visno=v0, first_pairing=r0.pairing, first_grade=g0, group=gg(g0), n_visits=len(y),
                             n_starts=len(ST[p]), followup_y=t[-1], last_grade=int(y[-1]), max_grade=int(y.max()), traj=traj_group(y),
                             any_drop=bool((np.diff(y) < 0).any()), late_first=late0,
                             e1_t=np.nan if late0 else e1(t, y), e2_t=np.nan if late0 else e2(t, y),
                             reach5=bool(g0 == 4 and (fut == 5).any()), any_below4=bool(late0 and below.any()),
                             conf_rev=bool(late0 and conf_rev), times=" ".join(f"{x:g}" for x in t), grades=" ".join(f"{x:g}" for x in y)))
    P = pd.DataFrame(rows); P["e1"] = P.e1_t.notna(); P["e2"] = P.e2_t.notna()
    P["e1_time"] = P.e1_t.fillna(P.followup_y); P["e2_time"] = P.e2_t.fillna(P.followup_y)
    P["stratum"] = P.group + "_" + P.e2.astype(int).astype(str)
    return P, pd.DataFrame(samp)


def _md5(f):
    h = hashlib.md5()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_md5scan():
    RESULTS_DATASET.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DATASET / "md5_all_images.csv"
    rows = []
    for f in sorted(p for p in EDO_DIR.rglob("*") if p.is_file()):
        rel = f.relative_to(EDO_DIR); parts = rel.parts
        rows.append(dict(path=str(rel), pid=parts[0], view=parts[1] if len(parts) > 2 else "", eye=parts[2] if len(parts) > 3 else "",
                         visit=parts[3] if len(parts) > 4 else "", source_id=f.name.split(" ")[0], ext=f.suffix.lower(),
                         size=f.stat().st_size, md5=_md5(f)))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(len(rows), "files")


def cmd_exclusions():
    FIN.mkdir(parents=True, exist_ok=True)
    G456 = [f"G456/F2_RS/RE/{v:02d}/52283 {v:02d} F2 RE RS.jpg" for v in (4, 12, 16, 18, 20, 22)]
    if not EXCL_CSV.exists():
        rows = [dict(kind="patient", pid="G101", visno=None, path="", reason="whole patient excluded: two image sets at the same visits", decided_by="", date=""),
                *[dict(kind="grade", pid="G351", visno=v, path="", reason="grade belonged to removed duplicated images; no image, not a target", decided_by="", date="") for v in (0, 6)],
                *[dict(kind="file", pid="G456", visno=int(p.split("/")[3]), path=p, reason="byte-identical to another patient's image at the same visit", decided_by="", date="") for p in G456]]
        pd.DataFrame(rows).astype({"visno": "Int64"}).to_csv(EXCL_CSV, index=False)
    EX = load_exclusions()
    for p in EX[EX.kind == "file"].path:
        assert (EDO_DIR / p).exists(), f"excluded file not on disk: {p}"
    CUR = inventory(EX); G = grades(EX)
    PR, MULTI = pair_visits(CUR, G)
    PR.to_csv(FIN / "pairing_final.csv", index=False)

    used = sorted(set(PR.le_file[PR.le_file != ""]) | set(PR.re_file[PR.re_file != ""]))
    U = pd.DataFrame([dict(path=p, pid=p.split("/")[0], md5=_md5(EDO_DIR / p)) for p in used])
    cross_patient = sum(len(set(x.pid)) > 1 for _, x in U[U.md5.duplicated(keep=False)].groupby("md5"))
    print(f"files used {len(U)} | cross-patient duplicates among them: {cross_patient}")
    if cross_patient:
        print("STOP: cross-patient duplicates remain"); sys.exit(1)


def cmd_features():
    EX = load_exclusions(); CUR = inventory(EX)
    PR = pd.read_csv(FIN / "pairing_final.csv").fillna({"le_file": "", "re_file": ""})
    _, ROW = retclip_first(); X = np.load(RETCLIP_FEATURES)
    out_add, out_pairs = DATA / "features_v2_added.npz", DATA / "features_v2_pairs.npz"
    assert not out_add.exists() and not out_pairs.exists(), "feature files exist: not overwritten"

    venc = extrvencoder(*modeldef())
    try:
        import torch
        venc = venc.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    except ImportError:
        pass

    def enc_pair(le, re_):
        return extrduallatent((str(EDO_DIR / le), str(EDO_DIR / re_)), venc, None)[0]

    J = CUR[(CUR.ext == ".jpg") & ~CUR.path.isin(ROW)].sort_values("path")
    sel = set(PR.le_file) | set(PR.re_file); sel.discard("")
    vecs = []
    for p in J.path:
        f = enc_pair(p, p); assert np.array_equal(f[:512], f[512:]); vecs.append(f[:512].astype(np.float32))
    V = np.stack(vecs) if vecs else np.zeros((0, 512), np.float32)
    A = J[["path", "pid", "view", "eye", "visno", "md5"]].reset_index(drop=True)
    A["selected_in_pairing"] = A.path.isin(sel); A["row"] = np.arange(len(A))
    np.savez(out_add, vectors=V, paths=np.array(list(A.path), dtype=str), md5=np.array(list(A.md5), dtype=str))
    A.to_csv(DATA / "features_v2_added.csv", index=False)

    ADD = {p: i for i, p in enumerate(A.path)}
    def eye_vec(p):
        if p in ROW: r, h = ROW[p]; return X[r][h * 512:(h + 1) * 512], f"orig:{r}"
        return V[ADD[p]], f"added:{ADD[p]}"
    P = PR[PR.pairing != "unpaired"].reset_index(drop=True); pv, src = [], []
    for r in P.itertuples():
        (lv, ls), (rv, rs) = eye_vec(r.le_file), eye_vec(r.re_file); pv.append(np.concatenate([lv, rv])); src.append((ls, rs))
    PV = np.stack(pv).astype(np.float32)
    P["le_src"], P["re_src"] = zip(*src)
    P["vector"] = np.where(P.le_src.str.startswith("orig") & (P.le_src.str.split(":").str[1] == P.re_src.str.split(":").str[1]), "existing feature row",
                  np.where(P.le_src.str.startswith("added") | P.re_src.str.startswith("added"), "new (added vectors)", "composed from two feature rows"))
    P["row"] = np.arange(len(P))
    np.savez(out_pairs, X=PV, pid=np.array(list(P.pid), dtype=str), visno=P.visno.values.astype(int), le_file=np.array(list(P.le_file), dtype=str), re_file=np.array(list(P.re_file), dtype=str))
    P.drop(columns=["has_le", "has_re"]).to_csv(DATA / "features_v2_pairs.csv", index=False)
    print(f"added {len(A)} images | pairs {len(P)}, by source: {P.vector.value_counts().to_dict()}")


def cmd_final():
    outs = [DATA / f for f in ("cohort_v2.csv", "samples_v2.csv", "split_v2_test.csv", "split_v2_cv5.csv")]
    assert not any(f.exists() for f in outs), "data/*_v2 files exist: not overwritten"

    EX = load_exclusions(); CUR = inventory(EX); G = grades(EX)
    PR = pd.read_csv(FIN / "pairing_final.csv").fillna({"le_file": "", "re_file": ""})
    FP = pd.read_csv(DATA / "features_v2_pairs.csv")
    assert (FP.set_index(["pid", "visno"])[["le_file", "re_file"]].sort_index().equals(
            PR[PR.pairing != "unpaired"].set_index(["pid", "visno"])[["le_file", "re_file"]].sort_index())), "feature pairs != pairing"
    HASF = set(zip(FP.pid, FP.visno))
    ST = starts(PR, G, need_feature=HASF); ST_nof = starts(PR, G)
    assert ST == ST_nof
    P, SAMP = patient_table(ST, G, PR)
    frow = FP.set_index(["pid", "visno"])["row"]
    SAMP["feature_row"] = [int(frow[(p, v)]) for p, v in zip(SAMP.pid, SAMP.visno)]
    SAMP["feature_kind"] = [FP.set_index(["pid", "visno"]).loc[(p, v), "vector"] for p, v in zip(SAMP.pid, SAMP.visno)]

    DUP_PAIRS = (("G175", "G456"), ("G273", "G474"), ("G336", "G397"), ("G351", "G577"))
    COH = set(P.patient_id)
    pairs_keep = [(a, b) for a, b in DUP_PAIRS if a in COH and b in COH]
    partner = {}
    for a, b in pairs_keep: partner[a] = b; partner[b] = a
    strat = P.set_index("patient_id").stratum
    rng = np.random.default_rng(SEED)
    NT = int(round(0.2 * len(P))); cnt = P.stratum.value_counts().sort_index()
    quota = {s: int(np.floor(0.2 * n)) for s, n in cnt.items()}; rem = {s: 0.2 * n - quota[s] for s, n in cnt.items()}
    for s in sorted(rem, key=lambda s: (-rem[s], s))[:NT - sum(quota.values())]: quota[s] += 1
    part, used_q = {}, {s: 0 for s in quota}
    order = []
    for s in sorted(cnt.index):
        ids = sorted(P[P.stratum == s].patient_id); rng.shuffle(ids); order += ids
    for pid in order:
        if pid in part: continue
        unit = [pid] + ([partner[pid]] if pid in partner and partner[pid] not in part else [])
        need = pd.Series([strat[u] for u in unit]).value_counts()
        fits = all(used_q[s] + k <= quota[s] for s, k in need.items())
        for u in unit: part[u] = "test" if fits else "constr"
        if fits:
            for s, k in need.items(): used_q[s] += k
    P["part"] = P.patient_id.map(part)
    C = P[P.part == "constr"].copy(); fold = {}; ntot = np.zeros(5, int); ne2 = np.zeros(5, int)
    rg = np.random.default_rng(SEED)
    for s, g in C.groupby("stratum"):
        ids = sorted(g.patient_id); rg.shuffle(ids); nst = np.zeros(5, int); conv = s.endswith("_1")
        for pid in ids:
            q = partner.get(pid)
            f = fold[q] if q in fold else min(range(5), key=lambda k: (nst[k], ne2[k] if conv else ntot[k], ntot[k], k))
            fold[pid] = f; nst[f] += 1; ntot[f] += 1; ne2[f] += conv
    C["fold"] = C.patient_id.map(fold) + 1
    for a, b in pairs_keep:
        assert part[a] == part[b] and (part[a] == "test" or fold[a] == fold[b])

    cols = ["patient_id", "first_visno", "first_pairing", "first_grade", "group", "n_visits", "n_starts", "followup_y", "last_grade", "max_grade",
            "traj", "any_drop", "late_first", "e1", "e1_t", "e2", "e2_t", "e1_time", "e2_time", "reach5", "any_below4", "conf_rev", "stratum", "times", "grades"]
    P[cols].to_csv(DATA / "cohort_v2.csv", index=False)
    SAMP.to_csv(DATA / "samples_v2.csv", index=False)
    P[["patient_id", "part", "first_grade", "group", "e2", "stratum"]].to_csv(DATA / "split_v2_test.csv", index=False)
    C[["patient_id", "fold", "first_grade", "group", "e2", "stratum"]].sort_values(["fold", "patient_id"]).to_csv(DATA / "split_v2_cv5.csv", index=False)
    print(f"cohort {len(P)} | starts {len(SAMP)} | test {int((P.part == 'test').sum())} | constr {len(C)}")


def cmd_update_check():
    EX = load_exclusions(); CUR = inventory(EX)
    PR, _ = pair_visits(CUR, grades(EX))
    used = sorted(set(PR.le_file[PR.le_file != ""]) | set(PR.re_file[PR.re_file != ""]))
    U = pd.DataFrame([dict(path=p, pid=p.split("/")[0], md5=_md5(EDO_DIR / p)) for p in used])
    cross_patient_used = sum(len(set(x.pid)) > 1 for _, x in U[U.md5.duplicated(keep=False)].groupby("md5"))
    cross_patient_remaining = sum(len(set(x.pid)) > 1 for _, x in CUR[CUR.md5.duplicated(keep=False)].groupby("md5"))
    if cross_patient_used or cross_patient_remaining:
        print("STOP: cross-patient duplicates"); sys.exit(1)

    FP = pd.read_csv(DATA / "features_v2_pairs.csv")
    key = ["pid", "visno"]; a = FP.set_index(key)[["pairing", "le_file", "re_file"]]; b = PR[PR.pairing != "unpaired"].set_index(key)[["pairing", "le_file", "re_file"]]
    pairs_changed = sorted(set(a.index) ^ set(b.index)) + [k for k in a.index.intersection(b.index) if not a.loc[k].equals(b.loc[k])]

    HASF = set(zip(FP.pid, FP.visno))
    ST = starts(PR, grades(EX), need_feature=HASF); P, _ = patient_table(ST, grades(EX), PR)
    C0 = pd.read_csv(DATA / "cohort_v2.csv").set_index("patient_id"); N = P.set_index("patient_id")
    strata_fields = ("stratum", "group", "e2", "first_grade")
    changed = False
    for p in sorted(set(C0.index) | set(N.index)):
        if p not in N.index or p not in C0.index:
            changed = True; continue
        for c in strata_fields:
            if str(C0.loc[p, c]) != str(N.loc[p, c]):
                changed = True
    if changed or pairs_changed:
        print("REBUILD / PAIR UPDATE NEEDED"); sys.exit(2)
    print("cohort / split unchanged: no rebuild needed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("md5scan", "exclusions", "features", "final", "update_check"))
    a = ap.parse_args()
    {"md5scan": cmd_md5scan, "exclusions": cmd_exclusions, "features": cmd_features,
     "final": cmd_final, "update_check": cmd_update_check}[a.cmd]()


if __name__ == "__main__":
    main()
