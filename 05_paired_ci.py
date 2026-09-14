#!/usr/bin/env python
"""
05_paired_ci.py — paired confidence intervals on the R@3 and FSR differences.

WHY THIS EXISTS
  Section 5.3 cannot use Pareto or "strict improvement on every axis" language until
  these exist. R@3 falls under projection on both pipelines, by 0.0018 and 0.0043. If
  the paired interval includes zero the sentence becomes "attains exact validity with
  no significant change in retrieval"; if it excludes zero it is a small measured cost
  and the dominance wording goes. Either way the headline result stands and only the
  adjective changes. FSR needs the same treatment and is currently missing from the
  main table entirely.

  Because projection is applied to the same samples, the per-replication difference
  removes sampling variance, which is why the FID t-statistics are so large. The same
  pairing must be used here.

NO GPU, NO CLUSTER. Runs on the login node or a laptop in under a second.

INPUT
  The per-replication values from the replicated_ci run. Either:
    a) replicated_ci.json with a structure containing per-rep lists, or
    b) a small CSV you write by hand: variant,rep,fid,r3,ble,fsr

USAGE
  python 05_paired_ci.py replicated_ci.json
  python 05_paired_ci.py per_rep.csv --pairs "LFM:CLFM+posthoc,CDFM:CDFM+posthoc"
"""
import sys, json, csv, argparse
import numpy as np

TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 9: 2.262,
         14: 2.145, 19: 2.093, 29: 2.045}


def tcrit(df):
    if df in TCRIT:
        return TCRIT[df]
    ks = sorted(TCRIT)
    return TCRIT[min(ks, key=lambda k: abs(k - df))]


def paired(a, b, label, metric, higher_is_better):
    """b minus a, per replication."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(len(a), len(b))
    if n < 2:
        print(f"  {label:<28}{metric:<6} only {n} replication(s); cannot form an interval")
        return None
    d = b[:n] - a[:n]
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    t = tcrit(n - 1)
    lo, hi = m - t * se, m + t * se
    tstat = m / se if se > 0 else float("inf")
    crosses = (lo <= 0 <= hi)
    good = (m > 0) if higher_is_better else (m < 0)
    verdict = ("no significant change" if crosses
               else ("significant improvement" if good else "significant cost"))
    print(f"  {label:<28}{metric:<6}{m:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  "
          f"t({n-1})={tstat:+.2f}  {verdict}")
    return dict(metric=metric, n=n, mean=float(m), lo=float(lo), hi=float(hi),
                t=float(tstat), crosses_zero=bool(crosses), verdict=verdict)


def load(path):
    """Return {variant: {metric: [per-rep values]}}."""
    if path.endswith(".json"):
        raw = json.load(open(path))
        out = {}

        def walk(node, name=None):
            if isinstance(node, dict):
                keys = {k.lower() for k in node}
                if {"fid", "r3"} & keys and any(
                        isinstance(node[k], list) for k in node):
                    out[name or "unnamed"] = {k.lower(): v for k, v in node.items()
                                              if isinstance(v, list)}
                for k, v in node.items():
                    walk(v, k if name is None else f"{name}/{k}")
            elif isinstance(node, list) and name and all(
                    isinstance(x, dict) for x in node):
                keys = set().union(*[set(x) for x in node])
                if {"fid", "R3"} & keys or {"fid", "r3"} & keys:
                    out[name] = {m.lower(): [x.get(m) for x in node]
                                 for m in keys if isinstance(node[0].get(m), (int, float))}
        walk(raw)
        if not out:
            sys.exit(f"Could not find per-replication lists in {path}. "
                     f"Top-level keys: {list(raw)[:10]}. Use the CSV form instead.")
        return out

    out = {}
    for r in csv.DictReader(open(path)):
        v = out.setdefault(r["variant"], {})
        for m in ("fid", "r3", "ble", "fsr"):
            if r.get(m) not in (None, ""):
                v.setdefault(m, []).append(float(r[m]))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--pairs", default="",
                help="comma-separated baseline:projected pairs; "
                     "omit to be shown the available variant names")
a = ap.parse_args()

data = load(a.path)
print(f"\nvariants found: {', '.join(sorted(data))}\n")

if not a.pairs:
    print("Pass --pairs baseline:projected[,baseline:projected] using names above.")
    sys.exit(0)

print("=" * 96)
print("PAIRED DIFFERENCES, projected minus baseline, per replication")
print("=" * 96)
res = {}
for pair in a.pairs.split(","):
    base, proj = pair.split(":")
    if base not in data or proj not in data:
        print(f"  skipping {pair}: name not found")
        continue
    rs = []
    for metric, hib in (("fid", False), ("r3", True), ("ble", False), ("fsr", False)):
        if metric in data[base] and metric in data[proj]:
            r = paired(data[base][metric], data[proj][metric], pair, metric, hib)
            if r:
                rs.append(r)
    res[pair] = rs
    print()

print("=" * 96)
print("""
WORDING THAT FOLLOWS

  R@3 interval crosses zero
    "Post-decode projection attains exact bone validity and lowers FID with no
     significant change in retrieval (paired 95% CI on R@3 includes zero)."
     Do NOT write Pareto dominance -- a non-significant decrease is not an improvement.

  R@3 interval excludes zero and is negative
    "Projection attains exact validity and lowers FID at a small measured retrieval
     cost of X (95% CI ...)." Drop every dominance and strict-improvement phrase.

  Either way, report the paired statistic rather than the overlap of the marginal
  intervals, as the FID result already does.
""")
