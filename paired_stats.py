#!/usr/bin/env python
# TODO 2 without waiting for a re-run: computes the paired FID statistics from per-rep values you
# already have. Works on a laptop; no cluster, no GPU, no W&B export.
#
# Three ways to supply the numbers:
#   python paired_stats.py --json replicated_ci_v2_raw.json
#   python paired_stats.py --base 0.147,0.179,... --proj 0.141,0.170,...
#   python paired_stats.py --demo          # uses the known LFM unconstrained per-rep values only
#
# Prints exactly the sentence Section 6.4 asks for: mean paired difference, 95% CI, t, and p.

import argparse, json, math, sys

def tsf(t, df):
    try:
        from scipy import stats
        return 2.0 * float(stats.t.sf(abs(t), df))
    except Exception:
        pass
    x = df / (df + t * t)
    def betacf(a, b, x, it=300):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30); h = d
        for m in range(1, it):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 / ((1.0 + aa * d) if abs(1.0 + aa * d) > 1e-30 else 1e-30)
            c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30); h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 / ((1.0 + aa * d) if abs(1.0 + aa * d) > 1e-30 else 1e-30)
            c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
            de = d * c; h *= de
            if abs(de - 1.0) < 1e-13: break
        return h
    a, b = df / 2.0, 0.5
    if x < (a + 1) / (a + b + 2):
        ib = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log(1 - x)) * betacf(a, b, x) / a
    else:
        ib = 1.0 - math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                            + b * math.log(1 - x) + a * math.log(x)) * betacf(b, a, 1 - x) / b
    return float(min(1.0, max(0.0, ib)))

TCRIT = {5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262,12:2.201,15:2.145,
         19:2.093,20:2.093,25:2.064,30:2.045}

def report(base, proj, name):
    n = len(base)
    assert n == len(proj), "the two lists must be matched seed by seed"
    d = [b - p for b, p in zip(base, proj)]
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n); t = m / se if se > 0 else float("inf")
    tc = TCRIT.get(n - 1, 2.093 if n <= 31 else 1.96)
    lo, hi = m - tc * se, m + tc * se
    p = tsf(t, n - 1)
    sig = lo > 0 or hi < 0
    print(f"\n{name}  (n = {n} matched sampling seeds)")
    print(f"  unconstrained mean FID  {sum(base)/n:.4f}")
    print(f"  post-decode  mean FID   {sum(proj)/n:.4f}")
    print(f"  mean paired difference  {m:+.4f}   SD {sd:.4f}")
    print(f"  95% CI                  [{lo:+.4f}, {hi:+.4f}]")
    print(f"  paired t({n-1}) = {t:+.3f}   p = {p:.3g}   -> {'significant' if sig else 'NOT significant'}")
    print(f"\n  Sentence for Section 6.5:")
    if sig:
        print(f"    Post-decode projection lowers FID by {m:.4f} on average "
              f"(95% CI [{lo:.4f}, {hi:.4f}], paired t({n-1}) = {t:.2f}, p = {p:.3g}, "
              f"{n} matched sampling seeds).")
    else:
        print(f"    The mean paired FID difference is {m:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}], "
              f"p = {p:.3g}); the improvement is not statistically significant at the 5% level "
              f"and is reported as such.")

LFM_BASE = [0.14733, 0.17908, 0.13974, 0.16361, 0.17824, 0.17122, 0.16724, 0.18193, 0.18304,
            0.16294, 0.16730, 0.16923, 0.18167, 0.15315, 0.16394, 0.16922, 0.18590, 0.17919,
            0.16697, 0.16830]

ap = argparse.ArgumentParser()
ap.add_argument("--json"); ap.add_argument("--base"); ap.add_argument("--proj")
ap.add_argument("--name", default="unconstrained vs post-decode")
ap.add_argument("--demo", action="store_true")
a = ap.parse_args()

if a.json:
    raw = json.load(open(a.json))
    raw = raw.get("raw", raw)
    pairs = [("LFM (unconstrained)", "CLFM + post-hoc proj", "LFM"),
             ("CDFM (unconstrained)", "CDFM + post-hoc proj", "CDFM")]
    for u, p, nm in pairs:
        if u in raw and p in raw:
            report(raw[u]["fid"], raw[p]["fid"], f"{nm}: unconstrained vs post-decode projection")
elif a.base and a.proj:
    f = lambda s: [float(x) for x in s.replace(" ", "").split(",") if x]
    report(f(a.base), f(a.proj), a.name)
elif a.demo:
    print("LFM unconstrained per-rep FID (mean %.4f, SD %.4f)" %
          (sum(LFM_BASE) / len(LFM_BASE),
           math.sqrt(sum((x - sum(LFM_BASE) / len(LFM_BASE)) ** 2 for x in LFM_BASE) / (len(LFM_BASE) - 1))))
    print("Supply the matched post-decode values with --proj to complete the paired test.")
else:
    ap.print_help(); sys.exit(1)
