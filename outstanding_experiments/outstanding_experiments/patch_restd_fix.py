#!/usr/bin/env python
"""
patch_restd_fix.py -- apply the re-standardisation fix to a script's in-ODE branch.

I have not seen inproc_endpoint.py, so this does NOT edit blind. It finds the
re-encode line, prints it with context, and only rewrites after --apply, keeping a
timestamped backup. If the pattern does not match, it says so and changes nothing:
make the edit by hand rather than trusting a regex on a file I cannot check.

THE FIX
    before:  z = (rvq.encoder(mn2) if is_latent else mn2)
    after:   z = ((rvq.encoder(mn2) - z_mean_t) / z_std_t) if is_latent else mn2

WHY
    The decode un-standardises with z*z_std + z_mean. The re-encode returns a raw
    latent which is then consumed as if standardised, by the velocity network on the
    next step and again by the final decode. The defect exists only in the latent
    branch, which is exactly the branch that collapses, so the direct-model control
    cannot discriminate it from the structural hypothesis.

USAGE
    python patch_restd_fix.py inproc_endpoint.py            # inspect only
    python patch_restd_fix.py inproc_endpoint.py --apply
"""
import re, sys, shutil, time, argparse

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

src = open(a.path).read()
lines = src.splitlines()

PAT = re.compile(r"^(\s*)z\s*=\s*\(?\s*rvq\.encoder\(\s*(\w+)\s*\)\s*if\s+(\w+)\s+else\s+(\w+)\s*\)?\s*$")
hits = [(i, m) for i, l in enumerate(lines) for m in [PAT.match(l)] if m]

if not hits:
    print(f"No matching re-encode line in {a.path}.")
    print("Looked for:  z = (rvq.encoder(X) if FLAG else Y)")
    print("\nLines mentioning rvq.encoder, for a manual check:")
    for i, l in enumerate(lines):
        if "rvq.encoder" in l:
            print(f"  {i+1:>5}: {l.rstrip()}")
    print("\nNothing was changed. Apply the fix by hand.")
    sys.exit(1)

print(f"{len(hits)} candidate line(s) in {a.path}:\n")
for i, m in hits:
    ind, mn2, flag, alt = m.groups()
    new = f"{ind}z = ((rvq.encoder({mn2}) - z_mean_t) / z_std_t) if {flag} else {alt}"
    lo, hi = max(0, i - 4), min(len(lines), i + 3)
    for j in range(lo, hi):
        mark = ">>" if j == i else "  "
        print(f"  {mark} {j+1:>5}: {lines[j].rstrip()}")
    print(f"\n     would become:\n     {new.strip()}\n")
    print("     check above that the decode two lines up un-standardises with")
    print("     z_std_t / z_mean_t. If it does not, this fix does not apply.\n")

if not a.apply:
    print("Inspect only. Re-run with --apply to write the change.")
    sys.exit(0)

bak = f"{a.path}.bak.{time.strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(a.path, bak)
for i, m in hits:
    ind, mn2, flag, alt = m.groups()
    lines[i] = f"{ind}z = ((rvq.encoder({mn2}) - z_mean_t) / z_std_t) if {flag} else {alt}"
open(a.path, "w").write("\n".join(lines) + "\n")
print(f"patched {len(hits)} line(s). backup -> {bak}")
print("\nSanity check before trusting the rerun: the direct arm must be unchanged,")
print("since it has no encoder in the path and should reproduce its previous FID.")
