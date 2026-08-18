"""Paired analysis of seed-42 vs seed-0 pruned checkpoints on held-out code.

Question: does WHICH calibration samples were drawn change the resulting model?

The pipeline is bit-for-bit deterministic and loglikelihood scoring is deterministic,
so the noise floor is exactly zero. Any difference here is attributable to the draw.

Design notes that matter:
  * The statistic is per-document log-likelihood normalised by document bytes, so
    documents of different length contribute comparably. Pairing is by doc_id, and
    validity is VERIFIED via doc_hash rather than assumed.
  * A null result is only meaningful if the design could have detected an effect.
    So this reports the resolution diagnostic q = N/N* from arXiv 2605.30315:
        N*     = (z_a + z_b)^2 * sigma_D^2 / delta^2
        MDE(N) = (z_a + z_b) * sigma_D / sqrt(N)
    q >= 1 means the observed gap is resolvable at this N. q < 1 means the run
    CANNOT distinguish the readings and "no significant difference" is not a finding.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

try:
    from scipy import stats as sps
except ImportError:  # noqa: BLE001
    sps = None

Z_ALPHA = 1.959963985  # two-sided 0.05
Z_BETA = 0.841621234  # 80% power


def load_arm(root: Path) -> dict[int, dict]:
    files = sorted(root.rglob("samples_*.jsonl"))
    if not files:
        raise SystemExit(f"no samples jsonl under {root}")
    out: dict[int, dict] = {}
    for line in files[0].read_text().splitlines():
        r = json.loads(line)
        logprob, nbytes = r["bits_per_byte"]
        out[r["doc_id"]] = {
            "logprob": float(logprob),
            "bytes": int(nbytes),
            "doc_hash": r.get("doc_hash"),
        }
    return out


def bootstrap_ci(values: list[float], iters: int = 20000, seed: int = 1234) -> tuple[float, float]:
    import random

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main() -> None:
    base = Path.home() / "reap-eval"
    runs = sorted(base.glob("paired_n*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("no paired_n* run directory found")
    run = runs[-1]
    print(f"run: {run.name}\n")

    a = load_arm(run / "seed42")
    b = load_arm(run / "seed0")

    # --- pairing validity, verified not assumed -------------------------------
    common = sorted(set(a) & set(b))
    print("=== pairing validity ===")
    print(f"  seed42 docs: {len(a)}   seed0 docs: {len(b)}   common: {len(common)}")
    mismatched = [i for i in common if a[i]["doc_hash"] != b[i]["doc_hash"]]
    if mismatched:
        raise SystemExit(f"  !! {len(mismatched)} doc_hash mismatches - arms scored DIFFERENT text")
    print("  OK all doc_hash values match: both arms scored identical documents")
    byte_mismatch = [i for i in common if a[i]["bytes"] != b[i]["bytes"]]
    if byte_mismatch:
        raise SystemExit(f"  !! {len(byte_mismatch)} byte-count mismatches")
    print("  OK byte counts identical\n")

    # --- the paired statistic --------------------------------------------------
    # nats per byte, seed42 minus seed0. Negative means seed42 assigns LOWER
    # likelihood, i.e. seed42 is the worse model on this corpus.
    deltas = [(a[i]["logprob"] - b[i]["logprob"]) / a[i]["bytes"] for i in common]
    n = len(deltas)
    mean_d = sum(deltas) / n
    var_d = sum((d - mean_d) ** 2 for d in deltas) / (n - 1)
    sigma_d = math.sqrt(var_d)

    tot_a = sum(a[i]["logprob"] for i in common)
    tot_b = sum(b[i]["logprob"] for i in common)
    bytes_tot = sum(a[i]["bytes"] for i in common)
    bpb_a = -tot_a / bytes_tot / math.log(2)
    bpb_b = -tot_b / bytes_tot / math.log(2)

    print("=== aggregate ===")
    print(f"  seed42 bits_per_byte: {bpb_a:.6f}")
    print(f"  seed0  bits_per_byte: {bpb_b:.6f}")
    print(f"  difference          : {bpb_a - bpb_b:+.6f} bits/byte")

    print("\n=== paired per-document statistic (nats/byte, seed42 - seed0) ===")
    print(f"  n           : {n}")
    print(f"  mean delta  : {mean_d:+.6e}")
    print(f"  sigma_D     : {sigma_d:.6e}")
    n_worse = sum(1 for d in deltas if d < 0)
    n_better = sum(1 for d in deltas if d > 0)
    n_tie = sum(1 for d in deltas if d == 0)
    print(f"  seed42 better on {n_better}, worse on {n_worse}, exact ties {n_tie}")

    lo, hi = bootstrap_ci(deltas)
    print(f"  95% bootstrap CI on mean delta: [{lo:+.6e}, {hi:+.6e}]")

    print("\n=== significance ===")
    if sps is not None:
        t_stat, t_p = sps.ttest_rel(
            [a[i]["logprob"] / a[i]["bytes"] for i in common],
            [b[i]["logprob"] / b[i]["bytes"] for i in common],
        )
        print(f"  paired t-test      : t={t_stat:+.4f}  p={t_p:.6g}")
        try:
            w_stat, w_p = sps.wilcoxon(deltas)
            print(f"  Wilcoxon signed-rank: W={w_stat:.1f}  p={w_p:.6g}")
        except ValueError as exc:
            print(f"  Wilcoxon: not computable ({exc})")
    else:
        se = sigma_d / math.sqrt(n)
        t_stat = mean_d / se if se else float("nan")
        print(f"  scipy unavailable; t = {t_stat:+.4f} (compare to +/-1.96)")

    # --- resolution diagnostic -------------------------------------------------
    print("\n=== resolution diagnostic (arXiv 2605.30315) ===")
    mde = (Z_ALPHA + Z_BETA) * sigma_d / math.sqrt(n)
    print(f"  MDE at n={n}: {mde:.6e} nats/byte (80% power, alpha=0.05)")
    print(f"  observed    : {abs(mean_d):.6e} nats/byte")

    if mean_d != 0:
        n_star = ((Z_ALPHA + Z_BETA) ** 2) * var_d / (mean_d**2)
        q = n / n_star
        print(f"  N* required : {n_star:,.0f} documents")
        print(f"  q = N/N*    : {q:.3f}")
        if q >= 1:
            print("  -> RESOLVED. The observed gap is detectable at this sample size.")
        else:
            need = math.ceil(n_star)
            print(f"  -> UNRESOLVED. Need ~{need:,} docs to resolve a gap this small.")
            print("     A non-significant result here does NOT support 'harmless'.")
    else:
        print("  observed difference is exactly zero")

    print("\n=== interpretation guide ===")
    print("  significant  -> reading A: the calibration draw materially changes the model")
    print("  resolved and not significant -> reading B: underdetermined but harmless")
    print("  unresolved   -> the experiment cannot distinguish A from B; report that, not a null")


if __name__ == "__main__":
    main()
