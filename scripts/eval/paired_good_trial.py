"""Paired comparison of the good-trial rate of two policies evaluated with the same eval seed and env count.

With the same --seed, env i gets the same startup randomisation (friction, mass, gains, servo model) for both policies, so
the per-env outcomes are paired and McNemar's test on the discordant envs can be more sensitive than comparing two
independent rates (only as far as outcomes correlate per env: compare the printed paired and unpaired SE). Good trial = no drop at t >= 2 s and >= 8 RPM over the alive steps after 2 s (as cube_motion.py --spread).

Usage: python scripts/eval/paired_good_trial.py BASELINE.npz CANDIDATE.npz [CANDIDATE2.npz ...]
       python scripts/eval/paired_good_trial.py --extract ROLLOUT.npz OUT.npz   (keeps only the per-env outcome, a few kB,
       so large rollouts can be deleted; the comparison accepts either kind of file)
"""
import math
import sys

import numpy as np


def good_trials(path):
    d = np.load(path, allow_pickle=True)
    if "good" in d:  # an --extract file
        return d["good"], d["rpm"]
    dt = float(d["dt"])
    k = int(round(2.0 / dt))
    alive = ~(d["done"][k:] > 0)
    w_cmd = d["obj_angvel"][k:, :, 2] * np.sign(d["cmd_angvel"][0, :, 2])
    rpm = np.where(alive, w_cmd, 0.0).sum(0) / np.maximum(alive.sum(0), 1) * 60 / (2 * np.pi)
    return (alive.all(0) & (rpm >= 8.0)), rpm


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value: binomial test of b successes in b + c trials at p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


if __name__ == "__main__":
    if sys.argv[1] == "--extract":
        good, rpm = good_trials(sys.argv[2])
        np.savez(sys.argv[3], good=good, rpm=rpm)
        sys.exit(0)
    base, rpm_b = good_trials(sys.argv[1])
    n = base.size
    print(f"baseline {sys.argv[1]}: good-trial {100 * base.mean():.1f} %, mean RPM {rpm_b.mean():.2f} (n = {n})")
    for path in sys.argv[2:]:
        cand, rpm_c = good_trials(path)
        assert cand.size == n, "different env counts: not paired"
        b, c = int((cand & ~base).sum()), int((base & ~cand).sum())
        diff = (b - c) / n
        se_paired = math.sqrt(max(b + c - (b - c) ** 2 / n, 0)) / n
        p1, p2 = cand.mean(), base.mean()
        se_unpaired = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
        drpm = rpm_c - rpm_b
        print(f"{path}: good-trial {100 * p1:.1f} % ({100 * diff:+.1f} points; only candidate good {b}, only baseline good {c}; "
              f"paired SE {100 * se_paired:.2f} -> {diff / se_paired if se_paired else float('nan'):+.1f} SE, McNemar p = {mcnemar_p(b, c):.3g}; "
              f"unpaired SE {100 * se_unpaired:.2f}); mean RPM {rpm_c.mean():.2f} ({drpm.mean():+.2f} +- {drpm.std(ddof=1) / math.sqrt(n):.2f} paired)")
