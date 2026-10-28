"""When do drops happen relative to (re)spawn? Uses the nominal eval rollouts (60 s, no time-outs, respawn after drop).

Usage (cwd = logs/rsl_rl/orca_v2_repose): python drop_timing.py RUN_NAME [RUN_NAME ...]  (CPU only, numpy)
"""
import numpy as np, glob, sys
bins = [0, 1, 2, 3, 5, 10, 20, 60]
runs = sys.argv[1:]
tot_drop = np.zeros(len(bins) - 1); tot_expo = np.zeros(len(bins) - 1)
print(f"{'run':28s} drops  " + " ".join(f"{a}-{b}s".rjust(7) for a, b in zip(bins, bins[1:])) + "   | w_z(cmd dir) rad/s by time since spawn: 0-1 1-2 2-3 3-5 5-10 10-60")
for r in runs:
    f = sorted(glob.glob(f"*_{r}/eval/model_*.npz"), key=lambda p: int(p.split("model_")[1].split(".")[0]))
    f = [p for p in f if "perturbed" not in p and "_cmd" not in p][-1]
    d = np.load(f); dt = float(d["dt"]); done = d["done"] > 0; T, N = done.shape
    sgn = np.sign(d["cmd_angvel"][0, :, 2]); w = d["obj_angvel"][..., 2] * sgn
    # time since spawn per step (spawn at t=0 and the step after each drop)
    tss = np.zeros((T, N)); c = np.zeros(N)
    for t in range(T):
        tss[t] = c * dt; c = np.where(done[t], 0, c + 1)
    dt_drop = tss[done]
    drops = np.histogram(dt_drop, bins)[0]
    expo = np.histogram(tss.ravel(), bins)[0] * dt  # env-seconds spent in each bin
    tot_drop += drops; tot_expo += expo
    haz = drops / np.maximum(expo, 1e-9) * 60  # drops per env-minute
    wb = [w[(tss >= a) & (tss < b)].mean() for a, b in [(0,1),(1,2),(2,3),(3,5),(5,10),(10,60)]]
    print(f"{r:28s} {done.sum():5d}  " + " ".join(f"{h:7.2f}" for h in haz) + "   | " + " ".join(f"{x:4.2f}" for x in wb))
haz = tot_drop / tot_expo * 60
print(f"{'ALL (drops/env-min)':28s} {int(tot_drop.sum()):5d}  " + " ".join(f"{h:7.2f}" for h in haz))
print("drop counts per bin:", tot_drop.astype(int).tolist())
print("share of drops within 2 s of spawn: %.0f %%; share of env-time within 2 s: %.1f %%" % (
    100 * tot_drop[:2].sum() / tot_drop.sum(), 100 * tot_expo[:2].sum() / tot_expo.sum()))
