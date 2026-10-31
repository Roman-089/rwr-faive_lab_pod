"""Replay real-hand trial CSVs (rwr_system node csv_log_path) through actuator models and compare with the measured joints.

Usage (CPU): python scripts/sim2sim/replay_hand_trial.py <hand_trials dataset dir> p7_t1 p2_t1 ...   (W&B rwr-fast/orca_v2_repose/hand_trials_2026-10-31)
Models: sim (the task's first-order PD lag Kd/Kp), servo-noplay (per-joint delay + acceleration-limited profile + stiff loop),
servo-play (+ play), servo-play-soft (profile + play in front of the task's own PD lag). Output: RMSE [deg] per joint pooled over the trials."""
import json, sys
import numpy as np
D = sys.argv[1]
USD = ["wrist","index_abd","middle_abd","pinky_abd","ring_abd","thumb_cmc","index_mcp","middle_mcp","pinky_mcp","ring_mcp",
       "thumb_abd","index_pip","middle_pip","pinky_pip","ring_pip","thumb_mcp","thumb_dip"]
LIM = np.deg2rad(np.array([(-65,35),(-25,30),(-27,27),(-30,30),(-27,27),(-45,33),(-25,100),(-25,100),(-25,100),(-25,100),
                (-18,55),(-15,107),(-15,107),(-15,107),(-15,107),(-25,100),(-15,107)], float))
# sim PD (orca_v2.py) lag Kd/Kp
KP = {"wrist":12.0,"abd":9.7,"mcp":11.8,"pip":14.4,"thumb_cmc":14.1,"thumb_abd":15.0,"thumb_mcp":3.5,"thumb_dip":3.5}
KD = {"wrist":0.85,"abd":0.74,"mcp":1.2,"pip":2.05,"thumb_cmc":1.76,"thumb_abd":1.37,"thumb_mcp":0.42,"thumb_dip":0.42}
def grp(n): return n if n in KP else n.split("_")[1]
TAU = np.array([KD[grp(n)]/KP[grp(n)] for n in USD])
J = json.load(open(f"{D}/actuator/actuator_model_orca_v2_right_2026-10-31.json"))["joints"]
def jp(n, k, default=None):
    e = J.get(n) or J.get("index_" + n.split("_")[1])
    return e.get(k, default) if e else default
AMAX = np.array([jp(n, "amax_rad_s2") for n in USD]); DLY = np.array([jp(n, "delay_s") for n in USD])
WN = np.array([jp(n, "wn_rad_s") for n in USD]); ZE = np.array([jp(n, "zeta") for n in USD])
_P = {"wrist":3.0,"thumb_cmc":2.6,"thumb_abd":5.8,"thumb_mcp":2.7,"thumb_dip":3.3}
PLAY = np.deg2rad([_P[n] if n in _P else {"abd":2.4,"mcp":2.7,"pip":3.8}[n.split("_")[-1]] for n in USD])
_S = {"wrist":.91,"thumb_cmc":.91,"thumb_abd":.80,"thumb_mcp":.91,"thumb_dip":.80}
SLOPE = np.array([_S[n] if n in _S else {"abd":.85,"mcp":.91,"pip":.70}[n.split("_")[-1]] for n in USD])

def load(f):
    a = np.genfromtxt(f, delimiter=",", names=True)
    t = a["time"]; obs = np.stack([a[f"obs_{i}"] for i in range(17)], 1)
    q = LIM[:, 0] + (obs + 1) / 2 * (LIM[:, 1] - LIM[:, 0])
    cmd = np.stack([a[f"cmd_{n}"] for n in USD], 1)
    return t, q, cmd

def simulate(t, q0, cmd, model, extra_delay=0.005):
    dt = 1 / 600; T = np.arange(t[0], t[-1], dt)
    k = np.clip(np.searchsorted(t, T, side="right") - 1, 0, len(t) - 1)
    out = np.zeros((len(T), 17)); x = q0.copy(); v = np.zeros(17); prof = q0.copy(); pv = np.zeros(17); jt = q0.copy(); qd = np.zeros(17)
    for i, tt in enumerate(T):
        if model == "sim":
            g = cmd[k[i]]
            x = x + (g - x) * (1 - np.exp(-dt / TAU))
        else:
            # per joint delayed command
            kd = np.clip(np.searchsorted(t, tt - DLY - extra_delay, side="right") - 1, 0, len(t) - 1)
            g = cmd[kd, np.arange(17)]
            err = g - prof
            vdes = np.sign(err) * np.sqrt(2 * AMAX * np.abs(err))
            pv = pv + np.clip(vdes - pv, -AMAX * dt, AMAX * dt)
            step = pv * dt; arrive = np.abs(step) >= np.abs(err)
            prof = np.where(arrive, g, prof + step); pv = np.where(arrive, 0, pv)
            tgt = prof
            if model == "servo":
                tgt = q0 + SLOPE * (prof - q0)
                jt = np.clip(jt, tgt - PLAY / 2, tgt + PLAY / 2); tgt = jt
            if model in ("servo-play", "servo-play-soft"):
                jt = np.clip(jt, tgt - PLAY / 2, tgt + PLAY / 2); tgt = jt
            if model.endswith("-soft"):  # the task's own PD lag Kd / Kp behind the profile
                x = x + (tgt - x) * (1 - np.exp(-dt / TAU))
            else:  # stiff loop as its equivalent first-order lag 2 zeta / wn (exact discretization)
                x = x + (tgt - x) * (1 - np.exp(-dt / (2 * np.maximum(ZE, 0.5) / WN)))
        out[i] = x
    return np.stack([np.interp(t, T, out[:, j]) for j in range(17)], 1)

models = ("sim", "servo-play", "servo-play-soft")
sq = {m: np.zeros(17) for m in models}; n = 0
for name in sys.argv[2:]:
    t, q, cmd = load(f"{D}/{name}.csv")
    for m in models:
        sq[m] += ((simulate(t, q[0], cmd, m) - q) ** 2).sum(0)
    n += len(t)
r = {m: np.degrees(np.sqrt(sq[m] / n)) for m in models}
print(f"pooled over {sys.argv[2:]}: RMSE deg per joint")
print(f"{'joint':11s} " + " ".join(f"{m:>12s}" for m in models) + "   best")
for j, nm in enumerate(USD):
    best = min(models, key=lambda m: r[m][j])
    print(f"{nm:11s} " + " ".join(f"{r[m][j]:12.1f}" for m in models) + f"   {best}")
print(f"{'mean':11s} " + " ".join(f"{r[m].mean():12.2f}" for m in models))
