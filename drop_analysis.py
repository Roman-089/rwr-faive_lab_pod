"""Drop analysis on 512-env LOWFRIC npz (coordinator ~01:20Z Nov 1).

For every drop event (done=1; the eval has no time-outs, so every reset is a drop) look at the 1 s before it:
cube xy bearing/distance from the user's circle centre, tilt, |wrist speed| (only the 4 envs whose links are logged),
|cube w_z| vs |command|. Compare with the same quantities over all alive steps, and give P(drop within 1 s | state)
by bin, which is the cleaner causal view. Usage: python rwr_drop_analysis.py NAME:NPZ [NAME:NPZ ...]
"""
import importlib.util
import os
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("cm", os.path.expanduser("~/rwr_faive_lab/scripts/eval/cube_motion.py"))
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)
CX, CY, R = -0.0116, 0.0136, 0.0275
W = 20  # 1 s at 20 Hz


def analyse(name, path):
    d = np.load(path, allow_pickle=True)
    T, N = d["done"].shape
    dt = float(d["dt"])
    drop = d["done"] > 0
    alive = ~drop  # the done step itself already shows the respawned cube
    x, y = d["obj_pos"][..., 0] - CX, d["obj_pos"][..., 1] - CY
    dist = np.hypot(x, y) * 1000  # mm
    bear = np.degrees(np.arctan2(y, x))  # 0 = +x, -90 = -y
    up = cm._world_z_in_body(d["obj_quat"])
    tilt = np.degrees(np.arccos(np.clip(np.abs(up).max(-1), 0, 1)))
    wz = np.abs(d["obj_angvel"][..., 2])
    cmd = np.abs(d["cmd_angvel"][..., 2])
    ratio = wz / np.maximum(cmd, 1e-6)
    # wrist angle for the logged envs (same definition as cube_motion)
    names = list(d["link_names"])
    q = d["link_quat"]
    nl = q.shape[1]
    rel = cm._qmul(cm._qconj(q[:, :, names.index("right_forearm")]), q[:, :, names.index("right_carpals")])
    rel = rel * np.sign(rel[..., :1])
    drel = cm._qmul(cm._qconj(rel[:1]), rel)
    drel = drel * np.sign(drel[..., :1])
    ax = np.abs(drel[..., 1:]).reshape(-1, 3).sum(0).argmax()
    wrist = 2 * np.arctan2(drel[..., 1 + ax], drel[..., 0])
    wspd = np.full((T, nl), np.nan)
    wspd[1:] = np.abs(np.diff(wrist, axis=0)) / dt

    # episode age: steps since start or since the last drop
    age = np.zeros((T, N), int)
    for t in range(1, T):
        age[t] = np.where(drop[t - 1], 0, age[t - 1] + 1)

    ev = np.argwhere(drop)  # (t, env)
    ok = [(t, e) for t, e in ev if t >= W and age[t - 1, e] >= W - 1]  # full 1-s window inside one episode
    early = np.mean([age[t - 1, e] < 2 / dt for t, e in ev]) if len(ev) else np.nan

    def win(a, t, e):
        return a[t - W:t, e]

    out = [f"## {name}  ({os.path.basename(path)}; {N} envs x {T} steps, {len(ev)} drops, {len(ok)} with a clean 1-s window)", ""]
    out.append(f"- drops within the first 2 s of an episode: {100 * early:.0f} % "
               f"(an alive step is in its first 2 s {100 * np.mean(age[alive] < 2 / dt):.1f} % of the time)")
    A = alive.copy()
    A[:W] = False  # skip the start transient for the baseline too
    rows = []

    def row(label, pre, base, fmt="{:.1f}"):
        rows.append(f"| {label} | {fmt.format(np.nanmean(pre))} | {fmt.format(np.nanpercentile(pre, 90))} | "
                    f"{fmt.format(np.nanmean(base))} | {fmt.format(np.nanpercentile(base, 90))} |")

    pre_dist = np.array([win(dist, t, e).mean() for t, e in ok])
    pre_last = np.array([dist[t - 1, e] for t, e in ok])
    pre_out = np.array([(win(dist, t, e) > R * 1000).mean() for t, e in ok])
    pre_tilt = np.array([win(tilt, t, e).mean() for t, e in ok])
    pre_tmax = np.array([win(tilt, t, e).max() for t, e in ok])
    pre_rat = np.array([win(ratio, t, e).mean() for t, e in ok])
    row("distance from centre, 1-s mean (mm)", pre_dist, dist[A])
    row("distance at the last step before the drop (mm)", pre_last, dist[A])
    row("share of the window outside the circle (%)", 100 * pre_out, 100 * (dist[A] > R * 1000))
    row("tilt, 1-s mean (deg)", pre_tilt, tilt[A])
    row("tilt, 1-s max (deg)", pre_tmax, tilt[A])
    row("cube |w_z| / |command|, 1-s mean", pre_rat, ratio[A], "{:.2f}")
    wl = [(t, e) for t, e in ok if e < nl]
    if wl:
        row(f"|wrist speed|, 1-s mean (rad/s; logged envs only, n={len(wl)})",
            np.array([np.nanmean(wspd[t - W:t, e]) for t, e in wl]), wspd[1:][A[1:, :nl]], "{:.3f}")
    else:
        rows.append(f"| \\|wrist speed\\| | n/a: no drop in the {nl} link-logged envs | | {np.nanmean(wspd[1:][A[1:, :nl]]):.3f} | |")
    out += ["", "| quantity | 1 s before drops: mean | p90 | all alive steps: mean | p90 |", "|---|---|---|---|---|"] + rows

    # bearing of the cube at the last step before each drop vs all alive steps, 8 sectors of 45 deg
    sect = lambda b: ((b + 180 + 22.5) // 45).astype(int) % 8  # noqa: E731
    labels = ["-x", "-x-y", "-y", "+x-y", "+x", "+x+y", "+y", "-x+y"]
    pb = np.bincount(sect(np.array([bear[t - 1, e] for t, e in ok])), minlength=8) / max(len(ok), 1)
    ab = np.bincount(sect(bear[A]), minlength=8) / A.sum()
    out += ["", "Bearing of the cube from the circle centre (last step before the drop vs all alive steps):", "",
            "| sector | " + " | ".join(labels) + " |", "|---" * 9 + "|",
            "| before drops % | " + " | ".join(f"{100 * v:.0f}" for v in pb) + " |",
            "| all alive % | " + " | ".join(f"{100 * v:.0f}" for v in ab) + " |"]

    # spawn-time vs mid-episode drops
    first = [(t, e) for t, e in ev if age[t - 1, e] < 2 / dt]
    out += ["", f"Early drops (episode age < 2 s): {len(first)} of {len(ev)}; "
            f"{sum(1 for t, e in first if not drop[:t, e].any())} in the very first episode (t < 2 s after the eval start), "
            f"{sum(1 for t, e in first if drop[:t, e].any())} right after a respawn."]
    fd = np.where(drop.any(0), drop.argmax(0), T)
    hist = np.bincount(fd[fd < 2 / dt] // 5, minlength=8)
    late = drop[int(2 / dt):].any(0)
    out += [f"Envs with any drop: {(fd < T).sum()} (drop-free {100 * np.mean(fd == T):.1f} %); first drop within 2 s of the eval start: "
            f"{(fd < 2 / dt).sum()} envs, by 0.25-s bin {hist.tolist()}. Envs with a drop after the first 2 s: {late.sum()} "
            f"(drop-free over t >= 2 s: {100 * (1 - late.mean()):.1f} %)."]

    # departure = start of the final stretch in which the cube is off the hand (dist > 35 mm or tilt > 25 deg)
    # until the drop is registered; lead-up = the 1 s before departure (mid-episode drops only)
    off = (dist > 35) | (tilt > 25)
    lead = []
    for t, e in ev:
        k = t - 1
        while k > 0 and off[k, e] and age[k, e] > 0:
            k -= 1
        k += 1
        if age[k - 1, e] >= W + 2 / dt - 1 and k - W >= 0:
            lead.append((t, e, k))
    if lead:
        fall = np.array([(t - k) * dt for t, e, k in lead])
        out += ["", f"Lead-up before the cube leaves the hand (mid-episode drops with a clean window, n={len(lead)}; "
                f"departure -> registered drop takes {np.median(fall):.2f} s median):", "",
                "| quantity | 1 s before departure: mean | p90 | all alive steps: mean | p90 |", "|---|---|---|---|---|"]
        rows = []
        row("distance from centre, 1-s mean (mm)", np.array([dist[k - W:k, e].mean() for t, e, k in lead]), dist[A])
        row("distance at departure - 1 step (mm)", np.array([dist[k - 1, e] for t, e, k in lead]), dist[A])
        row("share of the window outside the circle (%)", np.array([100 * (dist[k - W:k, e] > R * 1000).mean() for t, e, k in lead]),
            100 * (dist[A] > R * 1000))
        row("tilt, 1-s mean (deg)", np.array([tilt[k - W:k, e].mean() for t, e, k in lead]), tilt[A])
        row("tilt, 1-s max (deg)", np.array([tilt[k - W:k, e].max() for t, e, k in lead]), tilt[A])
        row("cube |w_z| / |command|, 1-s mean", np.array([ratio[k - W:k, e].mean() for t, e, k in lead]), ratio[A], "{:.2f}")
        wl = [(t, e, k) for t, e, k in lead if e < nl]
        if wl:
            row(f"|wrist speed|, 1-s mean (rad/s; logged envs only, n={len(wl)})",
                np.array([np.nanmean(wspd[k - W:k, e]) for t, e, k in wl]), wspd[1:][A[1:, :nl]], "{:.3f}")
        out += rows
        xb = np.array([bear[k, e] for t, e, k in lead])
        pb = np.bincount(sect(xb), minlength=8) / len(lead)
        out += ["", "Exit bearing (cube position at departure) vs all alive steps:", "",
                "| sector | " + " | ".join(labels) + " |", "|---" * 9 + "|",
                "| exits % | " + " | ".join(f"{100 * v:.0f}" for v in pb) + " |",
                "| all alive % | " + " | ".join(f"{100 * v:.0f}" for v in ab) + " |"]
        # mean profile over the 2 s before departure
        prof = [-40, -30, -20, -15, -10, -5, -1]
        pr = [(o, np.mean([dist[k + o, e] for t, e, k in lead if k + o >= 0]), np.mean([tilt[k + o, e] for t, e, k in lead if k + o >= 0]))
              for o in prof]
        out += ["", "Mean profile before departure (s): " + ", ".join(f"{o * dt:+.2f}: {a:.0f} mm / {b:.0f} deg" for o, a, b in pr)]

    # hazard: P(drop within the next 1 s | state now), per 1000 alive steps, by bins
    fut = np.zeros_like(drop)
    for k in range(1, W + 1):
        fut[:-k] |= drop[k:]
    fut = fut & A
    nxt_ok = A.copy()
    nxt_ok[-W:] = False

    def hazard(v, edges, label, unit):
        b = np.digitize(v[nxt_ok], edges)
        f = fut[nxt_ok]
        cells = []
        for i in range(len(edges) + 1):
            m = b == i
            lo = "<" + f"{edges[0]:g}" if i == 0 else (f">={edges[-1]:g}" if i == len(edges) else f"{edges[i - 1]:g}-{edges[i]:g}")
            cells.append((lo, m.mean() * 100, 1000 * f[m].mean() if m.any() else np.nan, f[m].sum()))
        return [f"", f"P(drop within 1 s) by {label} ({unit}); share of alive steps, drops per 1000 steps, drop-steps:", "",
                "| bin | " + " | ".join(c[0] for c in cells) + " |", "|---" * (len(cells) + 1) + "|",
                "| alive % | " + " | ".join(f"{c[1]:.1f}" for c in cells) + " |",
                "| per 1000 | " + " | ".join(f"{c[2]:.2f}" for c in cells) + " |"]

    out += hazard(dist, [10, 20, 27.5, 35], "distance from the circle centre", "mm")
    out += hazard(tilt, [5, 10, 15, 25], "tilt", "deg")
    out += hazard(ratio, [0.5, 0.8, 1.2, 1.5], "|w_z|/|command|", "-")
    yb = d["obj_pos"][..., 1] * 1000 - CY * 1000
    out += hazard(yb, [-20, -10, 0, 10], "cube y offset from the circle centre", "mm")
    return "\n".join(out)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        n, p = a.split(":", 1)
        print(analyse(n, p))
        print()
