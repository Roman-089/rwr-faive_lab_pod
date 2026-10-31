"""Build an interactive top-down view of the hand (HTML) for choosing the region the cube's rotation axis may occupy.

The cube turns about world z, so "the rotation axis stays in a circle on the palm" means the cube centre's x-y stays in
that circle. The page draws the hand from above (world x right, y up, env frame, mm) in a few frames of an eval rollout,
the cube-centre positions of that rollout as a density cloud, the commanded cube position, and lets the user click /
drag a circle; it reports the share of the rollout's cube positions inside and a config snippet (metres).

Usage (CPU): python scripts/eval/make_palm_region_html.py <rollout .npz> <out.html> [--frames 200 600 1000] [--title ...]
(rollout = scripts/eval/eval_policy.py output: mesh_verts_* in link frames, link_pos / link_quat, obj_pos, done, cmd_pos)
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

PALM = {"right_forearm", "right_tower", "right_carpals"}  # TPU
PLA = {"right_index_ap", "right_middle_ap", "right_pinky_ap", "right_ring_ap", "right_thumb_tp", "right_thumb_ap"}


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def link_polygons(d, t, env=0):
    """Top-view outline (2D convex hull, mm) and mean height of every link at step t."""
    out = []
    for i, name in enumerate(d["link_names"]):
        key = f"mesh_verts_{i}"
        if key not in d:
            continue
        v = d[key] @ quat_to_mat(d["link_quat"][t, env, i]).T + d["link_pos"][t, env, i]
        xy = v[:, :2] * 1000
        hull = ConvexHull(xy)
        kind = "palm" if name in PALM else "pla" if name in PLA else "skin"
        out.append({"name": str(name).removeprefix("right_"), "kind": kind, "z": float(v[:, 2].mean() * 1000),
                    "pts": np.round(xy[hull.vertices], 1).tolist()})
    return sorted(out, key=lambda p: p["z"])  # draw low links first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, nargs="+", default=[200, 600, 1000])
    ap.add_argument("--points", type=int, default=6000, help="cube-centre samples embedded in the page")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    d = dict(np.load(args.npz, allow_pickle=True))
    alive = ~(d["done"] > 0)
    cube = d["obj_pos"][..., :2][alive] * 1000  # mm, env frame
    rng = np.random.default_rng(0)
    pts = cube[rng.choice(len(cube), min(args.points, len(cube)), replace=False)]
    data = {
        "title": args.title or Path(args.npz).parent.parent.name,
        "source": str(args.npz),
        "cube_size_mm": float(d["object_size"]) * 1000,
        "cmd_mm": np.round(d["cmd_pos"][0, 0, :2] * 1000, 2).tolist(),
        "dt": float(d["dt"]),
        "frames": [{"t": t, "time_s": round(t * float(d["dt"]), 2), "links": link_polygons(d, t),
                    "cube": np.round(d["obj_pos"][t, 0, :2] * 1000, 1).tolist()} for t in args.frames],
        "points": np.round(pts, 1).tolist(),
        "cube_median_mm": [round(float(v), 1) for v in np.median(cube, 0)],
        "cube_p90_radius_mm": round(float(np.percentile(np.linalg.norm(cube - np.median(cube, 0), axis=1), 90)), 1),
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(html) / 1e3:.0f} kB): {len(data['frames'])} frames, {len(pts)} cube samples, "
          f"median cube xy {data['cube_median_mm']} mm, p90 radius {data['cube_p90_radius_mm']} mm")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Palm Axis Region</title>
<style>
:root {
  --bg: #f7f7f5; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --line: #d9d9d4;
  --palm: #c9b79c; --skin: #e8a39a; --pla: #9fb4c7; --cube: #2f6fbd; --circle: #d9480f; --cmd: #2b8a3e;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161615; --panel: #1f1f1d; --text: #ecebe6; --muted: #9a9a93; --line: #34342f;
    --palm: #8a7a63; --skin: #b06a62; --pla: #5f7487; --cube: #6ea8ff; --circle: #ff8a4c; --cmd: #5fd27a;
  }
}
:root[data-theme="dark"] {
  --bg: #161615; --panel: #1f1f1d; --text: #ecebe6; --muted: #9a9a93; --line: #34342f;
  --palm: #8a7a63; --skin: #b06a62; --pla: #5f7487; --cube: #6ea8ff; --circle: #ff8a4c; --cmd: #5fd27a;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.45 system-ui, -apple-system, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 16px; }
h1 { font-size: 20px; margin: 4px 0 2px; }
.sub { color: var(--muted); margin: 0 0 14px; font-size: 13px; }
.grid { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 16px; }
@media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
svg { width: 100%; height: auto; display: block; touch-action: none; cursor: crosshair; }
label { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin: 6px 0; font-size: 14px; }
input[type=number] { width: 90px; padding: 4px 6px; background: var(--bg); color: var(--text); border: 1px solid var(--line); border-radius: 6px; }
input[type=range] { width: 100%; }
select, button { padding: 5px 8px; background: var(--bg); color: var(--text); border: 1px solid var(--line); border-radius: 6px; font: inherit; }
button { cursor: pointer; }
.stat { font-size: 26px; font-weight: 600; }
pre { background: var(--bg); border: 1px solid var(--line); border-radius: 8px; padding: 8px; overflow-x: auto; font-size: 12px; white-space: pre-wrap; }
.legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: var(--muted); margin-top: 8px; }
.sw { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 4px; vertical-align: -1px; }
.hint { font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<main>
<h1>Palm axis region</h1>
<p class="sub" id="sub"></p>
<div class="grid">
  <div class="card">
    <svg id="view" role="img" aria-label="Top view of the hand with the selected circle"></svg>
    <div class="legend">
      <span><i class="sw" style="background:var(--palm)"></i>palm (TPU)</span>
      <span><i class="sw" style="background:var(--skin)"></i>finger links with silicone skin</span>
      <span><i class="sw" style="background:var(--pla)"></i>bare PLA links</span>
      <span><i class="sw" style="background:var(--cube)"></i>cube centre in the rollout</span>
      <span><i class="sw" style="background:var(--cmd)"></i>commanded cube position</span>
      <span><i class="sw" style="background:var(--circle)"></i>allowed axis region</span>
    </div>
    <p class="hint">Press to place the centre, drag to set the radius. View from above (world z): x right, y up, millimetres, env frame of the simulation.</p>
  </div>
  <div class="card">
    <label>Hand pose <select id="frame"></select></label>
    <label>Centre x [mm] <input id="cx" type="number" step="0.5"></label>
    <label>Centre y [mm] <input id="cy" type="number" step="0.5"></label>
    <label>Radius [mm] <input id="r" type="number" step="0.5" min="1"></label>
    <input id="rs" type="range" min="2" max="60" step="0.5" aria-label="Radius">
    <label><span>Show cube cloud</span><input id="showpts" type="checkbox" checked></label>
    <div style="margin:10px 0 4px">Rollout cube positions inside</div>
    <div class="stat" id="inside">–</div>
    <div class="hint" id="insideNote"></div>
    <div style="margin:12px 0 4px">Config (metres, env frame)</div>
    <pre id="snippet"></pre>
    <button id="copy">Copy</button>
    <button id="reset">Reset to rollout</button>
    <button id="theme" title="Toggle light / dark">Theme</button>
  </div>
</div>
</main>
<script>
const D = __DATA__;
const svg = document.getElementById("view");
const NS = "http://www.w3.org/2000/svg";
const st = { frame: 0, cx: D.cube_median_mm[0], cy: D.cube_median_mm[1], r: Math.max(5, D.cube_p90_radius_mm), drag: false };
document.getElementById("sub").textContent =
  `Policy rollout: ${D.title}. The cube spins about world z, so its rotation axis is the vertical line through the cube centre; the circle bounds where that centre may be.`;

// view box from all frames' link outlines and the cube cloud
let xs = [], ys = [];
D.frames.forEach(f => f.links.forEach(l => { if (l.name !== "forearm" && l.name !== "tower") l.pts.forEach(p => { xs.push(p[0]); ys.push(p[1]); }); }));
D.points.forEach(p => { xs.push(p[0]); ys.push(p[1]); });
const pad = 10, x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
svg.setAttribute("viewBox", `${x0} ${-y1} ${x1 - x0} ${y1 - y0}`);  // y up: draw at -y
const el = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };

const sel = document.getElementById("frame");
D.frames.forEach((f, i) => { const o = document.createElement("option"); o.value = i; o.textContent = `t = ${f.time_s} s (step ${f.t})`; sel.appendChild(o); });
sel.value = 1 < D.frames.length ? 1 : 0; st.frame = +sel.value;

function draw() {
  svg.innerHTML = "";
  const g = el("g", {});
  for (let gx = Math.ceil(x0 / 10) * 10; gx <= x1; gx += 10) el("line", { x1: gx, y1: -y1, x2: gx, y2: -y0, stroke: "var(--line)", "stroke-width": gx === 0 ? 0.6 : 0.25 }, g);
  for (let gy = Math.ceil(y0 / 10) * 10; gy <= y1; gy += 10) el("line", { x1: x0, y1: -gy, x2: x1, y2: -gy, stroke: "var(--line)", "stroke-width": gy === 0 ? 0.6 : 0.25 }, g);
  const f = D.frames[st.frame];
  for (const l of f.links) {
    el("polygon", { points: l.pts.map(p => `${p[0]},${-p[1]}`).join(" "), fill: `var(--${l.kind})`, "fill-opacity": 0.78, stroke: "var(--text)", "stroke-opacity": 0.35, "stroke-width": 0.3 }).appendChild(
      Object.assign(document.createElementNS(NS, "title"), { textContent: `${l.name} (${l.kind}), mean height ${l.z.toFixed(0)} mm` }));
  }
  if (document.getElementById("showpts").checked) {
    const pg = el("g", { fill: "var(--cube)", "fill-opacity": 0.18 });
    for (const p of D.points) el("circle", { cx: p[0], cy: -p[1], r: 0.45 }, pg);
  }
  const h = D.cube_size_mm / 2, c = f.cube;
  el("rect", { x: c[0] - h, y: -c[1] - h, width: 2 * h, height: 2 * h, fill: "none", stroke: "var(--cube)", "stroke-width": 0.6, "stroke-dasharray": "2 1.5" });
  el("circle", { cx: D.cmd_mm[0], cy: -D.cmd_mm[1], r: 1.4, fill: "var(--cmd)" });
  el("circle", { cx: st.cx, cy: -st.cy, r: st.r, fill: "var(--circle)", "fill-opacity": 0.12, stroke: "var(--circle)", "stroke-width": 0.8 });
  el("circle", { cx: st.cx, cy: -st.cy, r: 0.9, fill: "var(--circle)" });
  const t = el("text", { x: x0 + 3, y: -y1 + 6, "font-size": 4.5, fill: "var(--muted)" }); t.textContent = "wrist ↑   fingertips ↓   (10 mm grid)";
  update();
}

function update() {
  const inside = D.points.filter(p => (p[0] - st.cx) ** 2 + (p[1] - st.cy) ** 2 <= st.r ** 2).length;
  document.getElementById("inside").textContent = `${(100 * inside / D.points.length).toFixed(1)} %`;
  document.getElementById("insideNote").textContent = `${inside} of ${D.points.length} sampled cube-centre positions (all envs, alive steps)`;
  for (const k of ["cx", "cy", "r"]) { const i = document.getElementById(k); if (document.activeElement !== i) i.value = st[k].toFixed(1); }
  document.getElementById("rs").value = st.r;
  const m = v => (v / 1000).toFixed(4);
  document.getElementById("snippet").textContent =
`# Cube rotation-axis region: the cube centre's x-y (env frame,
# world z is the rotation axis) must stay within this circle.
CUBE_AXIS_REGION = dict(center=(${m(st.cx)}, ${m(st.cy)}), radius=${m(st.r)})  # m
# e.g. terminate / penalize when
# norm(obj_pos[:, :2] - center) > radius`;
}

function toLocal(ev) {
  const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
  const p = pt.matrixTransform(svg.getScreenCTM().inverse()); return [p.x, -p.y];
}
svg.addEventListener("pointerdown", ev => { const [x, y] = toLocal(ev); st.cx = +x.toFixed(1); st.cy = +y.toFixed(1); st.drag = true; svg.setPointerCapture(ev.pointerId); draw(); });
svg.addEventListener("pointermove", ev => { if (!st.drag) return; const [x, y] = toLocal(ev); st.r = Math.max(1, +Math.hypot(x - st.cx, y - st.cy).toFixed(1)); draw(); });
svg.addEventListener("pointerup", () => { st.drag = false; });
for (const k of ["cx", "cy", "r"]) document.getElementById(k).addEventListener("input", e => { const v = parseFloat(e.target.value); if (!isNaN(v)) { st[k] = k === "r" ? Math.max(1, v) : v; draw(); } });
document.getElementById("rs").addEventListener("input", e => { st.r = parseFloat(e.target.value); draw(); });
sel.addEventListener("change", e => { st.frame = +e.target.value; draw(); });
document.getElementById("showpts").addEventListener("change", draw);
document.getElementById("reset").addEventListener("click", () => { st.cx = D.cube_median_mm[0]; st.cy = D.cube_median_mm[1]; st.r = Math.max(5, D.cube_p90_radius_mm); draw(); });
document.getElementById("copy").addEventListener("click", () => { navigator.clipboard?.writeText(document.getElementById("snippet").textContent); });
document.getElementById("theme").addEventListener("click", () => {
  const r = document.documentElement, dark = r.dataset.theme ? r.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  r.dataset.theme = dark ? "light" : "dark"; draw();
});
draw();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
