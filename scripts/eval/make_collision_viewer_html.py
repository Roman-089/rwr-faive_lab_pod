"""Interactive 3D view (HTML, three.js) of the collision bodies Isaac uses for the ORCA v2 hand, animated by an eval rollout.

The hand collides through 53 convex hulls (physics:approximation = convexHull, 64 vertices each, in link frames), taken
from scripts/sim2sim/orca_v2_right_usd.json (extract_usd_model.py, the same hulls PhysX builds). The page places them
with the link poses of a rollout (scripts/eval/eval_policy.py npz: link_pos / link_quat of the first video envs,
obj_pos / obj_quat), colours them by the real hand's material (silicone skin, TPU palm, bare PLA), adds the 45 mm cube
and, optionally, the cube rotation-axis region chosen by the user.

Usage (CPU): python scripts/eval/make_collision_viewer_html.py <rollout.npz> <out.html> [--env 0] [--stride 2]
             [--region CX CY R] [--title ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[2]
HULLS = REPO / "scripts/sim2sim/orca_v2_right_usd.json"
PALM = {"right_forearm", "right_tower", "right_carpals"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--envs", type=int, nargs="+", default=None,
                    help="recorded envs to offer (default: all the npz has link poses for); a picker switches between them")
    ap.add_argument("--pick", type=int, default=0,
                    help="instead of --envs: N recorded envs spread over the per-env RPM distribution (slowest .. fastest)")
    ap.add_argument("--stride", type=int, default=2, help="keep every n-th control step (20 Hz)")
    ap.add_argument("--region", type=float, nargs=3, default=None, metavar=("CX", "CY", "R"))
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    names = [str(n) for n in d["link_names"]]
    hulls_json = json.load(open(HULLS))["collision_hulls"]
    hulls = []
    for li, link in enumerate(names):
        for h in hulls_json.get(link, []):
            v = np.asarray(h["vertices"], float)
            tri = ConvexHull(v).simplices
            # outward winding for three.js lighting
            c = v.mean(0)
            for t in tri:
                if np.dot(np.cross(v[t[1]] - v[t[0]], v[t[2]] - v[t[0]]), v[t[0]] - c) < 0:
                    t[1], t[2] = t[2], t[1]
            kind = "skin" if h["name"].endswith("_Skin") else "palm" if link in PALM else "pla"
            hulls.append({"link": li, "name": h["name"], "kind": kind,
                          "v": np.round(v * 1000, 2).ravel().tolist(), "f": tri.ravel().tolist()})
    s = slice(None, None, args.stride)
    n_rec = d["link_pos"].shape[1]
    envs = args.envs if args.envs is not None else list(range(n_rec))
    # per-env speed over the whole rollout (all envs), so the picker can say where each recorded env stands
    alive_all = ~(d["done"] > 0)
    w = np.where(alive_all, d["obj_angvel"][..., 2] * np.sign(d["cmd_angvel"][0, :, 2]), 0.0)
    rpm = w.sum(0) / np.maximum(alive_all.sum(0), 1) * 60 / (2 * np.pi)
    mean_rpm = float(rpm.mean())
    if args.pick:  # N envs at evenly spaced quantiles of the recorded envs' RPM
        rec = np.argsort(rpm[:n_rec])
        envs = sorted({int(rec[round(q * (n_rec - 1))]) for q in np.linspace(0, 1, args.pick)})
    env_data = [{
        "id": e, "rpm": round(float(rpm[e]), 2), "rank": int((rpm > rpm[e]).sum()) + 1,
        "link_pos": np.round(d["link_pos"][s, e] * 1000, 2).tolist(),
        "link_quat": np.round(d["link_quat"][s, e], 5).tolist(),
        "obj_pos": np.round(d["obj_pos"][s, e] * 1000, 2).tolist(),
        "obj_quat": np.round(d["obj_quat"][s, e], 5).tolist(),
        "alive": (~(d["done"][s, e] > 0)).astype(int).tolist(),
    } for e in envs]
    default = int(np.argmin([abs(x["rpm"] - mean_rpm) for x in env_data]))  # the recorded env closest to the mean
    data = {
        "title": args.title or Path(args.npz).name,
        "links": [n.removeprefix("right_") for n in names],
        "hulls": hulls,
        "dt": float(d["dt"]) * args.stride,
        "envs": env_data, "default_env": default, "n_envs": int(len(rpm)), "mean_rpm": round(mean_rpm, 2),
        "cube_mm": float(d["object_size"]) * 1000,
        "region": None if args.region is None else [args.region[0] * 1000, args.region[1] * 1000, args.region[2] * 1000],
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    Path(args.out).write_text(html)
    n_by = {k: sum(h["kind"] == k for h in hulls) for k in ("skin", "palm", "pla")}
    print(f"wrote {args.out} ({len(html) / 1e3:.0f} kB): {len(hulls)} hulls {n_by}, {len(env_data[0]['link_pos'])} frames, envs {envs} (default {envs[default]})")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hand Collision Bodies</title>
<style>
:root { --bg: #f4f4f1; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --line: #d9d9d4;
  --skin: #e0857a; --palm: #c4ad8a; --pla: #7f9fbf; --cube: #3b7dd8; --region: #d9480f; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #151514; --panel: #1f1f1d; --text: #ecebe6; --muted: #9a9a93; --line: #34342f;
  --skin: #c96f65; --palm: #a08b6b; --pla: #6a8aaa; --cube: #6ea8ff; --region: #ff8a4c; } }
:root[data-theme="dark"] { --bg: #151514; --panel: #1f1f1d; --text: #ecebe6; --muted: #9a9a93; --line: #34342f;
  --skin: #c96f65; --palm: #a08b6b; --pla: #6a8aaa; --cube: #6ea8ff; --region: #ff8a4c; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 16px; }
h1 { font-size: 20px; margin: 4px 0 2px; } .sub { color: var(--muted); margin: 0 0 12px; font-size: 13px; }
.grid { display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 14px; }
@media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
#view { width: 100%; aspect-ratio: 4 / 3; border-radius: 8px; overflow: hidden; position: relative; }
#tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 2px 6px; font-size: 12px; display: none; }
label { display: flex; align-items: center; gap: 8px; margin: 5px 0; }
.sw { display: inline-block; width: 11px; height: 11px; border-radius: 3px; }
input[type=range] { width: 100%; } button { padding: 5px 10px; background: var(--bg); color: var(--text); border: 1px solid var(--line); border-radius: 6px; font: inherit; cursor: pointer; }
.hint { color: var(--muted); font-size: 12px; }
</style>
<script type="importmap">{"imports": {"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head>
<body>
<main>
<h1>Hand collision bodies</h1>
<p class="sub" id="sub"></p>
<div class="grid">
  <div class="card"><div id="view"><div id="tip"></div></div>
    <p class="hint">Drag to orbit, scroll to zoom, right-drag to pan. Hover a hull for its name. World frame of the env: z up (the cube's rotation axis), millimetres.</p></div>
  <div class="card">
    <label><input type="checkbox" id="k_skin" checked><span class="sw" style="background:var(--skin)"></span>silicone skin hulls <span id="n_skin" class="hint"></span></label>
    <label><input type="checkbox" id="k_palm" checked><span class="sw" style="background:var(--palm)"></span>TPU palm hulls <span id="n_palm" class="hint"></span></label>
    <label><input type="checkbox" id="k_pla" checked><span class="sw" style="background:var(--pla)"></span>bare PLA hulls <span id="n_pla" class="hint"></span></label>
    <label><input type="checkbox" id="k_cube" checked><span class="sw" style="background:var(--cube)"></span>cube (45 mm)</label>
    <label id="regrow"><input type="checkbox" id="k_region" checked><span class="sw" style="background:var(--region)"></span>rotation-axis region</label>
    <label><input type="checkbox" id="k_wire">wireframe</label>
    <label><input type="checkbox" id="k_edges" checked>hull edges</label>
    <div style="margin-top:10px">Time <span id="tlabel"></span></div>
    <input type="range" id="t" min="0" step="1" value="0">
    <div style="display:flex; gap:6px; margin-top:6px"><button id="play">Play</button><button id="top">Top view</button><button id="side">Side view</button><button id="theme">Theme</button></div>
    <label style="margin-top:10px">Env <select id="env" style="flex:1"></select></label>
    <p class="hint" id="info"></p>
  </div>
</div>
</main>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
document.getElementById("sub").textContent = `The 53 convex hulls PhysX uses for the ORCA v2 hand in training (64 vertices each), posed by the rollout ${D.title}; coloured by the real hand's material.`;
for (const k of ["skin", "palm", "pla"]) document.getElementById("n_" + k).textContent = `(${D.hulls.filter(h => h.kind === k).length})`;
if (!D.region) document.getElementById("regrow").style.display = "none";

const host = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 4 / 3, 1, 5000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(200, -150, 400); scene.add(sun);

const linkGroups = D.links.map(() => { const g = new THREE.Group(); scene.add(g); return g; });
const mats = {}, hullMeshes = [];
function makeMats() {
  for (const k of ["skin", "palm", "pla"]) {
    mats[k] = mats[k] || new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0.0, transparent: true, opacity: 0.92 });
    mats[k].color.set(css("--" + k));
  }
}
makeMats();
const edgeMat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.35 });
for (const h of D.hulls) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(h.v, 3));
  g.setIndex(h.f); g.computeVertexNormals();
  const m = new THREE.Mesh(g, mats[h.kind]); m.userData = h;
  const e = new THREE.LineSegments(new THREE.EdgesGeometry(g, 25), edgeMat); m.add(e);
  linkGroups[h.link].add(m); hullMeshes.push(m);
}
const s = D.cube_mm;
const cube = new THREE.Mesh(new THREE.BoxGeometry(s, s, s), new THREE.MeshStandardMaterial({ transparent: true, opacity: 0.55, roughness: 0.5 }));
cube.add(new THREE.LineSegments(new THREE.EdgesGeometry(cube.geometry), new THREE.LineBasicMaterial()));
scene.add(cube);
let region = null;
if (D.region) {
  const [cx, cy, r] = D.region, pts = [];
  for (let i = 0; i <= 96; i++) { const a = i / 96 * 2 * Math.PI; pts.push(new THREE.Vector3(cx + r * Math.cos(a), cy + r * Math.sin(a), 0)); }
  region = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ linewidth: 2 }));
  const axis = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(cx, cy, -60), new THREE.Vector3(cx, cy, 30)]), new THREE.LineDashedMaterial({ dashSize: 3, gapSize: 2 }));
  axis.computeLineDistances(); region.add(axis); scene.add(region);
}
function colors() {
  makeMats(); scene.background = new THREE.Color(css("--bg")); edgeMat.color.set(css("--text"));
  cube.material.color.set(css("--cube")); cube.children[0].material.color.set(css("--cube"));
  if (region) { region.material.color.set(css("--region")); region.children[0].material.color.set(css("--region")); }
}
colors();

let E = D.envs[D.default_env];
const sel = document.getElementById("env");
D.envs.forEach((x, k) => { const o = document.createElement("option"); o.value = k;
  o.textContent = `env ${x.id}: ${x.rpm.toFixed(1)} RPM (rank ${x.rank} of ${D.n_envs}${k === D.default_env ? ", closest to the mean" : ""})`; sel.appendChild(o); });
sel.value = D.default_env;
const T = E.link_pos.length, slider = document.getElementById("t"); slider.max = T - 1;
function setFrame(i) {
  for (let l = 0; l < D.links.length; l++) {
    const p = E.link_pos[i][l], q = E.link_quat[i][l];
    linkGroups[l].position.set(p[0], p[1], p[2]); linkGroups[l].quaternion.set(q[1], q[2], q[3], q[0]);
  }
  const p = E.obj_pos[i], q = E.obj_quat[i];
  cube.position.set(p[0], p[1], p[2]); cube.quaternion.set(q[1], q[2], q[3], q[0]);
  if (region) region.position.z = p[2];
  document.getElementById("tlabel").textContent = `${(i * D.dt).toFixed(2)} s${E.alive[i] ? "" : " (cube dropped, respawn)"}`;
}
// camera framing from the hand's bounding box at frame 0
setFrame(0);
const box = new THREE.Box3(); linkGroups.forEach(g => box.expandByObject(g)); box.expandByObject(cube);
const ctr = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3()).length();
function view(dir) { camera.position.copy(ctr).add(dir.clone().normalize().multiplyScalar(size * 1.6)); controls.target.copy(ctr); controls.update(); }
view(new THREE.Vector3(0.9, -1.0, 0.8));
document.getElementById("top").onclick = () => view(new THREE.Vector3(0.0001, -0.0001, 1));
document.getElementById("side").onclick = () => view(new THREE.Vector3(1, 0, 0.15));

function vis() {
  for (const m of hullMeshes) m.visible = document.getElementById("k_" + m.userData.kind).checked;
  cube.visible = document.getElementById("k_cube").checked;
  if (region) region.visible = document.getElementById("k_region").checked;
  const w = document.getElementById("k_wire").checked; for (const k in mats) mats[k].wireframe = w;
  const e = document.getElementById("k_edges").checked; hullMeshes.forEach(m => m.children[0].visible = e);
}
document.querySelectorAll("input[type=checkbox]").forEach(c => c.addEventListener("change", vis));
let playing = false, last = 0, frame = 0;
slider.oninput = () => { frame = +slider.value; setFrame(frame); };
document.getElementById("play").onclick = e => { playing = !playing; e.target.textContent = playing ? "Pause" : "Play"; };
document.getElementById("theme").onclick = () => {
  const r = document.documentElement, dark = r.dataset.theme ? r.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  r.dataset.theme = dark ? "light" : "dark"; colors();
};
const tip = document.getElementById("tip"), ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
renderer.domElement.addEventListener("pointermove", ev => {
  const r = renderer.domElement.getBoundingClientRect();
  mouse.set((ev.clientX - r.left) / r.width * 2 - 1, -(ev.clientY - r.top) / r.height * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObjects(hullMeshes.filter(m => m.visible), false)[0];
  if (hit) { const h = hit.object.userData; tip.style.display = "block"; tip.style.left = (ev.clientX - r.left + 12) + "px"; tip.style.top = (ev.clientY - r.top + 8) + "px";
    tip.textContent = `${h.name} · link ${D.links[h.link]} · ${h.kind === "skin" ? "silicone skin" : h.kind === "palm" ? "TPU palm" : "bare PLA"}`; }
  else tip.style.display = "none";
});
function resize() { const w = host.clientWidth, h = host.clientHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(host); resize();
sel.onchange = () => { E = D.envs[+sel.value]; setFrame(frame); };
document.getElementById("info").textContent = `${D.hulls.length} hulls, ${T} frames at ${(1 / D.dt).toFixed(0)} Hz, real time. Mean over all ${D.n_envs} envs: ${D.mean_rpm.toFixed(1)} RPM; speed differs a lot between envs (randomised friction / mass).`;
renderer.setAnimationLoop(t => {
  if (playing && t - last > D.dt * 1000) { frame = (frame + 1) % T; slider.value = frame; setFrame(frame); last = t; }
  controls.update(); renderer.render(scene, camera);
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
