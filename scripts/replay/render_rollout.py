"""Render a rollout recorded by scripts/eval/eval_policy.py (or hpc/replay/record_rollout.py) to an MP4 with MuJoCo. Runs on the CPU (login node is fine).

Every hand link and the cube become MuJoCo mocap bodies that are moved to the recorded poses; nothing is
simulated. Arrows on the cube: green = commanded angular velocity, orange = actual angular velocity.

Usage:
    source ~/rwr_faive_lab/activate_rwr.sh
    python scripts/replay/render_rollout.py <rollout.npz> [--envs 0 1 2 3] [--size 480] [--out video.mp4]
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # Mesa EGL works headless on the login node

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

HAND_RGBA = "0.85 0.82 0.78 1"
CUBE_FACES = [  # (axis, sign, rgba): colored plates so the cube's rotation is visible
    (0, 1, "0.90 0.20 0.20 1"), (0, -1, "0.20 0.75 0.85 1"),
    (1, 1, "0.25 0.75 0.30 1"), (1, -1, "0.80 0.30 0.80 1"),
    (2, 1, "0.20 0.35 0.90 1"), (2, -1, "0.95 0.80 0.15 1"),
]


def fmt(a: np.ndarray) -> str:
    return " ".join(f"{x:.6g}" for x in a.ravel())


def build_model(data: dict) -> tuple[mujoco.MjModel, list[int]]:
    """MJCF with one mocap body per meshed link plus the cube; returns the model and the meshed link indices."""
    n_links = len(data["link_names"])
    links = [i for i in range(n_links) if f"mesh_verts_{i}" in data]
    floor_z = float(data["link_pos"][..., 2].min()) - 0.03
    s = float(data["object_size"]) / 2

    assets, bodies = [], []
    for i in links:
        assets.append(f'<mesh name="m{i}" vertex="{fmt(data[f"mesh_verts_{i}"])}" face="{fmt(data[f"mesh_faces_{i}"])}"/>')
        bodies.append(f'<body name="l{i}" mocap="true"><geom type="mesh" mesh="m{i}" rgba="{HAND_RGBA}"/></body>')
    plates = []
    for axis, sign, rgba in CUBE_FACES:
        pos, size = np.zeros(3), np.full(3, 0.75 * s)
        pos[axis], size[axis] = sign * s, 0.001
        plates.append(f'<geom type="box" pos="{fmt(pos)}" size="{fmt(size)}" rgba="{rgba}"/>')
    bodies.append(f'<body name="cube" mocap="true"><geom type="box" size="{s} {s} {s}" rgba="0.95 0.95 0.95 1"/>'
                  + "".join(plates) + "</body>")

    xml = f"""
<mujoco>
  <compiler inertiafromgeom="false"/>
  <option gravity="0 0 0"/>
  <visual><global offwidth="1920" offheight="1920"/><quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
  <default><geom contype="0" conaffinity="0"/></default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.65 0.8" rgb2="0.1 0.12 0.15" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.32 0.35" rgb2="0.24 0.26 0.28" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="20 20" reflectance="0.1"/>
    {"".join(assets)}
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 1.0" dir="-0.3 0.3 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <geom type="plane" size="1 1 0.01" pos="0 0 {floor_z}" material="grid"/>
    {"".join(bodies)}
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml), links


def add_arrow(scene: mujoco.MjvScene, start: np.ndarray, vec: np.ndarray, rgba) -> None:
    """Arrow from start along vec (already scaled to meters)."""
    if np.linalg.norm(vec) < 1e-4 or scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.004, start, start + vec)
    scene.ngeom += 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rollout", help=".npz from record_rollout.py")
    p.add_argument("--envs", type=int, nargs="+", default=None, help="envs to show (default: up to 4)")
    p.add_argument("--size", type=int, default=480, help="pixel size of each env tile")
    p.add_argument("--distance", type=float, default=0.32, help="camera distance to the cube [m]")
    p.add_argument("--azimuth", type=float, default=135.0)
    p.add_argument("--elevation", type=float, default=-30.0)
    p.add_argument("--arrow_scale", type=float, default=0.05, help="arrow length per rad/s [m]")
    p.add_argument("--frames", type=int, default=None, help="render only the first N control steps")
    p.add_argument("--out", default=None, help="output .mp4 (default: next to the .npz)")
    args = p.parse_args()

    data = dict(np.load(args.rollout))
    T, n_envs = data["obj_pos"].shape[:2]
    T = min(T, args.frames or T)
    envs = args.envs if args.envs is not None else list(range(min(4, n_envs)))
    fps = round(1.0 / float(data["dt"]))
    out = args.out or args.rollout.replace(".npz", ".mp4")

    model, links = build_model(data)
    mdata = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.size, args.size)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = args.distance, args.azimuth, args.elevation
    cube = model.body("cube").mocapid[0]
    link_mocap = [model.body(f"l{i}").mocapid[0] for i in links]

    cols = int(np.ceil(np.sqrt(len(envs))))
    rows = int(np.ceil(len(envs) / cols))
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    episode_start = np.zeros(n_envs, dtype=int)
    try:
        for t in range(T):
            frame = np.zeros((rows * args.size, cols * args.size, 3), dtype=np.uint8)
            for k, e in enumerate(envs):
                mdata.mocap_pos[link_mocap] = data["link_pos"][t, e, links]
                mdata.mocap_quat[link_mocap] = data["link_quat"][t, e, links]
                mdata.mocap_pos[cube] = data["obj_pos"][t, e]
                mdata.mocap_quat[cube] = data["obj_quat"][t, e]
                mujoco.mj_forward(model, mdata)
                cam.lookat[:] = data["cmd_pos"][t, e] if "cmd_pos" in data else data["obj_pos"][t, e]
                renderer.update_scene(mdata, cam)
                c, side = data["obj_pos"][t, e], np.array([0.006, 0.0, 0.0])  # side by side, so neither hides the other
                add_arrow(renderer.scene, c - side, data["cmd_angvel"][t, e] * args.arrow_scale, (0.2, 0.9, 0.2, 0.9))
                add_arrow(renderer.scene, c + side, data["obj_angvel"][t, e] * args.arrow_scale, (1.0, 0.55, 0.1, 0.9))
                img = Image.fromarray(renderer.render())
                ImageDraw.Draw(img).text(
                    (8, 6),
                    f"env {e}  t={(t - episode_start[e]) * float(data['dt']):5.1f}s\n"
                    f"cmd  wz {data['cmd_angvel'][t, e, 2]:+.2f} rad/s\n"
                    f"cube wz {data['obj_angvel'][t, e, 2]:+.2f} rad/s\n"
                    f"reward {data['reward'][t, e]:+.2f}",
                    fill=(255, 255, 255),
                )
                r, col = divmod(k, cols)
                frame[r * args.size:(r + 1) * args.size, col * args.size:(col + 1) * args.size] = np.asarray(img)
            episode_start[data["done"][t] > 0] = t + 1
            writer.append_data(frame)
            if t % 100 == 0:
                print(f"frame {t}/{T}")
    finally:
        writer.close()
        renderer.close()
    print(f"wrote {out} ({T} frames, {fps} fps, envs {envs})")


if __name__ == "__main__":
    main()
