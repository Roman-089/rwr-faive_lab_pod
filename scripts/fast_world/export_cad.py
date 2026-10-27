# Owner: X (Task X / FAST)
"""Export the FAST screw world as CAD – from the SAME geometry the sim uses (tasks/fast_screw/world.py).

    pip install cadquery
    python scripts/fast_world/export_cad.py --out ../rwr-fast/taskx/cad/screw_world

Writes:
    fixture_plate.step / .stl     print this (PETG/PLA, 100 % infill around the hex pocket recommended)
    bolt_M8x40_ref.step           ISO 4017 reference body (buy the real part)
    nut_M8_ref.step               ISO 4032 reference body (buy the real part)
    screw_world_assembly.step     plate + bolt + nut at the sim start height (import into Fusion)
    world.json                    all numbers incl. the hand pose relative to the plate
    drawing.png                   dimensioned top + side view
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "faive_lab" / "faive_lab" / "tasks" / "fast_screw"))
from world import WorldGeometry, hand_pose  # noqa: E402  (plain module import, no isaaclab needed)

import cadquery as cq  # noqa: E402

MM = 1000.0


def hex_prism(af, h):
    return cq.Workplane("XY").polygon(6, af / math.cos(math.pi / 6)).extrude(h)


def build(g: WorldGeometry):
    # plate: origin = centre of the bottom face, top face at z = plate_t
    plate = (cq.Workplane("XY").box(g.plate_x * MM, g.plate_y * MM, g.plate_t * MM, centered=(True, True, False))
             .edges("|Z").fillet(4.0))
    # bolt through-hole
    plate = plate.cut(cq.Workplane("XY").circle((g.bolt_d + g.hole_clearance) * MM / 2).extrude(g.plate_t * MM))
    # hex pocket from below for the bolt head (head flush with the bottom face)
    pocket_af = (g.bolt_head_af + 2 * g.pocket_clearance) * MM
    plate = plate.cut(hex_prism(pocket_af, g.bolt_head_h * MM))
    # 4 mounting holes
    for sx in (-1, 1):
        for sy in (-1, 1):
            plate = plate.cut(cq.Workplane("XY").center(sx * g.mount_hole_dx * MM / 2, sy * g.mount_hole_dy * MM / 2)
                              .circle(g.mount_hole_d * MM / 2).extrude(g.plate_t * MM))
    # engraved label on top
    plate = plate.cut(cq.Workplane("XY").workplane(offset=g.plate_t * MM - 0.6).center(0, -g.plate_y * MM / 2 + 7)
                      .text("FAST M8", 6, 0.6, kind="bold"))

    # reference bolt (head in the pocket, shaft up)
    bolt = hex_prism(g.bolt_head_af * MM, g.bolt_head_h * MM).union(
        cq.Workplane("XY").workplane(offset=g.bolt_head_h * MM).circle(g.bolt_d * MM / 2)
        .extrude((g.bolt_len) * MM))
    # reference nut at the sim start height
    z_nut = (g.nut_z_min + g.nut_start_height * g.nut_travel - g.nut_h / 2) * MM
    nut = (hex_prism(g.nut_af * MM, g.nut_h * MM)
           .cut(cq.Workplane("XY").circle(g.bolt_d * MM / 2).extrude(g.nut_h * MM))
           .translate((0, 0, z_nut)))
    return plate, bolt, nut


def drawing(g: WorldGeometry, out: Path):
    """Top view (section just below the top face) + side view of plate/bolt/nut, dimensioned, as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import trimesh
    except ImportError:
        print("(drawing skipped: pip install matplotlib trimesh)")
        return
    m = trimesh.load(out / "fixture_plate.stl")
    sec = m.section(plane_origin=[0, 0, g.plate_t * MM - 0.3], plane_normal=[0, 0, 1])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for e in sec.discrete:
        a.plot(e[:, 0], e[:, 1], "k-", lw=0.8)
    a.set_aspect("equal"); a.set_title("fixture plate – top view [mm]"); a.grid(alpha=0.2)
    a.annotate(f"{g.plate_x*MM:.0f}", (0, -g.plate_y * MM / 2 - 6), ha="center")
    a.annotate(f"{g.plate_y*MM:.0f}", (g.plate_x * MM / 2 + 4, 0), va="center")
    a.annotate(f"4x Ø{g.mount_hole_d*MM:.1f} @ {g.mount_hole_dx*MM:.0f} x {g.mount_hole_dy*MM:.0f}",
               (-g.plate_x * MM / 2, g.plate_y * MM / 2 + 3))
    a.annotate(f"Ø{(g.bolt_d+g.hole_clearance)*MM:.1f} thru, hex pocket {g.bolt_head_af*MM+2*g.pocket_clearance*MM:.1f} AF "
               f"x {g.bolt_head_h*MM:.1f} from below", (-g.plate_x * MM / 2, -g.plate_y * MM / 2 - 12), fontsize=8)
    # side view
    pt, ba, nh = g.plate_t * MM, g.bolt_above_plate * MM, g.nut_h * MM
    zc = (g.nut_z_min + g.nut_start_height * g.nut_travel) * MM
    b.add_patch(plt.Rectangle((-g.plate_x * MM / 2, 0), g.plate_x * MM, pt, fc="0.2"))
    b.add_patch(plt.Rectangle((-g.bolt_d * MM / 2, pt), g.bolt_d * MM, ba, fc="0.75", ec="k"))
    b.add_patch(plt.Rectangle((-g.nut_af * MM / 2, zc - nh / 2), g.nut_af * MM, nh, fc="0.85", ec="k"))
    b.axhspan(g.nut_z_min * MM, g.nut_z_max * MM, xmin=0.82, xmax=0.86, color="tab:orange", alpha=0.6)
    b.annotate(f"nut travel {g.nut_travel*MM:.1f} mm\n(pitch {g.thread_pitch*MM:.2f} mm/rev)",
               (8, g.nut_z_min * MM), fontsize=8, color="tab:orange")
    b.annotate(f"bolt above plate {ba:.1f}", (-g.plate_x * MM / 2, pt + ba + 1), fontsize=8)
    b.set_xlim(-g.plate_x * MM / 2 - 2, g.plate_x * MM / 2 + 2); b.set_ylim(-2, pt + ba + 8)
    b.set_aspect("equal"); b.set_title("side view [mm] (hand hangs above, palm down)"); b.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "drawing.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="cad_out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    g = WorldGeometry()
    plate, bolt, nut = build(g)
    cq.exporters.export(plate, str(out / "fixture_plate.step"))
    cq.exporters.export(plate, str(out / "fixture_plate.stl"), tolerance=0.02, angularTolerance=0.1)
    cq.exporters.export(bolt, str(out / "bolt_M8x40_ref.step"))
    cq.exporters.export(nut, str(out / "nut_M8_ref.step"))

    asm = (cq.Assembly(name="screw_world")
           .add(plate, name="fixture_plate", color=cq.Color(0.1, 0.1, 0.1))
           .add(bolt, name="bolt_M8x40", color=cq.Color(0.7, 0.7, 0.7))
           .add(nut, name="nut_M8", color=cq.Color(0.8, 0.8, 0.8)))
    asm.save(str(out / "screw_world_assembly.step"))
    drawing(g, out)

    pos, quat = hand_pose(g)
    info = {
        "units": "metres, world frame: table top z=0, plate centred at origin, bolt axis = +z",
        "geometry": {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.asdict().items()},
        "hand_root_pose_in_world": {"pos_m": pos, "quat_wxyz": quat,
                                    "meaning": "pose of the ORCA forearm frame (right_forearm) – palm down, "
                                               "fingers pointing down, palm facing +x"},
        "standard_parts": ["ISO 4017 M8x40 hex bolt", "ISO 4032 M8 hex nut", "4x M6 screws to fix the plate"],
    }
    (out / "world.json").write_text(json.dumps(info, indent=2))
    print("written:", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
