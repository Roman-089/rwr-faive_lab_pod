# Owner: X (Task X / FAST)
"""Single source of truth for the FAST screw world – used for BOTH the sim (world_usd.py) and the
printable CAD (scripts/fast_world/export_cad.py). Change a number here, regenerate both, and sim and
real fixture stay identical.

Setup (after the reference video "Robot fingers go invisible mode on screws"): a fixture plate lies on the
table, an M8 bolt stands upright in it (head captured in a hex pocket from below), and the hand hangs
palm-down above it and spins a hex nut on the bolt.

World frame: table top z = 0, plate centred at the origin, bolt axis = world z.
Right-hand thread: turning the nut clockwise seen from above (negative about +z) moves it DOWN (= screw in).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WorldGeometry:
    # fixture plate (printed, PETG/PLA) ---------------------------------------------------------
    plate_x: float = 0.100
    plate_y: float = 0.060
    plate_t: float = 0.008
    mount_hole_d: float = 0.0064          # M6 clearance, 4 holes to clamp/screw the plate down
    mount_hole_dx: float = 0.080
    mount_hole_dy: float = 0.040
    # bolt ISO 4017 M8 x 40 (standard part) -----------------------------------------------------
    bolt_d: float = 0.008
    bolt_len: float = 0.040
    bolt_head_af: float = 0.013           # across flats
    bolt_head_h: float = 0.0053
    pocket_clearance: float = 0.0002      # hex pocket oversize per side (print tolerance)
    hole_clearance: float = 0.0004        # through-hole oversize on the diameter
    # nut ISO 4032 M8 (standard part) -----------------------------------------------------------
    nut_af: float = 0.013
    nut_h: float = 0.0068
    thread_pitch: float = 0.00125         # M8 coarse
    nut_mass: float = 0.005
    # hand placement: where the nut centre sits in the HAND (forearm) frame ---------------------
    # hand frame: fingers +y, palm/thumb side -z, index on -x (from orcahand_right.usd)
    nut_in_hand: tuple = (-0.030, 0.235, -0.015)     # TODO: tune with the play video / real hand
    nut_start_height: float = 0.5                      # 0 = nut on the plate, 1 = top of the free thread

    # derived -----------------------------------------------------------------------------------
    @property
    def bolt_above_plate(self) -> float:
        return self.bolt_len - (self.plate_t - self.bolt_head_h)

    @property
    def nut_z_min(self) -> float:          # nut centre when sitting on the plate
        return self.plate_t + self.nut_h / 2

    @property
    def nut_z_max(self) -> float:          # nut centre at the top of the bolt (2 mm of thread left)
        return self.plate_t + self.bolt_above_plate - self.nut_h / 2 - 0.002

    @property
    def nut_travel(self) -> float:
        return self.nut_z_max - self.nut_z_min

    def asdict(self):
        d = asdict(self)
        d.update(bolt_above_plate=self.bolt_above_plate, nut_z_min=self.nut_z_min,
                 nut_z_max=self.nut_z_max, nut_travel=self.nut_travel)
        return d


# hand orientation: palm down, fingers pointing down (-z), palm facing +x
#   hand x -> world +y, hand y -> world -z, hand z -> world -x
HAND_R = ((0.0, 0.0, -1.0),
          (1.0, 0.0, 0.0),
          (0.0, -1.0, 0.0))


def quat_from_matrix(R):
    """(w, x, y, z) from a 3x3 rotation matrix (rows)."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        q = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        q = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def matvec(R, v):
    return tuple(sum(R[i][k] * v[k] for k in range(3)) for i in range(3))


def hand_pose(g: WorldGeometry = WorldGeometry()):
    """Hand root (forearm frame) pose in the world so that `nut_in_hand` lands on the nut at its start height."""
    nut_world = (0.0, 0.0, g.nut_z_min + g.nut_start_height * g.nut_travel)
    off = matvec(HAND_R, g.nut_in_hand)
    pos = tuple(n - o for n, o in zip(nut_world, off))
    return pos, quat_from_matrix(HAND_R)


if __name__ == "__main__":
    g = WorldGeometry()
    p, q = hand_pose(g)
    print("hand root pos [m]", tuple(round(c, 4) for c in p), " quat (w,x,y,z)", tuple(round(c, 4) for c in q))
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in g.asdict().items()})
