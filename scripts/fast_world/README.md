# FAST screw world – CAD export (owner: X)

`export_cad.py` builds the printable fixture plate + reference M8 bolt/nut from
`source/faive_lab/faive_lab/tasks/fast_screw/world.py` – the same numbers the sim uses.

```bash
pip install cadquery matplotlib trimesh
python scripts/fast_world/export_cad.py --out ../rwr-fast/taskx/cad/screw_world
```
Design notes and decisions: `rwr-fast/taskx/docs/sim_world.md`.
