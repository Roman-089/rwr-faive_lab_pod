"""Instantiate env cfgs of orca_v2_repose_ablation_env_cfg on a CPU login node (no Isaac Sim) and print their reward /
obs / action setup: catches import and __post_init__ errors before a queued job hits them.

Usage: PYTHONPATH=~/IsaacLab/source/isaaclab:~/IsaacLab/source/isaaclab_rl:source/faive_lab \
       python scripts/check_env_cfgs.py OrcaV2ReposeAsymRotLinEnvCfg [...]
omni / carb / isaacsim / warp / usdrt / pxr are replaced by stubs, so only the cfg classes are checked, not the sim."""
import importlib.abc, importlib.machinery, sys, types

class _Meta(type):
    """Stub classes: any attribute is another stub class, calling one returns a stub class (cfg imports only)."""
    def __getattr__(cls, k):
        if k.startswith("__"): raise AttributeError(k)
        return _Meta(k, (), {})
    def __call__(cls, *a, **kw): return _Meta(cls.__name__, (), {})
    def __iter__(cls): return iter(())

class _Stub(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"): raise AttributeError(k)
        v = _Meta(k, (), {}); setattr(self, k, v); return v

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    P = ("omni", "carb", "isaacsim", "warp", "usdrt", "pxr")
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.P:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec): m = _Stub(spec.name); m.__path__ = []; return m
    def exec_module(self, m): pass

sys.meta_path.insert(0, _Finder())
m = importlib.import_module("faive_lab.tasks.orca_inhand.orca_v2_repose_ablation_env_cfg")
for n in sys.argv[1:]:
    c = getattr(m, n)(); r = c.rewards
    g = lambda t: (getattr(r, t, None) and getattr(r, t).weight)
    print(f"{n}: rot {g('rotation_progress')} offaxis {g('off_axis_angular_velocity')} dir {g('angular_velocity_direction')} "
          f"mag {g('angular_velocity_magnitude')} drop {g('object_drop')} hist {c.observations.policy.history_length} "
          f"teacher {getattr(c.observations, 'teacher', None) is not None} action {type(c.actions.joint_pos).__name__}")
