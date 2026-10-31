"""In-process stand-in for the parts of ROS 2 (rclpy, std_msgs, tf2_ros) the course's policy node uses, with a
lock-step simulated clock, so the node runs unchanged on a machine without ROS (DTU HPC) and deterministically.

``install()`` puts fake ``rclpy`` / ``rclpy.node`` / ``rclpy.executors`` / ``rclpy.time`` / ``std_msgs`` /
``std_msgs.msg`` / ``tf2_ros`` modules into ``sys.modules`` (call it before importing the node). Only what the nodes
here use is implemented:

- ``Node``: parameters (``declare_parameter`` / ``get_parameter``; values from ``PARAMS[node_name]`` override the
  defaults, like ``--ros-args -p``), publishers, subscriptions, timers, ``get_clock().now()``, ``get_logger()``.
- Topics deliver synchronously on ``publish`` (latency 0), or after ``LATENCY_S[topic]`` simulated seconds.
- ``Executor``: runs timers in simulated time. Time is counted in ticks of 1/1200 s, so 120 Hz physics, 100 Hz state,
  20 Hz policy timers are exact; at equal times timers fire by ``priority`` (lower first), then by creation order.
  rclpy timers fire first one period after creation; so do these.
- ``tf2_ros.Buffer.lookup_transform`` always raises ``TransformException`` (no cube tracker in the loop).

Not modeled: transport jitter, QoS / queue overflow, multi-threaded executors, wall-clock timing.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

TICKS_PER_S = 1200
PARAMS: dict[str, dict] = {}       # node name -> {param: value}
LATENCY_S: dict[str, float] = {}   # topic -> delivery delay (simulated seconds)
LOG_LEVEL = {"info": True}         # set LOG_LEVEL["info"] = False to silence info logs


class Executor:
    def __init__(self):
        self.tick = 0
        self.timers: list[_Timer] = []
        self.pending: list[tuple[int, int, str, object]] = []  # (deliver tick, seq, topic, msg)
        self.subs: dict[str, list] = {}
        self._seq = 0

    def now_ns(self) -> int:
        return self.tick * 1_000_000_000 // TICKS_PER_S

    def publish(self, topic: str, msg) -> None:
        delay = round(LATENCY_S.get(topic, 0.0) * TICKS_PER_S)
        if delay <= 0:
            for cb in list(self.subs.get(topic, [])):
                cb(msg)
        else:
            self._seq += 1
            self.pending.append((self.tick + delay, self._seq, topic, msg))

    def run_until(self, stop_tick: int, stop=lambda: False) -> None:
        """Advance to ``stop_tick``, firing due messages, then due timers, at each tick; ``stop()`` ends early."""
        while self.tick <= stop_tick and not stop():
            due = sorted(p for p in self.pending if p[0] <= self.tick)
            self.pending = [p for p in self.pending if p[0] > self.tick]
            for _, _, topic, msg in due:
                for cb in list(self.subs.get(topic, [])):
                    cb(msg)
            for t in sorted((t for t in self.timers if t.active and t.next_tick == self.tick),
                            key=lambda t: (t.priority, t.order)):
                t.next_tick += t.period
                t.callback()
                if stop():
                    return
            self.tick += 1


EXECUTOR = Executor()


def reset_executor() -> Executor:
    global EXECUTOR
    EXECUTOR = Executor()
    return EXECUTOR


@dataclass
class _Timer:
    period: int
    next_tick: int
    callback: object
    priority: int
    order: int
    active: bool = True

    def cancel(self):
        self.active = False


class _Param:
    def __init__(self, value):
        self.value = value


class _Logger:
    def __init__(self, name):
        self.name = name

    def _log(self, level, msg, **_):
        if level != "info" or LOG_LEVEL["info"]:
            print(f"[{level.upper()}] [{EXECUTOR.now_ns() / 1e9:9.3f}] [{self.name}]: {msg}")

    def info(self, msg, **kw):
        self._log("info", msg, **kw)

    def warn(self, msg, **kw):
        self._log("warn", msg, **kw)

    warning = warn

    def error(self, msg, **kw):
        self._log("error", msg, **kw)


class _Time:
    def __init__(self, seconds=0, nanoseconds=0):
        self.nanoseconds = int(seconds * 1e9) + nanoseconds

    @staticmethod
    def from_msg(stamp):
        return _Time(nanoseconds=stamp)


class _Clock:
    def now(self):
        return _Time(nanoseconds=EXECUTOR.now_ns())


class _Publisher:
    def __init__(self, topic):
        self.topic = topic

    def publish(self, msg):
        EXECUTOR.publish(self.topic, msg)


class Node:
    _order = 0

    def __init__(self, node_name: str, **_):
        self._name = node_name
        self._params: dict[str, _Param] = {}
        self._subs: list[tuple[str, object]] = []
        self._timers: list[_Timer] = []

    def get_name(self):
        return self._name

    def declare_parameter(self, name, value=None, *_, **__):
        self._params[name] = _Param(PARAMS.get(self._name, {}).get(name, value))
        return self._params[name]

    def get_parameter(self, name):
        return self._params[name]

    def create_publisher(self, _msg_type, topic, _qos=10):
        return _Publisher(topic)

    def create_subscription(self, _msg_type, topic, callback, _qos=10):
        EXECUTOR.subs.setdefault(topic, []).append(callback)
        self._subs.append((topic, callback))

    def create_service(self, *_, **__):
        return None

    def create_timer(self, period_s, callback, priority: int = 10):
        period = round(period_s * TICKS_PER_S)
        if period < 1 or abs(period - period_s * TICKS_PER_S) > 1e-6:
            raise ValueError(f"timer period {period_s} s is not a multiple of 1/{TICKS_PER_S} s")
        Node._order += 1
        t = _Timer(period, EXECUTOR.tick + period, callback, priority, Node._order)
        EXECUTOR.timers.append(t)
        self._timers.append(t)
        return t

    def get_clock(self):
        return _Clock()

    def get_logger(self):
        return _Logger(self._name)

    def destroy_node(self):
        for t in self._timers:
            t.cancel()
        EXECUTOR.timers = [t for t in EXECUTOR.timers if t.active]
        for topic, cb in self._subs:
            EXECUTOR.subs[topic].remove(cb)
        self._subs, self._timers = [], []


# --- std_msgs ------------------------------------------------------------------------------------------------------

@dataclass
class MultiArrayDimension:
    label: str = ""
    size: int = 0
    stride: int = 0


@dataclass
class MultiArrayLayout:
    dim: list = field(default_factory=list)
    data_offset: int = 0


@dataclass
class Float32MultiArray:
    layout: MultiArrayLayout = field(default_factory=MultiArrayLayout)
    data: list = field(default_factory=list)


# --- tf2_ros -------------------------------------------------------------------------------------------------------

class TransformException(Exception):
    pass


class Buffer:
    def lookup_transform(self, *_, **__):
        raise TransformException("no TF in the fake ROS loop")


class TransformListener:
    def __init__(self, *_, **__):
        pass


class ExternalShutdownException(Exception):
    pass


def install() -> None:
    """Register the fake modules (idempotent)."""
    if "rclpy" in sys.modules and getattr(sys.modules["rclpy"], "_fake", False):
        return

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        m._fake = True
        sys.modules[name] = m
        return m

    rclpy = mod("rclpy", init=lambda *a, **k: None, ok=lambda: True, shutdown=lambda *a, **k: None,
                spin=lambda node: EXECUTOR.run_until(10**12))
    rclpy.node = mod("rclpy.node", Node=Node)
    rclpy.executors = mod("rclpy.executors", ExternalShutdownException=ExternalShutdownException)
    rclpy.time = mod("rclpy.time", Time=_Time)
    std_msgs = mod("std_msgs")
    std_msgs.msg = mod("std_msgs.msg", Float32MultiArray=Float32MultiArray, MultiArrayDimension=MultiArrayDimension,
                       MultiArrayLayout=MultiArrayLayout)
    mod("tf2_ros", Buffer=Buffer, TransformListener=TransformListener, TransformException=TransformException)
