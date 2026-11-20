"""dora-openarm-impedance: dora-openarm with joint impedance and gravity compensation.

docs/teleop.md, Impedance. dora-openarm drives each arm joint with an MIT command,
``kp·(q* − q) + kd·(0 − q̇) + tau``, with the driver config's kp and kd and tau = 0.
The springs alone then hold the arm's weight: it sags by gravity / kp, so kp has
to be stiff to stay near the target, and a stiff spring pushes hard on whatever
the arm touches.

This node runs dora-openarm unchanged, except that openarm_driver.SingleArmDriver
is replaced by a subclass that

* sends the arm's gravity torque at the latest measured joints as tau
  (dora_openarm_teleop.gravity, scaled by ``--gravity-scale``, clipped to
  TAU_LIMIT), so the arm holds its own weight,
* uses softer kp (``--kp``) for joints 1-7, so it yields to contact, and
* sends the commands' own joint velocity as dq (``--velocity-ff``). With dq = 0
  the damping term brakes every motion, and the arm trails a moving target by
  kd / kp seconds: 39 ms at the config's J1 gains, 138 ms at these, and
* adds each joint's friction to tau in the direction the commands move, while
  the joint trails its command that way (``--friction``). A joint only slides
  once its spring pulls harder than its friction, so without this the arm
  trails by friction / kp: 2-5 degrees at these kp, which reads as lag. Only
  while it trails, so the push stops once the joint catches up instead of
  carrying it past the target.

Everything else is openarm_driver's own: the safety checks, the gripper
(POS_FORCE, untouched), the start and stop moves (which get the feedforward
too) and the measured state. ``--impedance off`` runs dora-openarm exactly as
before.

Either way the node logs, every ``--latency-log-s``, how late the commands
reach it from the gate and how long dora-openarm spends on each event
(LatencyProbe): the measured arm trailed its commands by ~150 ms even at the
config's gains, more than kd / kp explains.

Arguments before ``--`` are this node's; the rest are dora-openarm's.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dora_openarm_teleop.gravity import ARM_DOF, ArmGravity
from dora_openarm_teleop.ik_node import split_argv

# kp for joints 1-7 (the driver config's openarm_pedestal: 70 70 70 60 10 10 10).
# About 3.5x softer at the shoulder and elbow: at dora-openarm's start pose a
# fingertip pushed 2 cm into a wall in front presses with ~8 N instead of ~28 N
# (docs/teleop.md, Impedance). The wrist keeps the config's 10: its joints sit
# 0.12 m from the fingertip at most, so softening them to 4 took 0.4 N off that
# at best, and let the gripper swing up to 16 deg off its command.
DEFAULT_KP = (20.0, 20.0, 12.0, 12.0, 10.0, 10.0, 10.0)
# Nm, the most tau this node adds per joint. The model's gravity torques stay
# under ~12 Nm (J1, arm stretched forward) and ~4.4 Nm (J4) in the arm's range;
# a value past these is a model or state error, not the arm's weight.
TAU_LIMIT = (15.0, 15.0, 8.0, 8.0, 2.0, 2.0, 2.0)
MAX_GRAVITY_SCALE = 1.5
# rad/s, the most dq this node sends: openarm_pedestal's joint_velocity_limits,
# for a config without them.
VELOCITY_LIMIT = (2.0, 2.0, 3.3, 3.3, 6.3, 6.3, 6.3)
# s: dq is the commands' change over the last this long. A few gate ticks (4 ms),
# so commands that arrive bunched together do not read as a jump.
VELOCITY_WINDOW_S = 0.02
# s: a pause between commands longer than this starts the estimate afresh.
VELOCITY_GAP_S = 0.1
# Nm, the friction torque added along the commands' motion, joints 1-7: 0.7x
# what the robot measured while sliding with dq = 0 and gravity fed forward
# (0.74 0.49 0.54 0.98 0.10 0.19 0.16; docs/teleop.md, Impedance). Short of the
# whole of it, so an arm the estimate overshoots does not run ahead of the hand.
DEFAULT_FRICTION = (0.52, 0.34, 0.38, 0.69, 0.07, 0.13, 0.11)
MAX_FRICTION = 3.0
# rad/s: the friction torque fades in as tanh(velocity / this), so commands at
# rest add none and a direction change does not flip it in one step.
FRICTION_SPEED = 0.05


def _log(message: str) -> None:
    print(f"[dora-openarm-impedance] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class ImpedanceSettings:
    """Gains and feedforward for joints 1-7; see the module docstring."""

    kp: tuple[float, ...] = DEFAULT_KP
    # None keeps the driver config's kd.
    kd: tuple[float, ...] | None = None
    gravity_scale: float = 1.0
    tau_limit: tuple[float, ...] = TAU_LIMIT
    scene: str = "demo"
    # Fraction of the commands' joint velocity sent as dq (0: dq = 0 as before).
    velocity_ff: float = 1.0
    # Nm per joint added along the commands' motion (zeros: none).
    friction: tuple[float, ...] = DEFAULT_FRICTION

    def __post_init__(self) -> None:
        for name in ("kp", "kd", "tau_limit", "friction"):
            values = getattr(self, name)
            if values is None:
                continue
            if len(values) != ARM_DOF or not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} needs {ARM_DOF} finite values, got {values!r}")
        if min(self.kp) <= 0.0:
            raise ValueError("kp must be positive")
        if self.kd is not None and min(self.kd) < 0.0:
            raise ValueError("kd must not be negative")
        if min(self.tau_limit) <= 0.0:
            raise ValueError("tau_limit must be positive")
        if not 0.0 <= self.gravity_scale <= MAX_GRAVITY_SCALE:
            raise ValueError(f"gravity_scale must be in [0, {MAX_GRAVITY_SCALE}]")
        if not 0.0 <= self.velocity_ff <= 1.0:
            raise ValueError("velocity_ff must be in [0, 1]")
        if not all(0.0 <= f <= MAX_FRICTION for f in self.friction):
            raise ValueError(f"friction must be in [0, {MAX_FRICTION}] Nm")


def feedforward(
    gravity: ArmGravity, settings: ImpedanceSettings, qpos: np.ndarray | None
) -> np.ndarray:
    """tau for joints 1-7 at the measured ``qpos`` (zeros before the first state)."""
    if qpos is None:
        return np.zeros(ARM_DOF)
    tau = settings.gravity_scale * gravity.torques(qpos)
    limit = np.asarray(settings.tau_limit)
    return np.clip(tau, -limit, limit)


class CommandVelocity:
    """The joint velocity of the commands, for the MIT command's dq.

    Their change over the last ``window_s`` (plus the newest sample older than
    that), measured at arrival, so it is right at the gate's 250 Hz and at the
    driver's 50 Hz start and stop moves alike, and two commands that arrive
    together do not make a spike. Clipped to ``limits``, and zero after a pause
    longer than ``gap_s``.
    """

    def __init__(
        self,
        limits: Sequence[float],
        window_s: float = VELOCITY_WINDOW_S,
        gap_s: float = VELOCITY_GAP_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limits = np.asarray(limits, dtype=float)
        self._window_s = window_s
        self._gap_s = gap_s
        self._clock = clock
        self._samples: deque[tuple[float, np.ndarray]] = deque()

    def __call__(self, q: Sequence[float]) -> np.ndarray:
        now = self._clock()
        samples = self._samples
        if samples and not 0.0 <= now - samples[-1][0] <= self._gap_s:
            samples.clear()
        samples.append((now, np.asarray(q, dtype=float).copy()))
        while len(samples) >= 2 and now - samples[1][0] >= self._window_s:
            samples.popleft()
        then, old = samples[0]
        if now - then <= 1e-4:
            return np.zeros_like(self._limits)
        velocity = (samples[-1][1] - old) / (now - then)
        return np.clip(velocity, -self._limits, self._limits)


class FeedforwardArm:
    """The arm's motor collection, adding tau and dq to every MIT command sent to it.

    ``tau()`` is the gravity feedforward. ``velocity(q)``, when given, is the
    commands' joint velocity: ``velocity_ff`` of it is sent as dq, and
    ``friction`` (Nm per joint) is added to tau along it, faded in by
    tanh(velocity / friction_speed). With ``measured()`` (the joints as the
    commands count them, None before the first state) it is also scaled by how
    far the joint trails its command in that direction, full at friction / kp
    and none once it has caught up; without it, it is not.
    """

    def __init__(
        self,
        arm: Any,
        tau: Callable[[], np.ndarray],
        velocity: Callable[[Sequence[float]], np.ndarray] | None = None,
        velocity_ff: float = 1.0,
        friction: Sequence[float] | None = None,
        friction_speed: float = FRICTION_SPEED,
        measured: Callable[[], np.ndarray | None] | None = None,
    ) -> None:
        self._arm = arm
        self._tau = tau
        self._velocity = velocity
        self._velocity_ff = velocity_ff
        self._friction = None if friction is None else np.asarray(friction, dtype=float)
        self._friction_speed = friction_speed
        self._measured = measured

    def mit_control_all(self, params: Sequence[Any]) -> Any:
        import openarm_can as oa

        tau = np.asarray(self._tau(), dtype=float)
        if self._velocity is None:
            dq = np.zeros(len(params))
        else:
            q = np.array([p.q for p in params])
            moving = self._velocity(q)
            dq = self._velocity_ff * moving
            if self._friction is not None:
                tau = tau + self._friction_torque(q, moving, np.array([p.kp for p in params]))
        return self._arm.mit_control_all(
            [
                oa.MITParam(p.kp, p.kd, p.q, p.dq + float(v), p.tau + float(t))
                for p, v, t in zip(params, dq, tau, strict=True)
            ]
        )

    def _friction_torque(self, q: np.ndarray, moving: np.ndarray, kp: np.ndarray) -> np.ndarray:
        assert self._friction is not None
        push = self._friction * np.tanh(moving / self._friction_speed)
        if self._measured is None:
            return push
        measured = self._measured()
        if measured is None:
            return np.zeros_like(push)
        # How far the joint trails along the motion, in units of the lag friction
        # alone would leave (friction / kp): 0 once it has caught up.
        trail = (q - np.asarray(measured, dtype=float)) * np.sign(moving)
        band = np.where(self._friction > 0.0, self._friction / np.maximum(kp, 1e-9), 1.0)
        return push * np.clip(trail / band, 0.0, 1.0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._arm, name)


class FeedforwardOpenArm:
    """openarm_can.OpenArm whose get_arm() is a FeedforwardArm; the rest passes through."""

    def __init__(
        self,
        openarm: Any,
        tau: Callable[[], np.ndarray],
        velocity: Callable[[Sequence[float]], np.ndarray] | None = None,
        **feedforward: Any,
    ) -> None:
        self._openarm = openarm
        self._arm = FeedforwardArm(openarm.get_arm(), tau, velocity, **feedforward)

    def get_arm(self) -> FeedforwardArm:
        return self._arm

    def __getattr__(self, name: str) -> Any:
        return getattr(self._openarm, name)


def _gains(arm: Sequence[float] | None, config: np.ndarray) -> np.ndarray:
    """The config's gains with joints 1-7 replaced (the gripper's entry is kept)."""
    gains = np.array(config, dtype=float)
    if arm is not None:
        gains[:ARM_DOF] = arm
    return gains


def impedance_driver(base: type, settings: ImpedanceSettings, default_config: Callable) -> type:
    """A subclass of openarm_driver.SingleArmDriver (``base``) with these settings."""

    class ImpedanceArmDriver(base):
        def __init__(self, arm_side: str, config: Any = None, kps=None, kds=None, **kwargs):
            config = config if config is not None else default_config()
            if kps is None:
                kps = _gains(settings.kp, config.get_default_kps())
            if kds is None:
                kds = _gains(settings.kd, config.get_default_kds())
            super().__init__(arm_side, config, kps=kps, kds=kds, **kwargs)
            self._gravity = ArmGravity(arm_side.removesuffix("_arm"), settings.scene)
            limits = config.get_joint_velocity_limits() or VELOCITY_LIMIT
            friction = settings.friction if any(settings.friction) else None
            velocity = (
                CommandVelocity(limits[:ARM_DOF])
                if settings.velocity_ff > 0.0 or friction is not None
                else None
            )
            # After the base class has read the motors: from here on every MIT
            # command, the start and stop moves included, carries the feedforward.
            self.openarm = FeedforwardOpenArm(
                self.openarm,
                self._feedforward,
                velocity,
                velocity_ff=settings.velocity_ff,
                friction=friction,
                measured=self._measured_joints,
            )

        def _measured_joints(self) -> np.ndarray | None:
            """Joints 1-7 as the MIT commands count them (with the config's offsets)."""
            state = self.latest_state
            if state is None:
                return None
            return state["qpos"][:ARM_DOF] + self.joint_offsets[:ARM_DOF]

        def _feedforward(self) -> np.ndarray:
            state = self.latest_state
            return feedforward(self._gravity, settings, None if state is None else state["qpos"])

    return ImpedanceArmDriver


class LatencyProbe:
    """A dora.Node that logs how late move_position commands reach dora-openarm.

    The gate stamps each command with metadata["timestamp"] (UNIX ns) and
    dora-echo passes it on, so the delay from the gate to this node is the time
    it is handed to dora-openarm minus that stamp. Also how long dora-openarm
    spends on each event before it asks for the next. One line every
    ``period_s``; everything else is the node's own.
    """

    def __init__(
        self,
        node: Any,
        period_s: float,
        log: Callable[[str], None] = _log,
        clock_ns: Callable[[], int] = time.time_ns,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._node = node
        self._period_s = period_s
        self._log = log
        self._clock_ns = clock_ns
        self._monotonic = monotonic
        self._delays_ms: list[float] = []
        self._busy_ms: list[float] = []
        self._events = 0
        self._since = monotonic()

    def __iter__(self) -> Iterator[Any]:
        for event in self._node:
            if event.get("type") == "INPUT" and event.get("id") == "move_position":
                stamp = (event.get("metadata") or {}).get("timestamp")
                if isinstance(stamp, datetime.datetime):
                    stamp = int(stamp.timestamp() * 1e9)
                if isinstance(stamp, int):
                    self._delays_ms.append((self._clock_ns() - stamp) / 1e6)
            started = self._monotonic()
            yield event
            finished = self._monotonic()
            self._busy_ms.append((finished - started) * 1000)
            self._events += 1
            if finished - self._since >= self._period_s:
                self._report(finished)

    def _report(self, now: float) -> None:
        span = now - self._since
        if self._delays_ms:
            delays = np.asarray(self._delays_ms)
            delay = (
                f"{len(delays)} move_position, gate -> here median "
                f"{np.median(delays):.1f} ms, p95 {np.percentile(delays, 95):.1f}, "
                f"max {delays.max():.1f}"
            )
        else:
            delay = "no move_position"
        busy = np.asarray(self._busy_ms)
        self._log(
            f"{delay}; {self._events} events in {span:.1f} s, "
            f"dora-openarm per event median {np.median(busy):.2f} ms, "
            f"p95 {np.percentile(busy, 95):.2f}, max {busy.max():.1f}"
        )
        self._delays_ms, self._busy_ms, self._events, self._since = [], [], 0, now

    def __getattr__(self, name: str) -> Any:
        return getattr(self._node, name)


def _seven(text: str) -> tuple[float, ...] | None:
    """Seven comma-separated numbers, or None for an empty string."""
    if not text.strip():
        return None
    try:
        values = tuple(float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be {ARM_DOF} comma-separated numbers") from None
    if len(values) != ARM_DOF:
        raise argparse.ArgumentTypeError(f"must be {ARM_DOF} comma-separated numbers, got {text!r}")
    return values


def build_parser() -> argparse.ArgumentParser:
    # Defaults from the environment, so that a module sets them with `env:`.
    environ = os.environ
    parser = argparse.ArgumentParser(
        prog="dora-openarm-impedance",
        description="dora-openarm with gravity compensation and soft joint gains "
        "(arguments after -- go to dora-openarm)",
    )
    parser.add_argument(
        "--impedance",
        choices=["on", "off"],
        default=(environ.get("IMPEDANCE") or "on").strip().lower(),
        help="off runs dora-openarm unchanged (default: $IMPEDANCE, else on)",
    )
    parser.add_argument(
        "--kp",
        type=_seven,
        default=environ.get("IMPEDANCE_KP", ""),
        help="kp for joints 1-7, comma-separated (default: $IMPEDANCE_KP, else "
        f"{','.join(f'{v:g}' for v in DEFAULT_KP)})",
    )
    parser.add_argument(
        "--kd",
        type=_seven,
        default=environ.get("IMPEDANCE_KD", ""),
        help="kd for joints 1-7, comma-separated (default: $IMPEDANCE_KD, else the config's)",
    )
    parser.add_argument(
        "--gravity-scale",
        type=float,
        default=float(environ.get("GRAVITY_SCALE") or 1.0),
        help="fraction of the model's gravity torque sent (default: $GRAVITY_SCALE, else 1)",
    )
    parser.add_argument(
        "--velocity-ff",
        type=float,
        default=float(environ.get("IMPEDANCE_VELOCITY_FF") or 1.0),
        help="fraction of the commands' joint velocity sent as the MIT dq, 0 for dq = 0 "
        "(default: $IMPEDANCE_VELOCITY_FF, else 1)",
    )
    parser.add_argument(
        "--friction",
        type=_seven,
        default=environ.get("IMPEDANCE_FRICTION", ""),
        help="Nm per joint 1-7 added to tau along the commands' motion, comma-separated, "
        "all zeros for none (default: $IMPEDANCE_FRICTION, else "
        f"{','.join(f'{v:g}' for v in DEFAULT_FRICTION)})",
    )
    parser.add_argument(
        "--latency-log-s",
        type=float,
        default=float(environ.get("LATENCY_LOG_S") or 10.0),
        help="log how late commands arrive every this many seconds, 0 for never "
        "(default: $LATENCY_LOG_S, else 10)",
    )
    parser.add_argument("--scene", default="demo", help="MJCF scene the arm model comes from")
    return parser


def settings_from_args(args: argparse.Namespace) -> ImpedanceSettings | None:
    """The settings, or None when the impedance is off."""
    if args.impedance == "off":
        return None
    return ImpedanceSettings(
        kp=args.kp if args.kp is not None else DEFAULT_KP,
        kd=args.kd,
        gravity_scale=args.gravity_scale,
        scene=args.scene,
        velocity_ff=args.velocity_ff,
        friction=args.friction if args.friction is not None else DEFAULT_FRICTION,
    )


def main(argv: list[str] | None = None) -> None:
    own, rest = split_argv(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(own)
    settings = settings_from_args(args)
    import dora
    import openarm_driver
    from dora_openarm import main as dora_openarm_main

    if args.latency_log_s > 0:
        # dora-openarm calls dora.Node() itself; hand it the probe instead.
        node_class = dora.Node
        dora.Node = lambda *a, **k: LatencyProbe(node_class(*a, **k), args.latency_log_s)

    if settings is None:
        _log("impedance off: dora-openarm as it is")
    else:
        # dora-openarm builds its arm as openarm_driver.SingleArmDriver(...), looked
        # up when the arm is started, so replacing the name here is enough.
        openarm_driver.SingleArmDriver = impedance_driver(
            openarm_driver.SingleArmDriver, settings, openarm_driver.get_default_config
        )
        kd = "config" if settings.kd is None else ",".join(f"{v:g}" for v in settings.kd)
        _log(
            f"kp={','.join(f'{v:g}' for v in settings.kp)} kd={kd} "
            f"gravity_scale={settings.gravity_scale:g} velocity_ff={settings.velocity_ff:g} "
            f"friction={','.join(f'{v:g}' for v in settings.friction)} "
            f"scene={settings.scene}"
        )
    sys.argv = ["dora-openarm", *rest]
    dora_openarm_main.main()


if __name__ == "__main__":
    main()
