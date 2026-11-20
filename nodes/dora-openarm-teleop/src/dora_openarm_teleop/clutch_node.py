"""Dora node ``dora-openarm-clutch``: grip clutch between webxr and the IK (docs/teleop.md, Clutch).

Inputs:
  tick              – timer (dora/timer/millis/5); one output pair per tick
  pose_right/left   – POSE8 hand poses from webxr (arm_origin frame)
  grip_right/left   – SCALAR squeeze 0..1 from webxr
  state_right/left  – STATE8 measured joints (optional): the first, and the first of
                      each new follower start_epoch, re-seed that arm's hold there
  reset_position    – QPOS16 from the sim after a reset (absent on the real robot)
  control           – STRING ui/command: "quit" stops the node, "reset" holds home,
                      "proceed" (the UI's Enter) moves an intro on that waits for it,
                      and "success", "fail" or "cancel" (an episode ended) send
                      both arms back to their start poses and run the intro again
                      (--intro-every-episode)
  button_x          – BOOL webxr's left X button (optional): a press does what
                      "proceed" does

Outputs:
  target_right/left – POSE8 on every tick for both arms, so the bimanual IK
                      (which only solves once both sides have a new target)
                      never stalls because one controller dropped out
  clutch_state      – JSON {"right": s, "left": s} when a state changes

The event loop (`run`) takes any iterable of dora-style event dicts and a send
callable, so tests drive it without a daemon.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pyarrow as pa

from dora_openarm_teleop import arrow
from dora_openarm_teleop.clutch import Clutch, ClutchParams, ClutchTick, table_lift
from dora_openarm_teleop.gate import SIM_ROBOTS
from dora_openarm_teleop.intro import IntroParams
from dora_openarm_teleop.model_info import SCENES

Send = Callable[[str, pa.Array, dict[str, Any]], None]
Log = Callable[[str], None]

POSE_INPUTS = {"pose_right": "right", "pose_left": "left"}
GRIP_INPUTS = {"grip_right": "right", "grip_left": "left"}
# When every webxr stream has closed there is nobody left to teleoperate.
WEBXR_INPUTS = frozenset(POSE_INPUTS) | frozenset(GRIP_INPUTS)
STATE_INPUTS = {"state_right": "right", "state_left": "left"}


def _log(message: str) -> None:
    print(f"[dora-openarm-clutch] {message}", file=sys.stderr, flush=True)


class TickMonitor:
    """Warn when the tick rate drifts from what the IK's --tick-hz assumes.

    The IK derives its velocity limits from --tick-hz, not from measured time,
    so a clutch ticking slower than configured silently weakens them.
    """

    def __init__(self, expected_hz: float, window_s: float = 5.0, tolerance: float = 0.1) -> None:
        self.expected_hz = expected_hz
        self.window_s = window_s
        self.tolerance = tolerance
        self._start: float | None = None
        self._count = 0

    def tick(self, now: float) -> str | None:
        """Count a tick; return a warning once per window if the rate is off."""
        if self.expected_hz <= 0:
            return None
        if self._start is None:
            self._start = now
            return None
        self._count += 1
        elapsed = now - self._start
        if elapsed < self.window_s:
            return None
        rate = self._count / elapsed
        self._start, self._count = now, 0
        if abs(rate - self.expected_hz) > self.tolerance * self.expected_hz:
            return (
                f"tick rate {rate:.1f} Hz differs from the expected {self.expected_hz:.1f} Hz "
                "(the IK's --tick-hz must match the real rate)"
            )
        return None


class _RateLimitedLog:
    """Collapse per-message warnings (malformed input at 500 Hz) into one line per period."""

    def __init__(self, log: Log, period_s: float = 5.0) -> None:
        self._log = log
        self._period_s = period_s
        self._counts: dict[str, int] = {}
        self._last: float | None = None

    def warn(self, now: float, key: str) -> None:
        self._counts[key] = self._counts.get(key, 0) + 1
        if self._last is None or now - self._last >= self._period_s:
            summary = ", ".join(f"{k} x{n}" for k, n in sorted(self._counts.items()))
            self._log(f"ignored malformed input: {summary}")
            self._counts.clear()
            self._last = now


def _publish(send: Send, result: ClutchTick, log: Log) -> None:
    metadata = {"timestamp": arrow.now_ns()}
    for side, target in result.targets.items():
        send(f"target_{side}", arrow.pose_array(target), metadata)
    if result.changed:
        send("clutch_state", arrow.json_array(result.states), metadata)
        # The reason is logged but not published: a bare state change reads the
        # same whether the pose stopped, the grip did, or the arm kept its grip
        # across the gap, which is what made the LOST reports undiagnosable.
        log(
            "clutch state "
            + " ".join(
                f"{s}={v}" + (f" ({d})" if (d := result.details.get(s)) else "")
                for s, v in result.states.items()
            )
        )


def _handle_input(
    event: Mapping[str, Any], now: float, clutch: Clutch, send: Send, log: Log
) -> bool:
    """Apply one INPUT event. Returns False if it was malformed."""
    input_id = event["id"]
    if input_id == "tick":
        # Outside the try: a failing send is a real error, not bad input.
        _publish(send, clutch.tick(now), log)
        return True
    value = event["value"]
    try:
        if input_id in POSE_INPUTS:
            return clutch.on_pose(POSE_INPUTS[input_id], now, arrow.values(value, "pose"))
        if input_id in GRIP_INPUTS:
            return clutch.on_grip(GRIP_INPUTS[input_id], now, arrow.scalar(value))
        if input_id in STATE_INPUTS:
            side = STATE_INPUTS[input_id]
            epoch = arrow.start_epoch(event.get("metadata"))
            seeded = clutch.on_robot_state(side, arrow.values(value, "qpos"), epoch)
            if seeded and clutch.intro_running(side):
                log(f"{input_id} (start_epoch {epoch}): starting the {side} arm's intro")
            elif seeded:
                log(f"{input_id} (start_epoch {epoch}): holding the arm's measured pose, locked")
            return seeded is not None
        if input_id == "reset_position":
            accepted = clutch.reset_position(arrow.values(value, "qpos"))
            if accepted:
                log("reset_position: holding its FK pose, arms locked")
            return accepted
    except (IndexError, KeyError, TypeError, ValueError, pa.ArrowException):
        return False
    return True


# ui/command values that end an episode: the intro runs again after each.
EPISODE_END_COMMANDS = frozenset({"success", "fail", "cancel"})


def _proceed(clutch: Clutch, log: Log, source: str) -> None:
    if clutch.proceed():
        log(f"{source}: closing the gripper, then the intro's flip")
    else:
        log(f"{source} ignored: no intro is waiting for it")


def run(
    events: Iterable[Mapping[str, Any] | None],
    send: Send,
    clutch: Clutch,
    *,
    clock: Callable[[], float] = time.monotonic,
    monitor: TickMonitor | None = None,
    log: Log = _log,
    intro_every_episode: bool = False,
) -> str:
    """Process events until told to stop. Returns why it stopped."""
    closed: set[str] = set()
    malformed = _RateLimitedLog(log)
    # webxr sends every button on every frame: a press is a False -> True edge.
    x_down = False
    for event in events:
        if event is None:
            return "event stream ended"
        kind = event.get("type")
        if kind == "STOP":
            return "stop"
        if kind == "ERROR":
            log(f"dora error: {event.get('error')}")
            continue
        if kind == "INPUT_CLOSED":
            closed.add(str(event.get("id")))
            # The UI owns quit; without it nobody can end the
            # session cleanly, and exiting closes the IK's inputs so the gate
            # holds the arms.
            if "control" in closed:
                return "control input closed"
            if WEBXR_INPUTS <= closed:
                return "webxr inputs closed"
            continue
        if kind != "INPUT":
            continue

        now = clock()
        input_id = event.get("id")
        if input_id == "control":
            try:
                command = arrow.text(event["value"]).strip()
            except (IndexError, KeyError, TypeError, pa.ArrowException):
                malformed.warn(now, "control")
                continue
            if command == "quit":
                return "control=quit"
            if command == "reset":
                clutch.reset()
                log("control=reset: holding home, arms locked")
            elif command == "proceed":
                _proceed(clutch, log, "control=proceed")
            elif command in EPISODE_END_COMMANDS and intro_every_episode:
                sides = clutch.restart_intro()
                if sides:
                    note = (
                        f", the {clutch.intro} arm then runs its intro"
                        if clutch.intro in sides
                        else ""
                    )
                    log(
                        f"control={command}: {' and '.join(sides)} going back to the start pose{note}"
                    )
            continue
        if input_id == "button_x":
            try:
                pressed = bool(event["value"][0].as_py())
            except (IndexError, KeyError, TypeError, AttributeError, pa.ArrowException):
                malformed.warn(now, "button_x")
                continue
            if pressed and not x_down:
                _proceed(clutch, log, "button X")
            x_down = pressed
            continue
        if not _handle_input(event, now, clutch, send, log):
            malformed.warn(now, str(input_id))
        if input_id == "tick" and monitor is not None:
            warning = monitor.tick(now)
            if warning:
                log(warning)
    return "event stream ended"


def build_parser() -> argparse.ArgumentParser:
    defaults = ClutchParams()
    parser = argparse.ArgumentParser(
        prog="dora-openarm-clutch",
        description="Grip clutch between webxr hand poses and the OpenArm IK (docs/teleop.md, Clutch).",
    )
    parser.add_argument(
        "--scene",
        default="cell",
        choices=sorted(SCENES),
        help="scene whose MJCF supplies home, table height and gripper ranges",
    )
    parser.add_argument(
        "--scale", type=float, default=defaults.scale, help="hand-to-robot motion scale"
    )
    parser.add_argument("--engage-grip", type=float, default=defaults.engage_grip)
    parser.add_argument("--release-grip", type=float, default=defaults.release_grip)
    parser.add_argument(
        "--stale-ms",
        type=float,
        default=defaults.fresh_s * 1000.0,
        help="pose/grip older than this holds the arm and locks it",
    )
    parser.add_argument(
        "--settle-ms",
        type=float,
        default=defaults.settle_s * 1000.0,
        help="after a pose gap, stay locked until the pose has been fresh this long "
        "and is at rest (webxr's smoother may still be sliding)",
    )
    parser.add_argument(
        "--engage-max-speed",
        type=float,
        default=defaults.engage_max_speed,
        help="m/s: after a pose gap, stay locked while the hand pose moves faster than this "
        f"(finite difference over {defaults.speed_window_s * 1000:.0f} ms; inf: off)",
    )
    parser.add_argument(
        "--engage-max-angular-speed",
        type=float,
        default=defaults.engage_max_angular_speed,
        help="rad/s: the same test for the hand orientation (inf: off)",
    )
    parser.add_argument(
        "--resume-gap-ms",
        type=float,
        default=defaults.resume_gap_s * 1000.0,
        help="an arm that was already following keeps its grip across an input gap up to "
        "this long, if the pose comes back where it was (0: off, every gap needs a "
        "release and a re-squeeze)",
    )
    parser.add_argument(
        "--resume-max-move",
        type=float,
        default=defaults.resume_max_move,
        help="m: how far the first pose back may be from the one the arm was following "
        "for it to keep its grip",
    )
    parser.add_argument(
        "--resume-max-turn",
        type=float,
        default=defaults.resume_max_turn,
        help=f"rad: the same test for the hand orientation "
        f"(default {defaults.resume_max_turn:g} = {math.degrees(defaults.resume_max_turn):.0f} deg)",
    )
    parser.add_argument(
        "--gripper-pickup",
        type=float,
        default=defaults.gripper_pickup,
        help="rad: after engaging, the gripper keeps the held angle until the hand's "
        "gripper comes this close to it or crosses it",
    )
    parser.add_argument(
        "--no-clamps",
        action="store_true",
        help="disable all workspace clamps (the gripper is still clipped to its range)",
    )
    parser.add_argument("--no-table-clamp", action="store_true")
    parser.add_argument(
        "--table-z",
        type=_finite_float,
        default=None,
        help="table top height in the arm_origin frame (m), measured on the real robot; "
        "default: the --scene table (the sim cell's, -0.335)",
    )
    parser.add_argument("--table-clearance", type=float, default=defaults.table_clearance)
    parser.add_argument(
        "--tip-model",
        choices=("hull", "point"),
        default="hull",
        help="table clamp geometry: finger collision hulls at the commanded opening, "
        "or DESIGN's single lowest closed fingertip point",
    )
    parser.add_argument("--no-reach-clamp", action="store_true")
    parser.add_argument("--reach-max", type=float, default=defaults.reach_max)
    parser.add_argument("--no-midline-clamp", action="store_true")
    parser.add_argument("--midline-y", type=float, default=defaults.midline_y)
    parser.add_argument(
        "--tick-hz",
        type=float,
        default=200.0,
        help="expected tick rate; a warning is logged when the measured rate is >10%% off (0: off)",
    )
    # Defaults from the environment, so that a module sets them with `env:` and
    # scripts/check-dataflow, which parses these args under the node's env,
    # rejects a bad value before the robot starts.
    intro_defaults = IntroParams()
    parser.add_argument(
        "--intro",
        type=_intro_side,
        default=os.environ.get("INTRO_ARM", "none"),
        help="the arm that runs the opening sequence after each Start Arm (docs/teleop.md, "
        "Intro): right, left or none (default: $INTRO_ARM, else none)",
    )
    parser.add_argument(
        "--intro-every-episode",
        type=_on_off,
        default=os.environ.get("INTRO_EVERY_EPISODE", "on"),
        help="on: after each episode (success, fail or cancel) both arms go back to their "
        "start poses and the intro arm runs the intro again; off: only after Start Arm "
        "(default: $INTRO_EVERY_EPISODE, else on)",
    )
    parser.add_argument(
        "--intro-open",
        type=float,
        default=intro_defaults.open_fraction,
        help="fraction of its range the intro opens the gripper to",
    )
    parser.add_argument(
        "--intro-turn-speed",
        type=float,
        default=intro_defaults.turn_speed,
        help="rad/s: the intro's average roll speed",
    )
    parser.add_argument(
        "--fixed-grippers",
        type=_fixed_grippers,
        default=os.environ.get("FIXED_GRIPPERS", "none"),
        help="grippers held at a fixed opening whatever the trigger does, as SIDE=FRACTION "
        "of the range open (0: closed), comma-separated, e.g. right=0.3, or none "
        "(default: $FIXED_GRIPPERS, else none)",
    )
    parser.add_argument(
        "--finger-down",
        type=_sides,
        default=os.environ.get("FINGER_DOWN", "none"),
        help="comma-separated sides whose finger normally on the right is kept lowest, "
        "the target's roll replacing the hand's: right, left, right,left or none "
        "(default: $FINGER_DOWN, else none)",
    )
    return parser


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be a finite number, got {text!r}")
    return value


def _on_off(text: str) -> bool:
    value = text.strip().lower()
    if value in ("on", "1", "true", "yes"):
        return True
    if value in ("off", "0", "false", "no", ""):
        return False
    raise argparse.ArgumentTypeError(f"must be on or off, got {text!r}")


def _intro_side(text: str) -> str | None:
    side = text.strip().lower()
    if side in ("", "none"):
        return None
    if side not in ("right", "left"):
        raise argparse.ArgumentTypeError(f"must be right, left or none, got {text!r}")
    return side


def _sides(text: str) -> tuple[str, ...]:
    sides: list[str] = []
    for part in text.split(","):
        side = part.strip().lower()
        if side in ("", "none"):
            continue
        if side not in ("right", "left"):
            raise argparse.ArgumentTypeError(
                f"must be right, left, right,left or none, got {text!r}"
            )
        if side not in sides:
            sides.append(side)
    return tuple(sides)


def _fixed_grippers(text: str) -> tuple[tuple[str, float], ...]:
    fixed: dict[str, float] = {}
    for part in text.split(","):
        item = part.strip().lower()
        if item in ("", "none"):
            continue
        side, _, fraction = item.partition("=")
        side = side.strip()
        try:
            value = float(fraction)
        except ValueError:
            value = math.nan
        if side not in ("right", "left") or side in fixed or not 0.0 <= value <= 1.0:
            raise argparse.ArgumentTypeError(
                f"must be SIDE=FRACTION items (right or left, 0 to 1), or none, got {text!r}"
            )
        fixed[side] = value
    return tuple(fixed.items())


def params_from_args(args: argparse.Namespace) -> ClutchParams:
    return ClutchParams(
        engage_grip=args.engage_grip,
        release_grip=args.release_grip,
        fresh_s=args.stale_ms / 1000.0,
        scale=args.scale,
        clamp_table=not (args.no_clamps or args.no_table_clamp),
        table_clearance=args.table_clearance,
        clamp_reach=not (args.no_clamps or args.no_reach_clamp),
        reach_max=args.reach_max,
        clamp_midline=not (args.no_clamps or args.no_midline_clamp),
        midline_y=args.midline_y,
        settle_s=args.settle_ms / 1000.0,
        engage_max_speed=args.engage_max_speed,
        engage_max_angular_speed=args.engage_max_angular_speed,
        gripper_pickup=args.gripper_pickup,
        resume_gap_s=args.resume_gap_ms / 1000.0,
        resume_max_move=args.resume_max_move,
        resume_max_turn=args.resume_max_turn,
        fixed_grippers=args.fixed_grippers,
        finger_down=args.finger_down,
    )


def intro_params_from_args(args: argparse.Namespace) -> IntroParams:
    return IntroParams(open_fraction=args.intro_open, turn_speed=args.intro_turn_speed)


def startup_warnings(clutch: Clutch, table_z: float | None) -> list[str]:
    """Loud lines for table settings that can hurt the real robot (docs/teleop.md, Clutch)."""
    params = clutch.params
    warnings = []
    if not params.clamp_table:
        warnings.append(
            "WARNING: table clamp OFF - nothing keeps the fingertips above the table; "
            "measure the table top in arm_origin and pass --table-z"
        )
        return warnings
    if table_z is not None:
        for side, arm in clutch.arms.items():
            lift = table_lift(arm.hold, arm.model, params)
            if lift > 0.001:
                # The first engaged tick would lift the arm by this much.
                warnings.append(
                    f"WARNING: --table-z {table_z} is {lift * 1000:.0f} mm above the {side} "
                    "start pose's lowest fingertip; engaging will lift that arm"
                )
    return warnings


def apply_robot(args: argparse.Namespace, environ: Mapping[str, str]) -> list[str]:
    """Fit the table clamp to the dataflow's ROBOT; returns lines to log.

    TABLE_Z (m, arm_origin frame), when set and no --table-z was given, is the
    measured table. A robot that is not simulated (gate.SIM_ROBOTS) has no table
    in the --scene MJCF that means anything, so without TABLE_Z or --table-z its
    table clamp is turned off rather than clamping against an imaginary table.
    """
    lines = []
    text = str(environ.get("TABLE_Z", "") or "").strip()
    if args.table_z is None and text:
        args.table_z = _finite_float(text)
        lines.append(f"TABLE_Z={args.table_z} m")
    robot = str(environ.get("ROBOT", "") or "").strip()
    if robot not in SIM_ROBOTS and args.table_z is None and not args.no_table_clamp:
        args.no_table_clamp = True
        lines.append(
            f"ROBOT={robot or '(unset)'} is not simulated and TABLE_Z is unset: "
            f"the {args.scene} scene's table is not this robot's, so the table clamp is off"
        )
    return lines


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    robot_lines = apply_robot(args, os.environ)
    params = params_from_args(args)
    intro_params = intro_params_from_args(args)
    # Load the model before joining the dataflow so the first ticks are served.
    clutch = Clutch.from_scene(
        args.scene, params, args.tip_model, args.table_z, args.intro, intro_params
    )
    table_z = clutch.arms["right"].model.table_z
    _log(
        f"scene={args.scene} scale={params.scale} grip {params.release_grip}/{params.engage_grip} "
        f"stale={params.fresh_s * 1000:.0f}ms settle={params.settle_s * 1000:.0f}ms"
        f"(<{params.engage_max_speed} m/s, <{params.engage_max_angular_speed} rad/s) "
        f"resume={params.resume_gap_s * 1000:.0f}ms"
        f"(<{params.resume_max_move * 1000:.0f} mm, "
        f"<{math.degrees(params.resume_max_turn):.0f} deg) "
        f"table={params.clamp_table}(z={table_z:.3f} m "
        f"{'from --table-z' if args.table_z is not None else f'from scene {args.scene}'}) "
        f"reach={params.clamp_reach}({params.reach_max} m) midline={params.clamp_midline} "
        f"tip={args.tip_model} gripper_pickup={params.gripper_pickup} rad "
        f"intro={args.intro or 'none'}(open {intro_params.open_fraction:g}, "
        f"{intro_params.turn_speed:g} rad/s, "
        f"{'every episode' if args.intro_every_episode else 'after Start Arm'}) "
        f"fixed_grippers={','.join(f'{s}={v:g}' for s, v in params.fixed_grippers) or 'none'} "
        f"finger_down={','.join(params.finger_down) or 'none'}"
    )
    for line in robot_lines:
        _log(line)
    for warning in startup_warnings(clutch, args.table_z):
        _log(warning)

    import dora

    node = dora.Node()
    reason = run(
        node,
        node.send_output,
        clutch,
        monitor=TickMonitor(args.tick_hz),
        intro_every_episode=args.intro_every_episode,
    )
    _log(f"exiting: {reason}")


if __name__ == "__main__":
    main()
