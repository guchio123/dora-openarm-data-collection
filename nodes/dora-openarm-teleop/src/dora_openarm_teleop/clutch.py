"""Grip clutch: per-arm hold, relative mapping and workspace clamps (docs/teleop.md, Clutch).

Pure logic without dora. Time is passed in as monotonic seconds so tests can
replay exact sequences; ``dora_openarm_teleop.clutch_node`` only parses messages and
forwards them here.

Each arm keeps a held target ``H`` (POSE8). While the grip is squeezed the
target follows the *displacement* of the hand since the moment of engagement,
never its absolute pose, so engaging and re-engaging cannot make the arm jump.

That only holds if the pose is the real hand. After a tracking loss, a recentre
or a network stall, webxr's pose smoother may still be sliding from the last
pose before the gap towards the real one (CLUTCH-1), and an anchor taken during
the slide makes the target follow the rest of it with the hand still. So after
any pose gap the arm stays locked until the pose has been fresh for
``settle_s`` and has come to rest (finite-difference speed over
``speed_window_s`` below ``engage_max_speed`` and ``engage_max_angular_speed``).

That rule is about *taking* an anchor, so it applies to a new engagement only.
An arm that was already engaged when the input stopped keeps its engagement
suspended rather than losing it (CLUTCH-3): when the input returns it is taken
back, without a release or a settle, if the gap was shorter than
``resume_gap_s`` and the first pose back is within ``resume_max_move`` /
``resume_max_turn`` of the one the arm was following when the gap began. A
longer gap, or a pose that came back somewhere else, is the CLUTCH-1 hazard and
keeps the full rule: disarm, release, settle.

The gripper uses pickup semantics (CLUTCH-2): after engaging, the target keeps
the held gripper angle until the hand's gripper comes within
``gripper_pickup`` of it or crosses it, so re-engaging with the trigger in
another position does not drop a held object.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dora_openarm_teleop import frames
from dora_openarm_teleop.intro import SETUP, Intro, IntroParams, Return, ease, finger_down

ENGAGED = "engaged"
RELEASED = "released"
STALE = "stale"
LOCKED = "locked"

# Gripper angle -> (N, 3) points in the ee_control_point frame that must stay
# above the table. A constant (1, 3) array is DESIGN's single tip_offset.
TipPoints = Callable[[float], np.ndarray]


@dataclass(frozen=True)
class ClutchParams:
    """Tunables; docs/teleop.md (Clutch) explains the defaults."""

    engage_grip: float = 0.7
    release_grip: float = 0.4
    fresh_s: float = 0.100
    scale: float = 1.0
    clamp_table: bool = True
    table_clearance: float = 0.005
    clamp_reach: bool = True
    # From the shoulder centre to the wrist (ee_control_point). Upper arm plus
    # forearm is 0.436 m fully stretched; 0.43 keeps the elbow bent ~19° so the
    # IK stays clear of the J4 = 0 singularity.
    reach_max: float = 0.43
    clamp_midline: bool = True
    midline_y: float = 0.05
    # After a pose gap: stay locked until the pose has been fresh this long and
    # is at rest. 0.3 s covers webxr's smoother slide for moves up to ~25 cm
    # (1 m/s rate limit plus its ~80 ms low-pass tail); the speed test covers
    # longer slides. At 0.05 m/s over 50 ms, an exponential tail with that time
    # constant has under 3 mm left to go.
    settle_s: float = 0.3
    engage_max_speed: float = 0.05  # m/s; inf disables
    engage_max_angular_speed: float = 0.5  # rad/s; inf disables
    speed_window_s: float = 0.05
    # CLUTCH-3: a gap must not cost the operator their grip. An arm that was
    # ENGAGED when the input stopped keeps the engagement suspended and takes it
    # back when the input returns, if the gap was no longer than resume_gap_s
    # and the first pose back is within resume_max_move / resume_max_turn of the
    # one the arm was following when the gap began. The anchor is re-based on
    # the returning pose, so the target does not move; the hand travel the
    # clutch could not see (at most resume_max_move) is dropped, never applied
    # as a jump. Anything else disarms exactly as before.
    #
    # 0.5 s covers the gaps a 2.4 GHz link produces (a burst of ~7 lost frames
    # at 72 Hz is 100-150 ms; the measured sessions' stale episodes resolve
    # inside a second) while bounding what an arm can be asked to bridge; the
    # deliberate silences (menu open, headset off) are far longer. 2 cm / 5 deg
    # is the noise a still hand plus webxr's smoother shows across such a gap,
    # an order of magnitude below CLUTCH-1's 25 cm slide.
    #
    # Only a pose delivered after the gap can be re-based on, which relies on
    # webxr having stopped republishing its cached pose by then: POSE_TIMEOUT <=
    # fresh_s, the interlock scripts/check-dataflow enforces. 0 disables resuming.
    resume_gap_s: float = 0.5
    resume_max_move: float = 0.02  # m
    resume_max_turn: float = 0.087  # rad (5 deg)
    # rad: the gripper target stays at the held angle until the hand's gripper
    # is this close to it (or crosses it).
    gripper_pickup: float = 0.05
    # (side, fraction of the gripper's range open) for grippers held at a fixed
    # opening whatever the hand's trigger does; 0 is closed.
    fixed_grippers: tuple[tuple[str, float], ...] = ()
    # Sides whose finger normally on the right is kept lowest: the target's roll
    # about its approach axis is set, the hand's ignored (intro.finger_down).
    finger_down: tuple[str, ...] = ()
    # rad/s, the average speed of the finger-down turn after each start; it eases
    # in and out like the intro's rolls.
    turn_speed: float = 0.5
    # rad: a measured gripper this close to closed counts as closed.
    closed_tol: float = 0.15
    # s: a side that moves on its own (fixed gripper, finger down) is commanded
    # a closed gripper at least this long after each follower start before it
    # moves (see ClutchArm._update_alignment).
    fixed_close_s: float = 0.3

    def __post_init__(self) -> None:
        sides = [side for side, _ in self.fixed_grippers]
        if not set(sides) <= {"right", "left"} or len(set(sides)) != len(sides):
            raise ValueError("fixed_grippers may name 'right' and 'left' once each")
        if not set(self.finger_down) <= {"right", "left"}:
            raise ValueError("finger_down may only name 'right' and 'left'")
        if not (math.isfinite(self.turn_speed) and self.turn_speed > 0.0):
            raise ValueError("turn_speed must be positive")
        if not all(0.0 <= fraction <= 1.0 for _, fraction in self.fixed_grippers):
            raise ValueError("fixed_grippers fractions must be in [0, 1]")
        if not (math.isfinite(self.closed_tol) and self.closed_tol > 0.0):
            raise ValueError("closed_tol must be positive")
        if not (math.isfinite(self.fixed_close_s) and self.fixed_close_s >= 0.0):
            raise ValueError("fixed_close_s must be a finite, non-negative number")
        if not 0.0 <= self.release_grip < self.engage_grip <= 1.0:
            raise ValueError("need 0 <= release_grip < engage_grip <= 1 (hysteresis)")
        if self.fresh_s <= 0.0:
            raise ValueError("fresh_s must be positive")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be a positive number")
        if self.reach_max <= 0.0 or self.table_clearance < 0.0:
            raise ValueError("reach_max must be positive and table_clearance non-negative")
        if not (math.isfinite(self.settle_s) and self.settle_s >= 0.0):
            raise ValueError("settle_s must be a finite, non-negative number")
        if not (self.engage_max_speed > 0.0 and self.engage_max_angular_speed > 0.0):
            raise ValueError("engage_max_speed and engage_max_angular_speed must be positive")
        if not (math.isfinite(self.speed_window_s) and self.speed_window_s > 0.0):
            raise ValueError("speed_window_s must be positive")
        if not (math.isfinite(self.gripper_pickup) and self.gripper_pickup > 0.0):
            raise ValueError("gripper_pickup must be positive")
        if not (math.isfinite(self.resume_gap_s) and self.resume_gap_s >= 0.0):
            raise ValueError("resume_gap_s must be a finite, non-negative number (0 disables)")
        if not (math.isfinite(self.resume_max_move) and self.resume_max_move >= 0.0):
            raise ValueError("resume_max_move must be a finite, non-negative number")
        if not (math.isfinite(self.resume_max_turn) and self.resume_max_turn >= 0.0):
            raise ValueError("resume_max_turn must be a finite, non-negative number")


@dataclass(frozen=True)
class ArmModel:
    """What the clutch needs to know about one arm, all in the arm_origin frame."""

    side: str
    home: np.ndarray  # pose7 of ee_control_point at keyframe home
    gripper_range: tuple[float, float]
    shoulder: np.ndarray  # (3,) centre of the reach sphere
    table_z: float
    tip_points: TipPoints

    def __post_init__(self) -> None:
        if self.side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got {self.side!r}")


def _constant_points(points: np.ndarray) -> TipPoints:
    fixed = np.atleast_2d(np.asarray(points, dtype=np.float64))
    return lambda _gripper: fixed


def _posed_hulls(hulls: tuple[Any, ...]) -> TipPoints:
    """Finger hull vertices at a gripper angle, reusing the last result.

    The trigger rests at exactly 0 or 1 most of the time, so the same angle
    repeats for long stretches and posing ~1500 vertices can be skipped.
    """
    last: list[Any] = [None, None]

    def tips(gripper: float) -> np.ndarray:
        if last[0] != gripper:
            last[1] = np.concatenate([hull.points_in_site(gripper) for hull in hulls])
            last[0] = gripper
        return last[1]

    return tips


def arm_models_from_scene(
    scene: str = "cell", tip_model: str = "hull", table_z: float | None = None
) -> dict[str, ArmModel]:
    """Build both ArmModels from the MJCF (mujoco is imported only here).

    ``tip_model="point"`` is DESIGN's single ``tip_offset`` (lowest closed
    fingertip point); ``"hull"`` (default) keeps every finger collision-hull
    vertex posed at the commanded gripper angle, which also covers tilted and
    open grippers whose corners dip below that one point.

    ``table_z`` (arm_origin frame, m) replaces the scene's table top. The arm
    facts are the real arm's too, but the table plane is not: on the real robot
    it must be measured, not taken from the sim cell (docs/teleop.md, Clutch).
    """
    from dora_openarm_teleop import model_info

    if tip_model not in ("hull", "point"):
        raise ValueError(f"tip_model must be 'hull' or 'point', got {tip_model!r}")
    home = model_info.home_ee_poses(scene)
    shoulders = model_info.shoulder_positions(scene)
    ranges = model_info.gripper_ranges(scene)
    if table_z is None:
        table_z = model_info.table_top_z_arm_origin(scene)
    elif not math.isfinite(table_z):
        raise ValueError("table_z must be a finite number")
    models = {}
    for side in model_info.SIDES:
        if tip_model == "point":
            tips = _constant_points(model_info.fingertip_offset(side, scene))
        else:
            tips = _posed_hulls(model_info.finger_hulls(side, scene))
        models[side] = ArmModel(
            side, home[side], ranges[side], shoulders[side], float(table_z), tips
        )
    return models


def table_lift(pose8: np.ndarray, model: ArmModel, params: ClutchParams) -> float:
    """How far (m) the table clamp alone would raise ``pose8``; 0 if it is clear.

    A positive value at the held pose means the first engaged tick lifts the arm,
    which is what a table plane set above the arm's start pose looks like.
    """
    if not params.clamp_table:
        return 0.0
    out = np.array(pose8, dtype=np.float64)
    low, high = model.gripper_range
    out[7] = min(max(out[7], low), high)
    z_row = frames.quat_to_matrix(out[3:7])[2]
    lowest = float(np.min(model.tip_points(out[7]) @ z_row))
    before = float(out[2])
    _lift_above_table(out, lowest, model, params)
    return float(out[2]) - before


def _parse_pose(values: Any) -> np.ndarray | None:
    """POSE8 as float64 with a unit quaternion, or None if malformed."""
    try:
        pose = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if pose.shape != (8,) or not np.all(np.isfinite(pose)):
        return None
    norm = np.linalg.norm(pose[3:7])
    # WebXR sends unit quaternions; anything far off is corruption, not rounding.
    if not 0.5 < norm < 1.5:
        return None
    pose[3:7] = frames.quat_normalize(pose[3:7])
    return pose


def _parse_grip(value: Any) -> float | None:
    try:
        grip = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(grip):
        return None
    return min(max(grip, 0.0), 1.0)


def clamp_target(target: np.ndarray, model: ArmModel, params: ClutchParams) -> np.ndarray:
    """Apply the gripper clip and the enabled workspace clamps to a POSE8.

    Only the position moves, never the orientation. The table clamp runs first
    and again last: the reach projection pulls towards a shoulder that sits
    above the table and the midline clip keeps z, so the second pass is
    normally a no-op, but it makes "fingertips above the table" hold whatever
    the other clamps did.
    """
    out = np.array(target, dtype=np.float64)
    low, high = model.gripper_range
    out[7] = min(max(out[7], low), high)

    lowest = 0.0
    if params.clamp_table:
        # Lowest tip point relative to p along world z; R and the gripper do
        # not change below, so this is computed once.
        z_row = frames.quat_to_matrix(out[3:7])[2]
        lowest = float(np.min(model.tip_points(out[7]) @ z_row))
        _lift_above_table(out, lowest, model, params)

    if params.clamp_reach:
        offset = out[:3] - model.shoulder
        distance = float(np.linalg.norm(offset))
        if distance > params.reach_max:
            out[:3] = model.shoulder + offset * (params.reach_max / distance)

    if params.clamp_midline:
        if model.side == "right":
            out[1] = min(out[1], params.midline_y)
        else:
            out[1] = max(out[1], -params.midline_y)

    if params.clamp_table:
        _lift_above_table(out, lowest, model, params)
    return out


def _lift_above_table(
    target: np.ndarray, lowest: float, model: ArmModel, params: ClutchParams
) -> None:
    floor = model.table_z + params.table_clearance - lowest
    target[2] = max(target[2], floor)


class ClutchArm:
    """Clutch state machine and held target for one arm."""

    def __init__(self, model: ArmModel, params: ClutchParams) -> None:
        self.model = model
        self.params = params
        low, high = model.gripper_range
        # 0 is closed (model_info.gripper_ranges); the far end of the range is open.
        self._closed_angle = min(max(0.0, low), high)
        fraction = dict(params.fixed_grippers).get(model.side)
        far = high if abs(high) >= abs(low) else low
        # The angle this side's gripper is held at, None when it follows the hand.
        self._fixed_open = (
            None if fraction is None else self._closed_angle + fraction * (far - self._closed_angle)
        )
        self._finger_down = model.side in params.finger_down
        # dora-openarm (--align-trigger gripper) takes no command until it has been
        # sent a closed gripper, and a target that moves before then can leave it
        # unaligned for good. So a side that moves on its own (a fixed gripper
        # opening, the finger-down turn) is first commanded a closed gripper after
        # each follower start, until the arm measures it closed.
        self._needs_alignment = self._fixed_open is not None or self._finger_down
        self._aligned = False
        self._closed_since: float | None = None
        self._measured_gripper: float | None = None
        self._hold = np.concatenate([model.home, [0.0]]).astype(np.float64)
        # The finger-down turn after a start or a reset: the orientation it eases
        # from (None: no turn pending) and when it began (None: not yet).
        self._turn_from: np.ndarray | None = self._hold[3:7].copy() if self._finger_down else None
        self._turn_since: float | None = None
        self._pose: np.ndarray | None = None
        self._pose_t: float | None = None
        self._grip: float | None = None
        self._grip_t: float | None = None
        self._armed = False
        self._engaged = False
        # (hand position p0, hand orientation q0, held target P) at engagement.
        self._anchor: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        # Settling after a pose gap: when the gap-free pose stream began, recent
        # (time, pose) samples for the speed test, and whether it has settled.
        self._pose_since: float | None = None
        self._recent: deque[tuple[float, np.ndarray]] = deque()
        self._settled = False
        # Gripper pickup: whether the target follows the hand's gripper yet, and
        # the side of the held angle the hand's gripper was on at engagement.
        self._gripper_following = True
        self._gripper_side = 0.0
        # Bridging a gap (CLUTCH-3): whether an engagement is suspended, the
        # pose the arm was following when the input stopped, when the input
        # stopped, and the first pose delivered after it.
        self._suspended = False
        self._suspend_pose: np.ndarray | None = None
        self._suspend_from: float | None = None
        self._resume_pose: np.ndarray | None = None
        self._resume_t: float | None = None
        self._state = STALE
        # What the last resume decision did, for the state-change log line.
        self._note = ""
        self._detail = ""

    @property
    def state(self) -> str:
        return self._state

    @property
    def detail(self) -> str:
        """Why the arm is in ``state``: empty when there is nothing to say."""
        return self._detail

    @property
    def hold(self) -> np.ndarray:
        """Current held target H (POSE8, float64 copy)."""
        return self._hold.copy()

    def on_pose(self, now: float, values: Any) -> bool:
        """Store a hand POSE8 sample. Returns False (and keeps nothing) if malformed."""
        pose = _parse_pose(values)
        if pose is None:
            return False
        if self._gap(self._pose_t, now):
            self._lose_input()
            # Whatever produced the gap, the pose may now be sliding: settle again.
            self._pose_since = now
            self._recent.clear()
            self._settled = False
        self._pose, self._pose_t = pose, now
        self._remember(now, pose)
        if self._suspended and self._resume_pose is None:
            # The first pose after the gap, and the only one an engagement may be
            # re-based on: everything published before the input was declared
            # lost may still be webxr's cached frame rather than the hand.
            self._resume_pose, self._resume_t = pose.copy(), now
        self._try_resume(now)
        return True

    def on_grip(self, now: float, value: Any) -> bool:
        """Store a grip sample (0..1). Returns False if malformed."""
        grip = _parse_grip(value)
        if grip is None:
            return False
        if self._gap(self._grip_t, now):
            self._lose_input()
        self._grip, self._grip_t = grip, now
        self._try_resume(now)
        return True

    def reset(self, pose7: np.ndarray | None = None, gripper: float = 0.0) -> None:
        """Hold ``pose7`` (home if None) and lock until the grip is released."""
        pose = self.model.home if pose7 is None else np.asarray(pose7, dtype=np.float64)
        low, high = self.model.gripper_range
        self._hold = np.concatenate([pose[:3], frames.quat_normalize(pose[3:7]), [0.0]])
        own = self._gripper_target()
        if own is not None:
            gripper = own
        self._hold[7] = min(max(float(gripper), low), high)
        if self._finger_down:
            # Turned in from here, slowly, rather than snapped to on the next tick.
            self._turn_from, self._turn_since = self._hold[3:7].copy(), None
        self._clear_suspension()
        self._disarm()
        self._note = ""
        if self._state in (ENGAGED, RELEASED):
            self._state = LOCKED

    def step(self, now: float) -> np.ndarray:
        """Advance the state machine at ``now`` and return the target (POSE8 float64)."""
        own = self._gripper_target(now)
        if own is not None:
            self._hold[7] = own
        if self._turn_from is not None:
            # Before the hand: the turn runs whether a headset is there or not.
            return self._turn(now)
        if not self._fresh(now):
            self._lose_input()
            self._state, self._detail = STALE, self._late_detail(now)
            return self._hold.copy()
        assert self._grip is not None and self._pose is not None
        self._try_resume(now)  # in case no sample arrived with both streams fresh
        if self._suspended:
            # Both streams are fresh again but no pose has arrived since the
            # input was lost, so what is held may still be webxr's cached frame.
            # At most one tick, the one between the grip and the next pose.
            self._state, self._detail = STALE, "waiting for a pose to resume on"
            return self._hold.copy()

        if self._engaged and self._grip < self.params.release_grip:
            # Only the grip ever ends an engagement the clutch already has; the
            # arming rule below guards *taking* an anchor (CLUTCH-1), which a
            # resumed engagement (CLUTCH-3) has not done.
            self._engaged = False
            self._anchor = None
        if not self._engaged:
            if not self._armed:
                # Locked until the pose has settled after a gap, and then until the
                # grip is released: a squeeze during settling needs a fresh release.
                if not self._settle(now) or self._grip >= self.params.release_grip:
                    self._state = LOCKED
                    self._detail = self._note or self._locked_detail(now)
                    self._note = ""
                    return self._hold.copy()
                self._armed = True
            if self._grip >= self.params.engage_grip:
                self._engaged = True
                self._anchor = (self._pose[:3].copy(), self._pose[3:7].copy(), self._hold.copy())
                self._gripper_following = False
                self._gripper_side = 0.0

        if self._engaged:
            self._pick_up_gripper()
            target = self._follow()
            if self._finger_down:
                target[3:7] = finger_down(target[3:7])
            self._hold = clamp_target(target, self.model, self.params)
        self._state = ENGAGED if self._engaged else RELEASED
        self._detail, self._note = self._note, ""
        return self._hold.copy()

    def _follow(self) -> np.ndarray:
        """Unclamped target: held pose moved by the hand's displacement since engagement."""
        assert self._anchor is not None and self._pose is not None
        p0, q0, held = self._anchor
        target = np.empty(8)
        target[:3] = held[:3] + self.params.scale * (self._pose[:3] - p0)
        delta = frames.quat_delta(q0, self._pose[3:7])
        target[3:7] = frames.quat_normalize(frames.quat_multiply(delta, held[3:7]))
        own = self._gripper_target()
        if own is not None:
            # Not picked up: it moves (closed, then open) while the anchor's stays.
            target[7] = own
        else:
            target[7] = self._hand_gripper() if self._gripper_following else held[7]
        return target

    def on_measured_gripper(self, value: float) -> None:
        """The arm's latest measured gripper angle, which the alignment waits on."""
        self._measured_gripper = float(value)

    def follower_restarted(self) -> None:
        """A new follower start: wait for the arm to align again (see _update_alignment)."""
        self._aligned = False
        self._closed_since = None

    def _update_alignment(self, now: float) -> None:
        """Note when the follower has aligned: a closed gripper, commanded for
        fixed_close_s, that the arm measures closed shows it takes commands."""
        if self._aligned:
            return
        if self._closed_since is None:
            self._closed_since = now
        measured = self._measured_gripper
        if (
            now - self._closed_since >= self.params.fixed_close_s
            and measured is not None
            and abs(measured - self._closed_angle) <= self.params.closed_tol
        ):
            self._aligned = True

    def _gripper_target(self, now: float | None = None) -> float | None:
        """The gripper this side holds on its own at ``now``; None when it follows the hand.

        Closed until the follower has aligned, for a side that needs it; then a
        fixed gripper's opening, or None.
        """
        if not self._needs_alignment:
            return None
        if now is not None:
            self._update_alignment(now)
        return self._fixed_open if self._aligned else self._closed_angle

    def _turn(self, now: float) -> np.ndarray:
        """The finger-down turn: hold still until the arm aligns, then ease the roll in.

        The arm is not taken by the hand meanwhile (state ``setup``); after the
        turn it is locked until the grip is released, as after any reset.
        """
        assert self._turn_from is not None
        self._state = SETUP
        if not self._aligned:
            self._detail = "closing the gripper so that the arm aligns"
            return self._hold.copy()
        goal = finger_down(self._turn_from)
        angle = float(frames.quat_angle(frames.quat_delta(self._turn_from, goal)))
        if self._turn_since is None:
            self._turn_since = now
        t = (now - self._turn_since) * self.params.turn_speed / angle if angle > 1e-9 else 1.0
        if t >= 1.0:
            self._hold[3:7] = goal
            self._turn_from = None
            self._detail = "turned the finger on the right down"
        else:
            self._hold[3:7] = frames.quat_slerp(self._turn_from, goal, ease(t))
            self._detail = "turning the finger on the right down"
        return self._hold.copy()

    def _hand_gripper(self) -> float:
        assert self._pose is not None
        low, high = self.model.gripper_range
        # Clipped first: webxr overshoots past closed (e.g. +10 deg), which the
        # arm can never reach, so the unclipped value might never come near it.
        return min(max(float(self._pose[7]), low), high)

    def _pick_up_gripper(self) -> None:
        """Start following the hand's gripper once it reaches the held angle.

        Following at once would snap the gripper to wherever the trigger rests
        when re-engaging, e.g. open a closed gripper and drop what it holds.
        """
        if self._gripper_following:
            return
        assert self._anchor is not None
        offset = self._hand_gripper() - float(self._anchor[2][7])
        side = float(np.sign(offset))
        if abs(offset) <= self.params.gripper_pickup:
            self._gripper_following = True
        elif self._gripper_side == 0.0:
            self._gripper_side = side  # first engaged tick
        elif side != self._gripper_side:
            self._gripper_following = True  # crossed the held angle between samples

    def _remember(self, now: float, pose: np.ndarray) -> None:
        """Keep the samples of the last speed window plus the newest one older than it."""
        recent = self._recent
        if recent and now < recent[-1][0]:
            recent.clear()  # clock went backwards: the old samples give no usable speed
        elif recent and now == recent[-1][0]:
            recent.pop()  # same instant: keep the newest value
        recent.append((now, pose))
        window = self.params.speed_window_s
        while len(recent) >= 2 and now - recent[1][0] >= window:
            recent.popleft()

    def _settle(self, now: float) -> bool:
        """Whether the pose has settled since the last gap (latched until the next one)."""
        if self._settled:
            return True
        if self._pose_since is None or now - self._pose_since < self.params.settle_s:
            return False
        if math.isfinite(self.params.engage_max_speed) or math.isfinite(
            self.params.engage_max_angular_speed
        ):
            (t0, p0), (t1, p1) = self._recent[0], self._recent[-1]
            dt = t1 - t0
            if dt < self.params.speed_window_s:
                return False
            speed = float(np.linalg.norm(p1[:3] - p0[:3])) / dt
            turn = float(frames.quat_angle(frames.quat_delta(p0[3:7], p1[3:7]))) / dt
            if speed > self.params.engage_max_speed:
                return False
            if turn > self.params.engage_max_angular_speed:
                return False
        self._settled = True
        return True

    def _fresh(self, now: float) -> bool:
        limit = self.params.fresh_s
        return (
            self._pose_t is not None
            and self._grip_t is not None
            and now - self._pose_t <= limit
            and now - self._grip_t <= limit
        )

    def _gap(self, last: float | None, now: float) -> bool:
        # A sample after a silence longer than the freshness window means the
        # input went stale in between, even if no tick ran to notice it.
        return last is None or now - last > self.params.fresh_s

    def _disarm(self) -> None:
        self._armed = False
        self._engaged = False
        self._anchor = None

    # --- bridging an input gap (CLUTCH-3) ------------------------------------

    def _lose_input(self) -> None:
        """The input stopped. An engaged arm keeps its engagement pending a return.

        Called on every stale tick as well as on the sample that ends the gap,
        so only the first call (the one still engaged) records the suspension.
        """
        if (
            self.params.resume_gap_s > 0.0
            and self._engaged
            and self._pose is not None
            and self._pose_t is not None
        ):
            self._suspended = True
            self._suspend_pose = self._pose.copy()
            # The gap starts at the older of the two streams' last samples:
            # webxr republishes a cached pose for POSE_TIMEOUT after the frames
            # stop, so the grip's is the truer end of the input.
            self._suspend_from = (
                self._pose_t if self._grip_t is None else min(self._pose_t, self._grip_t)
            )
            self._resume_pose = self._resume_t = None
        self._disarm()

    def _clear_suspension(self) -> None:
        self._suspended = False
        self._suspend_pose = self._suspend_from = None
        self._resume_pose = self._resume_t = None

    def _try_resume(self, now: float) -> None:
        """Settle a suspended engagement once the input is back: take it or let it go."""
        if not self._suspended or self._resume_pose is None or not self._fresh(now):
            return
        assert self._pose is not None and self._suspend_pose is not None
        assert self._suspend_from is not None and self._resume_t is not None
        params = self.params
        gap = self._resume_t - self._suspend_from
        moved = float(np.linalg.norm(self._resume_pose[:3] - self._suspend_pose[:3]))
        turned = float(
            frames.quat_angle(frames.quat_delta(self._suspend_pose[3:7], self._resume_pose[3:7]))
        )
        # An operator who let go has no grip left to keep: that is a new
        # engagement whenever it comes, and it settles like any other.
        held = self._grip is not None and self._grip >= params.release_grip
        near = moved <= params.resume_max_move and turned <= params.resume_max_turn
        keep = held and gap <= params.resume_gap_s and near
        self._clear_suspension()
        if not keep:
            if not held:
                self._note = "let go during the gap"
            elif gap > params.resume_gap_s:
                self._note = f"{gap * 1000:.0f} ms gap, too long to keep the grip"
            else:
                self._note = (
                    f"the pose came back {moved * 1000:.0f} mm / "
                    f"{math.degrees(turned):.0f} deg away after {gap * 1000:.0f} ms"
                )
            return
        # Re-based on the pose that is there now (the first one back, except for
        # a grip that arrived after it), so this tick's target is the held one.
        self._engaged = True
        # Not armed: a *new* engagement after the gap still settles (CLUTCH-1).
        self._armed = False
        self._anchor = (self._pose[:3].copy(), self._pose[3:7].copy(), self._hold.copy())
        self._gripper_following = False
        self._gripper_side = 0.0
        self._note = f"kept the grip across a {gap * 1000:.0f} ms gap"

    def _late_detail(self, now: float) -> str:
        if self._pose_t is None or self._grip_t is None:
            return "no hand yet" if self._pose_t is None else "no grip yet"
        pose_late, grip_late = now - self._pose_t, now - self._grip_t
        which, late = ("pose", pose_late) if pose_late >= grip_late else ("grip", grip_late)
        kept = ", keeping the grip" if self._suspended else ""
        return f"{which} {late * 1000:.0f} ms late{kept}"

    def _locked_detail(self, now: float) -> str:
        if self._grip is not None and self._grip >= self.params.release_grip:
            return "release the grip"
        if self._pose_since is None:
            return "settling"
        left = max(0.0, self.params.settle_s - (now - self._pose_since))
        return f"settling, {left * 1000:.0f} ms" if left > 0.0 else "settling, hold still"


@dataclass(frozen=True)
class ClutchTick:
    """Result of one tick: a POSE8 (float32) per side and the per-arm states."""

    targets: dict[str, np.ndarray]
    states: dict[str, str]
    # States differ from the previous tick (always True on the first), or an
    # intro moved on to its next step.
    changed: bool
    # Why each arm is in its state ("grip 214 ms late", "release the grip",
    # "kept the grip across a 137 ms gap"); logged, never published: a state
    # change alone is the same line whether the pose stopped or the grip did.
    details: dict[str, str] = field(default_factory=dict)


ForwardKinematics = Callable[[np.ndarray], Mapping[str, np.ndarray]]


class Clutch:
    """Both arms plus the reset handling shared between them."""

    def __init__(
        self,
        models: Mapping[str, ArmModel],
        params: ClutchParams | None = None,
        fk: ForwardKinematics | None = None,
        intro: str | None = None,
        intro_params: IntroParams | None = None,
    ) -> None:
        params = params or ClutchParams()
        if intro not in (None, "right", "left"):
            raise ValueError(f"intro must be 'right', 'left' or None, got {intro!r}")
        if intro is not None and intro in params.finger_down:
            raise ValueError(f"the {intro} arm cannot run the intro and keep its finger down")
        self.params = params
        # Right before left, the order of every bimanual message.
        self.arms = {side: ClutchArm(models[side], params) for side in ("right", "left")}
        self._fk = fk
        self._last_states: dict[str, str] | None = None
        # The follower's start_epoch last seen per arm (None: a robot without one).
        self._robot_epochs: dict[str, int | None] = {}
        # The side that runs the opening sequence after each follower start
        # (docs/teleop.md, Intro), its parameters and the sequences under way.
        self.intro = intro
        self.intro_params = intro_params or IntroParams()
        self._intros: dict[str, Intro | Return] = {}
        # Where each arm held after its follower's last start (FK of the measured
        # start pose): an intro run again after an episode goes back there.
        self._start_poses: dict[str, np.ndarray] = {}
        # What each arm in "setup" (intro or finger-down turn) said last tick.
        self._last_setup_details: dict[str, str] = {}
        # The latest measured gripper angle per arm, which the intro waits on.
        self._measured_gripper: dict[str, float] = {}

    @classmethod
    def from_scene(
        cls,
        scene: str = "cell",
        params: ClutchParams | None = None,
        tip_model: str = "hull",
        table_z: float | None = None,
        intro: str | None = None,
        intro_params: IntroParams | None = None,
    ) -> Clutch:
        from dora_openarm_teleop import model_info

        def fk(qpos16: np.ndarray) -> Mapping[str, np.ndarray]:
            return model_info.ee_poses_from_qpos(qpos16, scene)

        return cls(
            arm_models_from_scene(scene, tip_model, table_z), params, fk, intro, intro_params
        )

    def on_pose(self, side: str, now: float, values: Any) -> bool:
        return self.arms[side].on_pose(now, values)

    def on_grip(self, side: str, now: float, value: Any) -> bool:
        return self.arms[side].on_grip(now, value)

    def reset(self) -> None:
        """control=reset: every arm holds home with the gripper closed, locked.

        An intro under way is dropped: the arm is no longer where it started.
        """
        self._intros.clear()
        for arm in self.arms.values():
            arm.reset()

    def proceed(self) -> bool:
        """control=proceed (the left controller's X, or the UI's Enter): an intro
        waiting for it moves on.

        Returns False when no intro was waiting.
        """
        return any(intro.proceed() for intro in self._intros.values())

    def restart_intro(self) -> list[str]:
        """An episode ended: every arm goes back to its start pose from wherever the
        hand left it, and the intro side then runs its intro again.

        Returns the sides that started back. An arm before its first start, or
        still in its intro or on its way back, is left alone.
        """
        started = []
        for side, arm in self.arms.items():
            if side not in self._start_poses or side in self._intros:
                continue
            if side == self.intro:
                self._intros[side] = Intro(
                    self._start_poses[side],
                    arm.model.gripper_range,
                    self.intro_params,
                    current=arm.hold,
                )
            else:
                self._intros[side] = Return(self._start_poses[side], arm.hold, self.intro_params)
            started.append(side)
        return started

    def intro_running(self, side: str) -> bool:
        return side in self._intros

    def reset_position(self, qpos16: Any) -> bool:
        """reset_position (QPOS16): hold the FK pose of that configuration, locked.

        Locking here as well matters when the sim resets on its own: an arm
        engaged against the old anchor would otherwise drag the new pose along.
        Returns False (and changes nothing) for malformed input.
        """
        try:
            values = np.asarray(qpos16, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return False
        if values.shape != (16,) or not np.all(np.isfinite(values)):
            return False
        if self._fk is None:
            raise RuntimeError("Clutch was built without forward kinematics")
        poses = self._fk(values)
        self._intros.clear()
        for offset, side in ((0, "right"), (8, "left")):
            self.arms[side].reset(poses[side], values[offset + 7])
        return True

    def on_robot_state(self, side: str, qpos: Any, epoch: int | None = None) -> bool | None:
        """The robot's measured joints for one arm (8 values), and its follower's start_epoch.

        The first state of an arm, and the first after its follower restarted (a new
        start_epoch: Pause Arm then Start Arm moved it to the start pose), make that arm
        hold the FK pose of where it actually is, locked until the grip is released. So
        the IK starts at the arm instead of the scene's home pose, and the gate, which
        waits while the IK is more than jump_tol from the arm, engages on a robot whose
        start pose is not the scene's home (the Cell's J4 is 0.2 rad off it). Any other
        state is ignored: holding the measured pose all the time would let the hold
        follow the arm's sag under gravity, except for the gripper angle, which an
        intro waits on. The intro side starts its intro from the re-seeded hold.
        Returns True when the arm was re-seeded, False when the state was ignored,
        None for malformed input.
        """
        try:
            values = np.asarray(qpos, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.shape[0] < 8 or not np.all(np.isfinite(values[:8])):
            return None
        arm = self.arms[side]
        self._measured_gripper[side] = float(values[7])
        arm.on_measured_gripper(values[7])
        if side in self._robot_epochs and (epoch is None or epoch == self._robot_epochs[side]):
            return False
        self._robot_epochs[side] = epoch
        if self._fk is None:
            raise RuntimeError("Clutch was built without forward kinematics")
        qpos16 = np.zeros(16)
        offset = 0 if side == "right" else 8
        qpos16[offset : offset + 8] = values[:8]
        arm.follower_restarted()
        arm.reset(self._fk(qpos16)[side], values[7])
        self._start_poses[side] = arm.hold.copy()
        if side == self.intro:
            self._intros[side] = Intro(arm.hold, arm.model.gripper_range, self.intro_params)
        return True

    def tick(self, now: float) -> ClutchTick:
        targets, states, details = {}, {}, {}
        for side, arm in self.arms.items():
            intro = self._intros.get(side)
            if intro is not None:
                # The hand is ignored while an intro runs; its target goes
                # through the same clamps as a followed one.
                target = intro.step(now, self._measured_gripper.get(side))
                target = clamp_target(target, arm.model, self.params)
                if not intro.done:
                    targets[side] = target.astype(np.float32)
                    states[side], details[side] = intro.state, intro.detail
                    continue
                # Done: the clutch holds where the intro ended, locked until
                # the grip is released, and takes over from this tick on.
                del self._intros[side]
                arm.reset(target[:7], target[7])
            targets[side] = arm.step(now).astype(np.float32)
            states[side], details[side] = arm.state, arm.detail
        # An intro or a finger-down turn stays "setup" through most of its steps;
        # a new step is published (and logged) like a state change, so the log
        # says where it is.
        setup_details = {side: details[side] for side, state in states.items() if state == SETUP}
        changed = states != self._last_states or setup_details != self._last_setup_details
        self._last_states = states
        self._last_setup_details = setup_details
        return ClutchTick(targets, states, changed, details)

    def states(self) -> dict[str, str]:
        return {
            side: intro.state if (intro := self._intros.get(side)) else arm.state
            for side, arm in self.arms.items()
        }
