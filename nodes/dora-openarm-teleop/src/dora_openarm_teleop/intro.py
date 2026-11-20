"""An arm's opening sequence before the clutch takes it (docs/teleop.md, Intro).

Pure logic without dora, with time passed in, like ``dora_openarm_teleop.clutch``.

The sequence turns the end effector about its approach axis, levelled so that
the approach is horizontal, and works the gripper:

0. ``return`` (only when started with ``current``, after an episode): the end
   effector goes from ``current`` back to the start pose, gripper as it is.
1. ``close``: the gripper closes where the arm started. dora-openarm runs with
   ``--align-trigger gripper``: it only starts executing commands (ALIGNED)
   once the commanded gripper is closed, so nothing turns before the measured
   gripper has closed, which shows that the commands are reaching the arm.
2. ``present``: the end effector rolls until the finger that is normally on the
   right points down and the gap between the fingers is level. Then the
   gripper opens to ``open_fraction`` of its range.
3. ``wait``: until :meth:`Intro.proceed` (the left controller's X, or the UI's
   Enter: command ``proceed``).
4. ``grip``: the gripper closes, until it measures closed or
   ``close_timeout_s`` has passed (it may be closing on something).
5. ``flip``: the end effector rolls back through the start orientation and on
   until the finger that is normally on the left points down.

Then the clutch holds the last target, locked until the grip is released.

"Normally" is the start pose of dora-openarm (J4 at 90 degrees, the rest 0),
where the fingers point forward and the ee_control_point site's +y is the
robot's left (+y of arm_origin) for both arms. So the finger on the right is
the one on the site's -y side, and a roll that turns the site's +y up puts it
at the bottom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dora_openarm_teleop import frames

# Clutch states published while an intro runs (docs/teleop.md, HUD).
SETUP = "setup"
WAITING = "waiting"

RETURN = "return"
CLOSE = "close"
PRESENT = "present"
OPEN = "open"
WAIT = "wait"
GRIP = "grip"
FLIP = "flip"
DONE = "done"
# What each phase is doing, in order: the clutch logs it on a state change.
_DETAILS = {
    RETURN: "going back to the start pose",
    CLOSE: "closing the gripper so that the arm aligns",
    PRESENT: "turning the finger on the right down",
    OPEN: "opening the gripper",
    WAIT: "waiting for X (or Enter)",
    GRIP: "closing the gripper",
    FLIP: "turning the finger on the left down",
    DONE: "done",
}

# Roll about the levelled approach axis: +90 deg turns the site's +y up.
PRESENT_ROLL = math.pi / 2
FLIP_ROLL = -math.pi / 2

_WORLD_UP = np.array([0.0, 0.0, 1.0])
# arm_origin +x: forward, where the fingers point at dora-openarm's start pose.
_FORWARD = np.array([1.0, 0.0, 0.0])


@dataclass(frozen=True)
class IntroParams:
    """Tunables; docs/teleop.md (Intro) explains the defaults."""

    # Fraction of the gripper's range it opens to while presenting (0: closed).
    open_fraction: float = 0.5
    # rad/s, the average roll speed. Each roll eases in and out (smoothstep),
    # so it peaks at 1.5x this; 0.5 turns 90 deg in 3.1 s.
    turn_speed: float = 0.5
    # s, each gripper ramp (close, open, grip).
    gripper_s: float = 0.3
    # rad: a measured gripper this close to closed counts as closed.
    closed_tol: float = 0.15
    # s after the grip starts: flip even if the gripper does not measure closed,
    # since it may be closing on something.
    close_timeout_s: float = 2.0
    # m/s, the average speed of the way back to the start pose after an episode
    # (eased like the rolls, so it peaks at 1.5x this). The turn on the way
    # back keeps to turn_speed; whichever takes longer sets the time.
    return_speed: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.open_fraction <= 1.0:
            raise ValueError("open_fraction must be in [0, 1]")
        if not (math.isfinite(self.turn_speed) and self.turn_speed > 0.0):
            raise ValueError("turn_speed must be positive")
        if not (math.isfinite(self.gripper_s) and self.gripper_s > 0.0):
            raise ValueError("gripper_s must be positive")
        if not (math.isfinite(self.closed_tol) and self.closed_tol > 0.0):
            raise ValueError("closed_tol must be positive")
        if not (math.isfinite(self.close_timeout_s) and self.close_timeout_s >= self.gripper_s):
            raise ValueError("close_timeout_s must be at least gripper_s")
        if not (math.isfinite(self.return_speed) and self.return_speed > 0.0):
            raise ValueError("return_speed must be positive")


def level_approach(quat: np.ndarray) -> np.ndarray:
    """Unit horizontal direction the fingers point in (site -z), or forward if vertical."""
    approach = -frames.quat_to_matrix(quat)[:, 2]
    approach[2] = 0.0
    norm = float(np.linalg.norm(approach))
    if norm < 0.1:
        # Pointing nearly straight up or down: no horizontal direction to keep.
        return _FORWARD.copy()
    return approach / norm


def rolled(approach: np.ndarray, roll: float) -> np.ndarray:
    """Orientation (wxyz) with the fingers along ``approach``, rolled ``roll`` rad.

    At roll 0 the site's +y is horizontal, to the left of the approach; positive
    rolls turn it up (right-handed about the approach).
    """
    z = -np.asarray(approach, dtype=np.float64)
    left = np.cross(_WORLD_UP, approach)
    left /= np.linalg.norm(left)
    y = math.cos(roll) * left + math.sin(roll) * _WORLD_UP
    x = np.cross(y, z)
    return frames.matrix_to_quat(np.column_stack([x, y, z]))


# Within this angle of vertical the fingers point (nearly) straight up or down,
# where "the finger on the right is lowest" stops meaning anything.
FINGER_DOWN_BLEND = math.radians(20)


def finger_down(quat: np.ndarray, blend: float = FINGER_DOWN_BLEND) -> np.ndarray:
    """``quat`` rolled about its approach axis so the finger normally on the right is lowest.

    That finger is on the site's -y side, so the site's +y is turned as far
    towards up as the approach allows: all the way when the fingers point
    level, while the approach itself is kept. Within ``blend`` of pointing
    straight up or down the result fades back to ``quat``'s own roll, so a target
    passing near vertical does not spin the wrist.
    """
    rotation = frames.quat_to_matrix(quat)
    z = rotation[:, 2]
    up = _WORLD_UP - float(_WORLD_UP @ z) * z  # up, across the approach
    level = float(np.linalg.norm(up))  # sin(angle between the approach and vertical)
    y = rotation[:, 1]
    if level > 1e-9:
        weight = min(level / math.sin(blend), 1.0)
        mixed = weight * (up / level) + (1.0 - weight) * y
        # Opposite and equally weighted: the constrained roll wins.
        y = mixed if np.linalg.norm(mixed) > 1e-6 else up / level
    y = y - float(y @ z) * z
    y /= np.linalg.norm(y)
    return frames.matrix_to_quat(np.column_stack([np.cross(y, z), y, z]))


def ease(t: float) -> float:
    """Smoothstep: 0 to 1 with zero speed at both ends."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def return_pose(
    current: np.ndarray, start: np.ndarray, gripper: float, elapsed: float, params: IntroParams
) -> np.ndarray | None:
    """POSE8 on the way from ``current`` back to ``start`` (POSE7s), ``elapsed`` s
    in, or None once there.

    A straight line and a slerp, eased, at ``params.return_speed`` and
    ``params.turn_speed`` on average, whichever takes longer.
    """
    distance = float(np.linalg.norm(start[:3] - current[:3]))
    angle = float(frames.quat_angle(frames.quat_delta(current[3:7], start[3:7])))
    duration = max(distance / params.return_speed, angle / params.turn_speed)
    t = elapsed / duration if duration > 1e-9 else 1.0
    if t >= 1.0:
        return None
    s = ease(t)
    position = current[:3] + s * (start[:3] - current[:3])
    quat = frames.quat_slerp(current[3:7], start[3:7], s)
    return np.concatenate([position, frames.quat_normalize(quat), [gripper]])


class Return:
    """After an episode, an arm without an intro goes back to its start pose.

    It has the interface of :class:`Intro` that the clutch uses, so the clutch
    runs it the same way: the hand is ignored until it is done, and then the
    arm holds the start pose, locked until the grip is released.
    """

    def __init__(
        self, start: np.ndarray, current: np.ndarray, params: IntroParams | None = None
    ) -> None:
        self.params = params or IntroParams()
        start = np.asarray(start, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)
        for name, pose in (("start", start), ("current", current)):
            if pose.shape != (8,) or not np.all(np.isfinite(pose)):
                raise ValueError(f"{name} must be a finite POSE8")
        self._start = np.concatenate([start[:3], frames.quat_normalize(start[3:7])])
        self._current = np.concatenate([current[:3], frames.quat_normalize(current[3:7])])
        self._gripper = float(current[7])
        self._phase = RETURN
        self._since: float | None = None

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def done(self) -> bool:
        return self._phase == DONE

    @property
    def state(self) -> str:
        return SETUP

    @property
    def detail(self) -> str:
        return _DETAILS[self._phase]

    def proceed(self) -> bool:
        return False

    def step(self, now: float, measured_gripper: float | None = None) -> np.ndarray:
        if self._since is None:
            self._since = now
        pose = return_pose(
            self._current, self._start, self._gripper, now - self._since, self.params
        )
        if pose is None:
            self._phase = DONE
            return np.concatenate([self._start, [self._gripper]])
        return pose


class Intro:
    """One arm's opening sequence, from its start pose to the pose the clutch holds."""

    def __init__(
        self,
        start: np.ndarray,
        gripper_range: tuple[float, float],
        params: IntroParams | None = None,
        current: np.ndarray | None = None,
    ) -> None:
        """``start``: the pose the sequence runs at (the arm's start pose).

        ``current``: where the arm is now, if not at ``start`` (after an
        episode). The sequence then begins by going back to ``start``.
        """
        self.params = params or IntroParams()
        start = np.asarray(start, dtype=np.float64)
        if start.shape != (8,) or not np.all(np.isfinite(start)):
            raise ValueError("start must be a finite POSE8")
        if current is not None:
            current = np.asarray(current, dtype=np.float64)
            if current.shape != (8,) or not np.all(np.isfinite(current)):
                raise ValueError("current must be a finite POSE8")
        self._position = start[:3].copy()
        self._start_quat = frames.quat_normalize(start[3:7])
        low, high = gripper_range
        # 0 is closed (model_info.gripper_ranges); the far end of the range is open.
        self._closed = min(max(0.0, low), high)
        far = high if abs(high) >= abs(low) else low
        self._open = self._closed + self.params.open_fraction * (far - self._closed)
        approach = level_approach(self._start_quat)
        self._approach = approach
        self._present_quat = rolled(approach, PRESENT_ROLL)
        self._flip_quat = rolled(approach, FLIP_ROLL)
        if current is None:
            self._start_gripper = min(max(float(start[7]), low), high)
            self._phase = CLOSE
        else:
            self._from_position = current[:3].copy()
            self._from_quat = frames.quat_normalize(current[3:7])
            self._start_gripper = min(max(float(current[7]), low), high)
            self._phase = RETURN
        self._since: float | None = None
        self._note = ""

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def done(self) -> bool:
        return self._phase == DONE

    @property
    def state(self) -> str:
        """The clutch state to publish: ``waiting`` for X or Enter, ``setup`` otherwise."""
        return WAITING if self._phase == WAIT else SETUP

    @property
    def detail(self) -> str:
        return self._note or _DETAILS[self._phase]

    def proceed(self) -> bool:
        """X or Enter: leave ``wait``. Returns False (and does nothing) in any other phase."""
        if self._phase != WAIT:
            return False
        self._enter(GRIP)
        return True

    def step(self, now: float, measured_gripper: float | None) -> np.ndarray:
        """Advance at ``now`` and return the target (POSE8 float64).

        ``measured_gripper`` is the arm's latest measured gripper angle, or None
        before the first state.
        """
        if self._since is None:
            self._since = now
        self._note = ""
        # A phase that ends on this tick hands the same tick to the next one, so
        # a sequence of instant phases (no roll needed, say) takes no ticks.
        for _ in _DETAILS:
            target = self._target(now - self._since, measured_gripper)
            if target is not None:
                return target
            self._since = now
        raise AssertionError("intro phases did not settle")

    def _target(self, elapsed: float, measured: float | None) -> np.ndarray | None:
        """This phase's target, or None after moving on to the next phase."""
        phase = self._phase
        gripper_t = elapsed / self.params.gripper_s
        if phase == RETURN:
            pose = return_pose(
                np.concatenate([self._from_position, self._from_quat]),
                np.concatenate([self._position, self._start_quat]),
                self._start_gripper,
                elapsed,
                self.params,
            )
            return self._enter(CLOSE) if pose is None else pose
        if phase == CLOSE:
            gripper = self._ramp(self._start_gripper, self._closed, gripper_t)
            if gripper_t >= 1.0 and self._measured_closed(measured):
                return self._enter(PRESENT)
            if gripper_t >= 1.0:
                self._note = "waiting for the arm to close the gripper (ALIGNED)"
            return self._pose(self._start_quat, gripper)
        if phase == PRESENT:
            angle = float(
                frames.quat_angle(frames.quat_delta(self._start_quat, self._present_quat))
            )
            t = elapsed * self.params.turn_speed / angle if angle > 1e-9 else 1.0
            if t >= 1.0:
                return self._enter(OPEN)
            quat = frames.quat_slerp(self._start_quat, self._present_quat, ease(t))
            return self._pose(quat, self._closed)
        if phase == OPEN:
            if gripper_t >= 1.0:
                return self._enter(WAIT)
            return self._pose(self._present_quat, self._ramp(self._closed, self._open, gripper_t))
        if phase == WAIT:
            return self._pose(self._present_quat, self._open)
        if phase == GRIP:
            gripper = self._ramp(self._open, self._closed, gripper_t)
            if gripper_t >= 1.0 and (
                self._measured_closed(measured) or elapsed >= self.params.close_timeout_s
            ):
                return self._enter(FLIP)
            return self._pose(self._present_quat, gripper)
        if phase == FLIP:
            turn = PRESENT_ROLL - FLIP_ROLL
            t = elapsed * self.params.turn_speed / turn
            if t >= 1.0:
                return self._enter(DONE)
            # By roll angle, not slerp: the two ends are 180 deg apart, and only
            # the way back through the start orientation stays inside J5's range.
            roll = PRESENT_ROLL + (FLIP_ROLL - PRESENT_ROLL) * ease(t)
            return self._pose(rolled(self._approach, roll), self._closed)
        return self._pose(self._flip_quat, self._closed)

    def _enter(self, phase: str) -> None:
        self._phase = phase
        self._since = None

    def _measured_closed(self, measured: float | None) -> bool:
        return (
            measured is not None and abs(float(measured) - self._closed) <= self.params.closed_tol
        )

    @staticmethod
    def _ramp(start: float, end: float, t: float) -> float:
        t = min(max(t, 0.0), 1.0)
        return start + (end - start) * t

    def _pose(self, quat: np.ndarray, gripper: float) -> np.ndarray:
        return np.concatenate([self._position, frames.quat_normalize(quat), [gripper]])
