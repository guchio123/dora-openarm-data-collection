"""Clutch logic (docs/teleop.md, Clutch) and the dora-openarm-clutch event loop, with injected time."""

from __future__ import annotations

import heapq
import json
import math
from collections.abc import Iterator
from itertools import chain

import numpy as np
import pyarrow as pa
import pytest

from dora_openarm_teleop import arrow, clutch_node, frames, model_info, timing
from dora_openarm_teleop.clutch import (
    ENGAGED,
    LOCKED,
    RELEASED,
    STALE,
    Clutch,
    ClutchParams,
    arm_models_from_scene,
    table_lift,
)
from dora_openarm_teleop.intro import Intro, IntroParams

DT = 0.005  # 200 Hz tick
NO_CLAMPS = ClutchParams(clamp_table=False, clamp_reach=False, clamp_midline=False)


def _no_resume() -> ClutchParams:
    """The clutch before CLUTCH-3: every gap costs the grip."""
    return ClutchParams(clamp_table=False, clamp_reach=False, clamp_midline=False, resume_gap_s=0.0)


# Ticks for a still pose to settle after the start (or a gap): settle_s plus the speed window.
SETTLE_TICKS = round((ClutchParams().settle_s + ClutchParams().speed_window_s) / DT) + 2


@pytest.fixture(scope="module")
def hull_models():
    return arm_models_from_scene("cell", "hull")


@pytest.fixture(scope="module")
def home():
    return model_info.home_ee_poses()


def hand(position, quat=(1.0, 0.0, 0.0, 0.0), gripper=0.0) -> np.ndarray:
    return np.array([*position, *quat, gripper], dtype=np.float32)


class Rig:
    """Feeds both controllers and ticks at 200 Hz with a fake monotonic clock."""

    def __init__(self, clutch: Clutch) -> None:
        self.clutch = clutch
        self.t = 100.0
        self.hands = {"right": hand((0.3, -0.2, 0.0)), "left": hand((0.3, 0.2, 0.0))}
        self.grips = {"right": 0.0, "left": 0.0}
        self.live = {"right": True, "left": True}

    def step(self, n: int = 1):
        result = None
        for _ in range(n):
            self.t += DT
            for side in ("right", "left"):
                if self.live[side]:
                    assert self.clutch.on_pose(side, self.t, self.hands[side])
                    assert self.clutch.on_grip(side, self.t, self.grips[side])
            result = self.clutch.tick(self.t)
        return result

    def move(self, side: str, delta, n: int = 50):
        """Move a hand by `delta` (m) linearly over n ticks."""
        start = self.hands[side][:3].copy()
        result = None
        for i in range(1, n + 1):
            self.hands[side][:3] = start + np.asarray(delta) * i / n
            result = self.step()
        return result

    def engage(self, side: str):
        self.grips[side] = 0.0
        self.step(2)
        self.grips[side] = 1.0
        return self.step()


def _rig(models, params=NO_CLAMPS, settle: bool = True) -> Rig:
    """A rig whose still hands have already settled (both arms released), unless settle=False."""
    rig = Rig(Clutch(models, params, fk=model_info.ee_poses_from_qpos))
    if settle:
        result = rig.step(SETTLE_TICKS)
        assert result.states == {"right": RELEASED, "left": RELEASED}
    return rig


def _pose_equal(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> None:
    np.testing.assert_allclose(a[:3], b[:3], atol=atol)
    assert abs(abs(float(np.dot(a[3:7], b[3:7]))) - 1.0) < atol


# --- states and holding -----------------------------------------------------


def test_starts_stale_holding_home(hull_models, home) -> None:
    clutch = Clutch(hull_models)
    first = clutch.tick(1.0)
    assert first.states == {"right": STALE, "left": STALE}
    assert first.changed
    for side in ("right", "left"):
        assert first.targets[side].dtype == np.float32
        _pose_equal(first.targets[side].astype(float), home[side])
        assert first.targets[side][7] == 0.0
    assert not clutch.tick(1.005).changed


def test_no_motion_while_released_even_if_hand_moves_20cm(hull_models, home) -> None:
    rig = _rig(hull_models)
    before = rig.step(3)
    assert before.states["right"] == RELEASED
    for delta in ([0.2, 0, 0], [0, -0.2, 0], [0, 0, -0.2]):
        rig.hands["right"][3:7] = frames.quat_from_axis_angle([0, 0, 1], 0.8)
        result = rig.move("right", delta)
        assert result.states["right"] == RELEASED
        np.testing.assert_array_equal(result.targets["right"], before.targets["right"])
    _pose_equal(result.targets["right"].astype(float), home["right"])


def test_grip_held_at_start_is_locked_until_released_and_repressed(hull_models) -> None:
    rig = _rig(hull_models, settle=False)
    rig.grips["right"] = 1.0
    held = rig.step(5)
    assert held.states["right"] == LOCKED
    result = rig.move("right", [0.1, 0, 0])
    assert result.states["right"] == LOCKED
    np.testing.assert_array_equal(result.targets["right"], held.targets["right"])
    result = rig.step(SETTLE_TICKS)  # settled, but the grip is still squeezed
    assert result.states["right"] == LOCKED
    rig.grips["right"] = 0.0
    assert rig.step().states["right"] == RELEASED
    rig.grips["right"] = 1.0
    assert rig.step().states["right"] == ENGAGED


def test_hysteresis(hull_models) -> None:
    rig = _rig(hull_models)
    rig.step()
    for grip, expected in (
        (0.6, RELEASED),
        (0.7, ENGAGED),
        (0.5, ENGAGED),
        (0.4, ENGAGED),
        (0.39, RELEASED),
        (0.69, RELEASED),
    ):
        rig.grips["right"] = grip
        assert rig.step().states["right"] == expected, grip


def test_engage_produces_no_jump(hull_models) -> None:
    rig = _rig(hull_models, ClutchParams())
    held = rig.step(3).targets["right"].copy()
    # The hand is far from the arm and turned: absolute mapping would jump.
    rig.hands["right"] = hand((0.6, 0.3, 0.4), frames.quat_from_axis_angle([1, 1, 0], 2.0), -0.5)
    rig.step()
    rig.grips["right"] = 1.0
    first = rig.step()
    assert first.states["right"] == ENGAGED
    np.testing.assert_allclose(first.targets["right"], held, atol=1e-6)  # gripper too
    np.testing.assert_allclose(rig.step().targets["right"], held, atol=1e-6)


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_relative_translation_and_world_frame_rotation(hull_models, scale: float) -> None:
    params = ClutchParams(clamp_table=False, clamp_reach=False, clamp_midline=False, scale=scale)
    rig = _rig(hull_models, params)
    q0 = frames.quat_from_axis_angle([0.3, -1.0, 0.2], 1.1)
    rig.hands["right"] = hand((0.1, -0.3, 0.2), q0)
    held = rig.engage("right").targets["right"].astype(float)

    turn = frames.quat_from_axis_angle([0, 0, 1], np.deg2rad(30))
    q1 = frames.quat_multiply(turn, q0)
    rig.hands["right"][3:7] = q1
    result = rig.move("right", [0.05, -0.08, 0.12])
    target = result.targets["right"].astype(float)

    np.testing.assert_allclose(
        target[:3], held[:3] + scale * np.array([0.05, -0.08, 0.12]), atol=1e-6
    )
    # Turning the hand 30° about world z turns the gripper 30° about world z,
    # whatever the hand's and the gripper's own orientations are.
    expected_rot = frames.quat_to_matrix(turn) @ frames.quat_to_matrix(held[3:7])
    np.testing.assert_allclose(frames.quat_to_matrix(target[3:7]), expected_rot, atol=1e-5)


def test_release_holds_and_reengage_continues_from_hold(hull_models, home) -> None:
    rig = _rig(hull_models)
    rig.engage("right")
    rig.move("right", [0.10, 0, 0])
    rig.grips["right"] = 0.0
    released = rig.step().targets["right"].copy()
    np.testing.assert_allclose(released[:3], home["right"][:3] + [0.10, 0, 0], atol=1e-6)
    rig.move("right", [-0.10, 0, 0])
    np.testing.assert_array_equal(rig.step().targets["right"], released)
    rig.grips["right"] = 1.0
    rig.step()
    result = rig.move("right", [0.05, 0, 0])
    np.testing.assert_allclose(
        result.targets["right"][:3], home["right"][:3] + [0.15, 0, 0], atol=1e-6
    )


# --- staleness ----------------------------------------------------------------


def test_one_side_stale_while_other_keeps_tracking(hull_models, home) -> None:
    rig = _rig(hull_models)
    rig.engage("right")
    rig.engage("left")
    rig.move("left", [0.05, 0, 0], n=10)
    left_held = rig.step().targets["left"].copy()

    rig.live["left"] = False  # left controller lost; the right keeps moving
    start = rig.hands["right"][:3].copy()
    right_start = rig.step().targets["right"][:3].astype(float)
    states = []
    for i in range(1, 101):
        rig.hands["right"][:3] = start + [0.0, 0.0, 0.001 * i]
        result = rig.step()
        states.append(result.states["left"])
        np.testing.assert_array_equal(result.targets["left"], left_held)
        np.testing.assert_allclose(
            result.targets["right"][:3], right_start + [0.0, 0.0, 0.001 * i], atol=1e-6
        )
        assert result.states["right"] == ENGAGED
    assert states[0] == ENGAGED  # still within the 100 ms window
    assert states[-1] == STALE

    # Tracking returns with the grip still squeezed and the hand elsewhere: locked, no jump.
    rig.live["left"] = True
    rig.hands["left"][:3] += [0.2, 0.1, -0.1]
    result = rig.step()
    assert result.states["left"] == LOCKED
    np.testing.assert_array_equal(result.targets["left"], left_held)
    rig.grips["left"] = 0.0
    assert rig.step().states["left"] == LOCKED  # released, but the pose is still settling
    result = rig.step(SETTLE_TICKS)
    assert result.states["left"] == RELEASED
    rig.grips["left"] = 1.0
    result = rig.step()
    assert result.states["left"] == ENGAGED
    np.testing.assert_allclose(result.targets["left"][:7], left_held[:7], atol=1e-6)


def test_gap_between_samples_disarms_even_without_ticks(hull_models) -> None:
    rig = _rig(hull_models)
    rig.engage("right")
    held = rig.step().targets["right"].copy()
    rig.t += 0.15  # the node was starved: no tick saw the silence
    rig.hands["right"][:3] += [0.1, 0, 0]
    result = rig.step()
    assert result.states["right"] == LOCKED
    np.testing.assert_array_equal(result.targets["right"], held)


# --- a gap must not cost the operator their grip (CLUTCH-3) ---------------------


def _gap_while_following(rig: Rig, side: str, gap_s: float, move=(0.0, 0.0, 0.0)):
    """Engage, then lose every input for gap_s while the hand moves `move`.

    Returns (held target before the gap, the first result after the input is back).
    """
    rig.engage(side)
    held = rig.step().targets[side].copy()
    assert rig.clutch.states()[side] == ENGAGED
    rig.live[side] = False
    rig.step(round(gap_s / DT))
    assert rig.clutch.states()[side] == STALE
    rig.hands[side][:3] += move
    rig.live[side] = True
    return held, rig.step()


def test_a_short_gap_with_the_hand_where_it_was_keeps_the_grip(hull_models) -> None:
    """The reported case: the grip is still squeezed, so a burst of lost frames must not
    make the operator release and re-squeeze. One tick after the input is back the arm
    follows again, from the pose that came back, with the target where it was."""
    rig = _rig(hull_models)
    held, result = _gap_while_following(rig, "right", 0.15)
    assert result.states["right"] == ENGAGED
    np.testing.assert_allclose(result.targets["right"], held, atol=1e-6)
    # Following again, not merely labelled ENGAGED.
    moved = rig.move("right", [0.05, 0.0, 0.0])
    assert moved.states["right"] == ENGAGED
    np.testing.assert_allclose(moved.targets["right"][:3], held[:3] + [0.05, 0, 0], atol=1e-6)


def test_the_hand_travel_a_bridged_gap_hid_is_dropped_not_jumped(hull_models) -> None:
    """The anchor is re-based on the pose that came back, so the up-to-2 cm the clutch
    could not see is dropped rather than applied to the arm in one tick."""
    rig = _rig(hull_models)
    held, result = _gap_while_following(rig, "right", 0.15, move=(0.015, 0.0, 0.0))
    assert result.states["right"] == ENGAGED
    np.testing.assert_allclose(result.targets["right"], held, atol=1e-6)


@pytest.mark.parametrize(
    ("gap_s", "move"),
    [
        (0.6, (0.0, 0.0, 0.0)),  # longer than resume_gap_s, even with the hand still
        (0.15, (0.05, 0.0, 0.0)),  # the pose came back 5 cm away
        (0.15, (0.0, 0.0, 0.03)),  # and 3 cm away
    ],
)
def test_a_gap_the_clutch_cannot_bridge_still_needs_a_release(hull_models, gap_s, move) -> None:
    """CLUTCH-1 unchanged: the arm lets go, holds, and stays locked while the grip is
    held - a release below release_grip plus the settle is the only way back."""
    rig = _rig(hull_models)
    held, result = _gap_while_following(rig, "right", gap_s, move=move)
    assert result.states["right"] == LOCKED
    np.testing.assert_array_equal(result.targets["right"], held)
    result = rig.step(SETTLE_TICKS)  # holding the grip is never enough
    assert result.states["right"] == LOCKED
    np.testing.assert_array_equal(result.targets["right"], held)
    rig.grips["right"] = 0.0
    assert rig.step(SETTLE_TICKS).states["right"] == RELEASED
    rig.grips["right"] = 1.0
    result = rig.step()
    assert result.states["right"] == ENGAGED
    np.testing.assert_allclose(result.targets["right"], held, atol=1e-6)


def test_a_turn_during_the_gap_is_bridged_only_while_it_is_small(hull_models) -> None:
    params = ClutchParams(clamp_table=False, clamp_reach=False, clamp_midline=False)
    for turn, expected in ((0.5 * params.resume_max_turn, ENGAGED), (10.0 * 0.087, LOCKED)):
        rig = _rig(hull_models, params)
        rig.engage("right")
        held = rig.step().targets["right"].copy()
        q0 = rig.hands["right"][3:7].copy()
        rig.live["right"] = False
        rig.step(round(0.15 / DT))
        rig.hands["right"][3:7] = frames.quat_multiply(
            frames.quat_from_axis_angle([0, 0, 1], turn), q0
        )
        rig.live["right"] = True
        result = rig.step()
        assert result.states["right"] == expected, turn
        np.testing.assert_allclose(result.targets["right"], held, atol=1e-6)


def test_resuming_can_be_disabled(hull_models) -> None:
    """--resume-gap-ms 0 is the behaviour before CLUTCH-3: every gap costs the grip."""
    params = ClutchParams(
        clamp_table=False, clamp_reach=False, clamp_midline=False, resume_gap_s=0.0
    )
    rig = _rig(hull_models, params)
    held, result = _gap_while_following(rig, "right", 0.15)
    assert result.states["right"] == LOCKED
    np.testing.assert_array_equal(result.targets["right"], held)


def test_letting_go_during_a_bridged_gap_still_settles_before_the_next_engagement(
    hull_models,
) -> None:
    """The engagement is taken back, not the arming: a NEW anchor after the gap still
    needs the settle, because that is what CLUTCH-1 is about."""
    rig = _rig(hull_models)
    rig.engage("right")
    held = rig.step().targets["right"].copy()
    rig.live["right"] = False
    rig.step(round(0.15 / DT))
    rig.grips["right"] = 0.0  # let go while the input was away
    rig.live["right"] = True
    result = rig.step()
    assert result.states["right"] == LOCKED
    assert result.details["right"] == "let go during the gap"
    rig.grips["right"] = 1.0
    result = rig.step(round(ClutchParams().settle_s / DT) - 4)
    assert result.states["right"] == LOCKED  # a squeeze during settling needs a release
    rig.grips["right"] = 0.0
    rig.step(SETTLE_TICKS)
    rig.grips["right"] = 1.0
    result = rig.step()
    assert result.states["right"] == ENGAGED
    np.testing.assert_allclose(result.targets["right"], held, atol=1e-6)


def test_a_bridged_gap_does_not_drop_what_the_gripper_holds(hull_models, home) -> None:
    """CLUTCH-2 survives CLUTCH-3. The trigger is not in the closeness test (only the
    hand's position and orientation are), so a trigger that moved while the input was
    away must not open a closed gripper on the tick the engagement comes back."""
    rig = _rig(hull_models)
    rig.engage("right")
    assert rig.step().targets["right"][7] == pytest.approx(0.0)  # closed, holding
    rig.live["right"] = False
    rig.step(round(0.15 / DT))
    rig.hands["right"][7] = -0.7854  # the trigger relaxed while the frames were away
    rig.live["right"] = True
    result = rig.step()
    assert result.states["right"] == ENGAGED
    assert result.targets["right"][7] == pytest.approx(0.0), "the object was dropped"
    rig.hands["right"][7] = -0.02  # back within gripper_pickup of the held angle
    assert rig.step().targets["right"][7] == pytest.approx(-0.02)


def test_only_a_pose_from_after_the_gap_can_be_resumed_on(hull_models) -> None:
    """webxr republishes its cached pose for POSE_TIMEOUT after the frames stop, and that
    pose is not the hand. The resume waits for a pose delivered after the input was lost,
    so a grip that comes back first cannot re-base the anchor on the cache."""
    clutch = Clutch(hull_models, NO_CLAMPS)
    t = 100.0
    pose = hand((0.3, -0.2, 0.0))
    for i in range(SETTLE_TICKS + 2):
        t += DT
        clutch.on_pose("right", t, pose)
        clutch.on_grip("right", t, 1.0 if i >= SETTLE_TICKS else 0.0)
        result = clutch.tick(t)
    assert result.states["right"] == ENGAGED
    held = result.targets["right"].copy()
    # The frames stop. webxr keeps republishing the cached pose for POSE_TIMEOUT.
    for _ in range(round(0.09 / DT)):
        t += DT
        clutch.on_pose("right", t, pose)
        clutch.tick(t)
    silent = t
    t += 0.06  # the frame that ends the stall: its grip is delivered first
    clutch.on_grip("right", t, 1.0)
    assert clutch.tick(t).states["right"] == STALE, "no pose since the gap: nothing to resume on"
    t += 0.002  # the pose from that frame, 4 cm from the cached one
    clutch.on_pose("right", t, hand((0.34, -0.2, 0.0)))
    result = clutch.tick(t)
    assert result.states["right"] == LOCKED  # the hand really moved: today's behaviour
    np.testing.assert_array_equal(result.targets["right"], held)
    # The pose stream itself never gapped (POSE_TIMEOUT < fresh_s hid it), so the
    # suspension and this refusal were driven entirely by the grip.
    assert t - silent < ClutchParams().fresh_s
    assert not clutch.arms["right"]._suspended


def test_grip_alone_is_not_fresh(hull_models) -> None:
    clutch = Clutch(hull_models, NO_CLAMPS)
    for i in range(10):
        clutch.on_grip("right", 1.0 + i * DT, 1.0)
        assert clutch.tick(1.0 + i * DT).states["right"] == STALE


# --- settling after a pose gap (CLUTCH-1) --------------------------------------


def _lose_and_return(rig: Rig, side: str, lost_s: float = 0.2):
    """Controller untracked for lost_s, then back at the same place, grip released."""
    rig.live[side] = False
    rig.step(round(lost_s / DT))
    assert rig.clutch.states()[side] == STALE
    rig.live[side] = True
    rig.grips[side] = 0.0


def test_gap_locks_until_the_pose_has_been_fresh_for_settle_s(hull_models) -> None:
    rig = _rig(hull_models)
    _lose_and_return(rig, "left")
    held = rig.step().targets["left"].copy()
    ticks = 1
    while rig.clutch.states()["left"] == LOCKED:
        np.testing.assert_array_equal(rig.step().targets["left"], held)
        ticks += 1
        assert ticks < 1000
    assert rig.clutch.states()["left"] == RELEASED
    # The still pose has settled once settle_s has passed since it came back (tick 1).
    assert (ticks - 1) * DT == pytest.approx(ClutchParams().settle_s, abs=DT + 1e-9)


def test_moving_pose_after_a_gap_stays_locked_until_at_rest(hull_models) -> None:
    rig = _rig(hull_models)
    _lose_and_return(rig, "right")
    held = rig.step().targets["right"].copy()
    # A smoother still sliding towards the real hand at 10 cm/s, for a whole second.
    result = rig.move("right", [0.10, 0.0, 0.0], n=200)
    assert result.states["right"] == LOCKED
    np.testing.assert_array_equal(result.targets["right"], held)
    rig.grips["right"] = 1.0  # a squeeze while locked needs a release afterwards
    rig.step()
    rig.grips["right"] = 0.0
    window = round(ClutchParams().speed_window_s / DT)
    states = [rig.step().states["right"] for _ in range(window + 2)]
    assert states[0] == LOCKED  # the last 50 ms still contain motion
    assert states[-1] == RELEASED


def test_turning_pose_after_a_gap_stays_locked_until_at_rest(hull_models) -> None:
    rig = _rig(hull_models)
    _lose_and_return(rig, "right")
    q0 = rig.hands["right"][3:7].copy()
    for i in range(1, 121):  # 1 rad/s about z for 0.6 s, position still
        rig.hands["right"][3:7] = frames.quat_multiply(
            frames.quat_from_axis_angle([0, 0, 1], i * DT), q0
        )
        assert rig.step().states["right"] == LOCKED
    result = rig.step(round(ClutchParams().speed_window_s / DT) + 2)
    assert result.states["right"] == RELEASED


def test_settled_arm_engages_while_the_hand_moves(hull_models) -> None:
    """Settling is only checked after a gap: normal reach-and-squeeze is unaffected."""
    rig = _rig(hull_models)
    rig.grips["right"] = 0.0
    start = rig.hands["right"][:3].copy()
    for i in range(1, 41):  # 0.4 m/s, squeezing half-way
        rig.hands["right"][:3] = start + [0.002 * i, 0.0, 0.0]
        rig.grips["right"] = 1.0 if i > 20 else 0.0
        result = rig.step()
    assert result.states["right"] == ENGAGED


def test_settling_can_be_disabled(hull_models) -> None:
    params = ClutchParams(
        clamp_table=False,
        clamp_reach=False,
        clamp_midline=False,
        settle_s=0.0,
        engage_max_speed=float("inf"),
        engage_max_angular_speed=float("inf"),
    )
    rig = _rig(hull_models, params, settle=False)
    assert rig.step().states["right"] == RELEASED


# The webxr node's smoother settings (dora_openarm_webxr.main: ONE_EURO_MIN_CUTOFF 2.0,
# ONE_EURO_BETA 0.04, d_cutoff 1.5, MAX_LINEAR_SPEED 1.0, MAX_ANGULAR_SPEED 6.0).
_SMOOTHER = {
    "min_cutoff": 2.0,
    "beta": 0.04,
    "d_cutoff": 1.5,
    "max_linear_speed": 1.0,
    "max_angular_speed": 6.0,
}


def _replay_smoothed_return(offset, turn: float, grip_at, lost_s: float = 1.0):
    """The review's smoother_released.py / smoother_reengage.py harness.

    The vendored webxr smoother keeps its pre-gap pose through the gap (suspend(), the
    behaviour before SMOOTHER_RESET_GAP), so when tracking returns the published pose
    slides towards the real, motionless hand. ``grip_at(ms since tracking returned)``
    scripts the grip. Returns (ms of engagement or None, max position drift of the
    target after engaging in m, max orientation drift in rad).
    """
    smoothing = pytest.importorskip("dora_openarm_webxr.smoothing")
    clutch = Clutch(arm_models_from_scene("cell", "point"), NO_CLAMPS)
    smoother = smoothing.OneEuroPoseSmoother(**_SMOOTHER)
    a = np.array([0.2, -0.15, -0.1, 1.0, 0.0, 0.0, 0.0])
    b = a.copy()
    b[:3] += offset
    b[3:7] = frames.quat_from_axis_angle([0.0, 0.0, 1.0], turn)
    left = hand((0.2, 0.15, -0.1))
    t = 100.0

    def tick(tracked: bool, pose: np.ndarray, grip: float):
        nonlocal t
        t += DT
        if tracked:
            smoothed = smoother.smooth(t, pose)
            clutch.on_pose("right", t, np.concatenate([smoothed, [0.0]]))
            clutch.on_grip("right", t, grip)
        else:
            smoother.suspend(t)  # webxr omits pose, trigger and grip
        clutch.on_pose("left", t, left)
        clutch.on_grip("left", t, 0.0)
        return clutch.tick(t)

    for _ in range(100):
        tick(True, a, 0.0)
    for _ in range(200):
        result = tick(True, a, 1.0)
    assert result.states["right"] == ENGAGED
    for _ in range(round(lost_s / DT)):
        result = tick(False, a, 1.0)
    assert result.states["right"] == STALE

    engaged_ms, reference, drift, turned = None, None, 0.0, 0.0
    for k in range(400):  # 2 s with the hand still at b
        ms = k * DT * 1000.0
        result = tick(True, b, grip_at(ms))
        target = result.targets["right"].astype(np.float64)
        if result.states["right"] == ENGAGED:
            if engaged_ms is None:
                engaged_ms, reference = ms, target.copy()
            drift = max(drift, float(np.linalg.norm(target[:3] - reference[:3])))
            delta = frames.quat_delta(reference[3:7], target[3:7])
            turned = max(turned, float(frames.quat_angle(delta)))
    return engaged_ms, drift, turned


def _squeeze_at(squeeze_ms: float):
    return lambda ms: 1.0 if ms >= squeeze_ms else 0.0


def _regrip_at(regrip_ms: float):
    # Grip held through the loss, released for 100 ms, squeezed again at regrip_ms.
    return lambda ms: 0.0 if regrip_ms - 100.0 <= ms < regrip_ms else 1.0


@pytest.mark.parametrize(
    ("offset", "grip", "must_engage"),
    [
        # smoother_released.py: let go while untracked, squeeze soon after it returns
        # (the review measured 245, 195 and 48 mm of drift with the hand still).
        ([0.0, -0.30, 0.0], 50, False),
        ([0.0, -0.30, 0.0], 100, False),
        ([0.0, -0.30, 0.0], 200, False),
        ([0.0, -0.15, 0.0], 100, False),
        ([0.0, -0.30, 0.0], 600, True),
        ([0.0, -0.30, 0.0], 800, True),
        # smoother_reengage.py: held through the loss, release and re-squeeze
        # (40 cm, re-grip at 250 ms: 145 mm at up to 1 m/s).
        ([0.40, 0.0, 0.0], -150, False),
        ([0.40, 0.0, 0.0], -250, False),
        ([0.20, 0.0, 0.0], -400, False),
        ([0.40, 0.0, 0.0], -600, False),
        ([0.10, 0.0, 0.0], -600, True),
        ([0.40, 0.0, 0.0], -1000, True),
    ],
)
def test_reengaging_on_a_sliding_smoothed_pose_does_not_move_the_target(
    offset, grip: float, must_engage: bool
) -> None:
    grip_at = _squeeze_at(grip) if grip > 0 else _regrip_at(-grip)
    engaged_ms, drift, _ = _replay_smoothed_return(np.array(offset), 0.0, grip_at)
    assert drift < 0.005, f"target drifted {drift * 1000:.0f} mm with the hand still"
    if must_engage:
        assert engaged_ms is not None  # settling delays engagement, it does not forbid it


@pytest.mark.parametrize("offset_m", [0.0, 0.05])
def test_a_resumed_arm_follows_a_suspended_smoother_no_further_than_the_hand_went(
    offset_m: float,
) -> None:
    """The one case where CLUTCH-3 meets a sliding smoother, with the real One Euro in the
    loop: a hand missing from frames that keep arriving, for longer than fresh_s but not
    longer than SMOOTHER_RESET_GAP, so webxr suspends its filter instead of resetting it
    and the first pose back is the pre-gap one. The engagement is kept (the closeness test
    cannot see the slide), and the arm then follows the slide - which is the hand's own
    displacement arriving late, bounded by SMOOTHER_RESET_GAP of hand travel, not the
    CLUTCH-1 drift of an anchor taken mid-slide. A still hand moves the target nothing."""
    engaged_ms, drift, _ = _replay_smoothed_return(
        np.array([0.0, -offset_m, 0.0]), 0.0, lambda _ms: 1.0, lost_s=0.11
    )
    assert engaged_ms is not None and engaged_ms < 10.0
    assert drift <= offset_m + 0.002, f"followed {drift * 1000:.0f} mm for a {offset_m} m move"
    if offset_m == 0.0:
        assert drift < 0.001


def test_reengaging_on_a_turning_smoothed_pose_does_not_turn_the_target() -> None:
    engaged_ms, drift, turned = _replay_smoothed_return(np.zeros(3), 1.5, _squeeze_at(600))
    assert engaged_ms is not None
    assert drift < 0.005 and turned < 0.01


# --- frame stalls through the real webxr node (P1 timing coupling) -------------------

# The coupling the launcher sets (docs/teleop.md, Clutch): webxr POSE_TIMEOUT <= clutch fresh_s
# < webxr SMOOTHER_RESET_GAP. POSE_TIMEOUT 0.2 was the value before.
P1_POSE_TIMEOUT = 0.1
P1_SMOOTHER_RESET_GAP = 0.12
OLD_POSE_TIMEOUT = 0.2
QUEST_FRAME_S = 1.0 / 72.0
POSE_TIMEOUT_S = timing.POSE_TIMEOUT_S


class _WebxrNode:
    def __init__(self, clock: list[float]) -> None:
        self.clock = clock
        self.sent: list[tuple[float, str, pa.Array]] = []

    def send_output(self, output_id, value, metadata=None) -> None:
        self.sent.append((self.clock[0], output_id, value))


class _NoHeadset:
    def send_control(self, payload) -> None:
        pass


def _stalled_frames_through_webxr(
    monkeypatch,
    models,
    pose_timeout: float,
    lost_frames: int,
    squeeze_ms: float,
    grip_delay_after_stall: float = 0.0,
    hold_grip: bool = False,
    move_m: float = 0.08,
    params: ClutchParams = NO_CLAMPS,
) -> dict:
    """Page frames -> vendored dora_openarm_webxr (_process_frame on arrival, _publish_poses
    on its own 200 Hz tick) -> 1 ms -> Clutch (its own 200 Hz tick, another phase).

    The operator holds the right grip with the hand still; then `lost_frames` consecutive
    frames are lost (Wi-Fi) while the hand moves `move_m` and the grip is let go. Frames
    come back with the hand still at the new place and the grip released, and it is
    squeezed `squeeze_ms` after the first frame back. With `hold_grip` the operator never
    lets go at all (the reported case: the grip is squeezed throughout), and `squeeze_ms`
    is ignored. Between frames webxr republishes the last frame's pose until POSE_TIMEOUT
    has passed, then publishes nothing. The first grip back reaches the clutch
    `grip_delay_after_stall` later than the usual 1 ms (dora delivery jitter between
    webxr's clock and the clutch's).
    """
    main = pytest.importorskip("dora_openarm_webxr.main")
    clock = [0.0]
    node = _WebxrNode(clock)
    monkeypatch.setattr(main, "node", node)
    monkeypatch.setattr(main, "webrtc_server", _NoHeadset())
    monkeypatch.setattr(main.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(main, "_POSE_TIMEOUT", pose_timeout)
    monkeypatch.setattr(main, "_SMOOTHER_RESET_GAP", P1_SMOOTHER_RESET_GAP)
    monkeypatch.setattr(main, "_PIVOT_MODE", "latched", raising=False)
    state = main._ConnectionState()
    clutch = Clutch(models, params)

    def wx(position) -> dict:
        x, y, z = (float(v) for v in position)
        return {"x": x, "y": y, "z": z, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

    head, before = (0.0, 1.6, 0.0), np.array([0.2, 1.1, -0.4])
    after = before + np.array([move_m, 0.0, 0.0])
    left = hand((0.2, 0.15, -0.1))
    t0, latency = 10.0, 0.001
    first_lost = 72  # frame 72 arrives 1 s in; the grip is held from 0.6 s
    stall_at = t0 + 0.003 + first_lost * QUEST_FRAME_S
    first_back = t0 + 0.003 + (first_lost + lost_frames) * QUEST_FRAME_S
    events: list[tuple[float, int, str, object]] = []
    for k in range(int(2.2 / QUEST_FRAME_S)):
        if not first_lost <= k < first_lost + lost_frames:
            events.append((t0 + 0.003 + k * QUEST_FRAME_S, 0, "frame", k))
    for j in range(int(2.2 / DT)):
        events.append((t0 + 0.0011 + j * DT, 1, "webxr_tick", None))
        events.append((t0 + 0.0037 + j * DT, 3, "clutch_tick", None))
    heapq.heapify(events)

    grips: list[float] = []  # clutch receive times of the right grip
    out: dict = {
        "engaged_ms": None,
        "drift": 0.0,
        "held_drift": 0.0,  # how far the target ever left the pre-stall one
        "pose_gap": False,
        "states": [],
        "details": [],
    }
    reference, held, was_free, sequence = None, np.zeros(3), False, 0
    while events:
        t, _, kind, payload = heapq.heappop(events)
        clock[0] = t
        if kind == "frame":
            back = payload >= first_lost
            if hold_grip:
                grip = 1.0 if t >= t0 + 0.6 else 0.0
            elif back:
                grip = 1.0 if t >= first_back + squeeze_ms / 1000.0 else 0.0
            else:
                grip = 1.0 if t >= t0 + 0.6 else 0.0
            sequence += 1
            start = len(node.sent)
            main._process_frame(
                {
                    "type": "frame",
                    "sequence": sequence,
                    "pose_reference": wx(head),
                    "pose_right": wx(after if back else before),
                    "trigger_right": 0.0,
                    "grip_right": grip,
                },
                state,
            )
            delay = latency + (
                grip_delay_after_stall if payload == first_lost + lost_frames else 0.0
            )
            for sent_at, output_id, value in node.sent[start:]:
                if output_id == "grip_right":
                    heapq.heappush(events, (sent_at + delay, 2, "grip", value.to_pylist()[0]))
        elif kind == "webxr_tick":
            start = len(node.sent)
            main._publish_poses(state)
            for sent_at, output_id, value in node.sent[start:]:
                if output_id == "pose_right":
                    pose = np.asarray(value.to_pylist()[0]["pose"], dtype=np.float64)
                    heapq.heappush(events, (sent_at + latency, 2, "pose", pose))
        elif kind == "grip":
            grips.append(t)
            assert clutch.on_grip("right", t, payload)
        elif kind == "pose":
            since = clutch.arms["right"]._pose_since
            assert clutch.on_pose("right", t, payload)
            if t > stall_at and clutch.arms["right"]._pose_since != since:
                out["pose_gap"] = True  # the clutch restarted settling
        else:
            assert clutch.on_pose("left", t, left) and clutch.on_grip("left", t, 0.0)
            result = clutch.tick(t)
            arm_state = result.states["right"]
            if t < stall_at:
                assert t < t0 + 0.7 or arm_state == ENGAGED
                held = result.targets["right"][:3].astype(np.float64)
                continue
            was_free = was_free or arm_state != ENGAGED
            out["held_drift"] = max(
                out["held_drift"],
                float(np.linalg.norm(result.targets["right"][:3].astype(np.float64) - held)),
            )
            out["details"].append((arm_state, result.details["right"]))
            if t < first_back:
                continue
            ms = (t - first_back) * 1000.0
            if not out["states"] or out["states"][-1][0] != arm_state:
                out["states"].append((arm_state, ms))
            if arm_state == ENGAGED and was_free:
                target = result.targets["right"][:3].astype(np.float64)
                if reference is None:
                    out["engaged_ms"], reference = ms, target
                out["drift"] = max(out["drift"], float(np.linalg.norm(target - reference)))
    out["grip_gap"] = float(np.max(np.diff(grips)))
    out["disarmed"] = out["grip_gap"] > clutch.params.fresh_s
    return out


@pytest.mark.parametrize("squeeze_ms", [15, 40, 60])
def test_a_frame_stall_that_disarms_never_slides_the_reengaged_target(
    monkeypatch, hull_models, squeeze_ms
) -> None:
    """P1, the reverify CLUTCH-1 residual. 7 frames lost: a 111 ms gap, longer than fresh_s,
    so the clutch lets go of the arm; but webxr republished the stale pose for
    POSE_TIMEOUT, so the clutch sees no pose gap and does not settle. With
    POSE_TIMEOUT <= fresh_s the gap also outlasted webxr's timeout, which resets its
    smoother: the first pose back is the hand itself, and re-engaging with the hand still
    moves nothing. With POSE_TIMEOUT 0.2 the smoother slid 8 cm into the new anchor."""
    run = _stalled_frames_through_webxr(monkeypatch, hull_models, P1_POSE_TIMEOUT, 7, squeeze_ms)
    assert run["disarmed"] and ClutchParams().fresh_s < run["grip_gap"] < P1_SMOOTHER_RESET_GAP
    assert not run["pose_gap"]
    assert run["engaged_ms"] is not None and run["engaged_ms"] < squeeze_ms + 25, run
    assert run["drift"] < 0.001, run
    assert P1_POSE_TIMEOUT <= ClutchParams().fresh_s < P1_SMOOTHER_RESET_GAP

    old = _stalled_frames_through_webxr(monkeypatch, hull_models, OLD_POSE_TIMEOUT, 7, squeeze_ms)
    assert old["disarmed"] and not old["pose_gap"]
    assert old["drift"] > 0.01, old  # 54.7 / 36.8 / 17.6 mm in the reverify run


@pytest.mark.parametrize("lost_frames", [8, 11, 20])
def test_a_held_grip_survives_a_burst_of_lost_frames(monkeypatch, hull_models, lost_frames) -> None:
    """CLUTCH-3 through the real webxr node and its real One Euro smoother: the operator
    never lets go, the hand is still, and a Wi-Fi burst takes 8-20 consecutive frames
    (111-278 ms, the shape the 2026-09-16 session shows). The arm reports the gap and then
    takes its own engagement back; the target never leaves the one it held. Before
    CLUTCH-3 (--resume-gap-ms 0) that burst cost the grip until a release and re-squeeze."""
    run = _stalled_frames_through_webxr(
        monkeypatch, hull_models, POSE_TIMEOUT_S, lost_frames, 0, hold_grip=True, move_m=0.0
    )
    assert run["disarmed"] and run["grip_gap"] > ClutchParams().fresh_s, run
    assert any(s == STALE for s, _ in run["details"]), run  # the gap is still reported
    # and then the arm is following again, within a tick of the pose coming back.
    assert [s for s, _ in run["states"]] == [ENGAGED], run
    assert run["engaged_ms"] < 10.0, run
    assert run["held_drift"] < 0.001, run
    assert any("kept the grip across" in d for _, d in run["details"]), run["details"]

    before = _stalled_frames_through_webxr(
        monkeypatch,
        hull_models,
        POSE_TIMEOUT_S,
        lost_frames,
        0,
        hold_grip=True,
        move_m=0.0,
        params=_no_resume(),
    )
    assert [s for s, _ in before["states"]] == [LOCKED], before
    assert before["engaged_ms"] is None, before  # never follows again while the grip is held


def test_a_held_grip_is_not_kept_when_the_hand_moved_during_the_burst(
    monkeypatch, hull_models
) -> None:
    """The other half of CLUTCH-3. The same burst, but the hand moved 8 cm while the frames
    were away, so webxr's reset smoother hands back a pose 8 cm from the one the arm was
    following: the arm lets go and holds, exactly as before, and only a release below
    release_grip plus the settle gives it back."""
    run = _stalled_frames_through_webxr(
        monkeypatch, hull_models, POSE_TIMEOUT_S, 11, 0, hold_grip=True, move_m=0.08
    )
    assert run["disarmed"] and [s for s, _ in run["states"]] == [LOCKED], run
    assert run["engaged_ms"] is None and run["held_drift"] < 0.001, run
    assert any("came back 8" in d and "mm" in d for _, d in run["details"]), run["details"]


def test_a_stall_the_clutch_times_longer_than_webxr_still_resets_the_smoother(
    monkeypatch, hull_models
) -> None:
    """P1 margin. webxr times a stall between frame arrivals, the clutch between grip
    deliveries, and dora delivery jitter separates the two. 6 lost frames are a 97 ms gap
    for webxr; the first grip back reaches the clutch 4 ms late, so the clutch sees 101 ms
    and lets go of the arm. With the generated POSE_TIMEOUT (10 ms below fresh_s) webxr has
    timed the frame out and the first pose back is the hand. With POSE_TIMEOUT equal to
    fresh_s it has not, and the re-engaged target slides with the hand still."""
    from dora_openarm_teleop.timing import POSE_TIMEOUT_S

    fresh_s = ClutchParams().fresh_s
    assert POSE_TIMEOUT_S <= fresh_s - 0.01
    run = _stalled_frames_through_webxr(
        monkeypatch, hull_models, POSE_TIMEOUT_S, 6, 15, grip_delay_after_stall=0.004
    )
    assert run["disarmed"] and fresh_s < run["grip_gap"] < fresh_s + 0.002, run
    assert not run["pose_gap"]
    assert run["engaged_ms"] is not None and run["drift"] < 0.001, run

    equal = _stalled_frames_through_webxr(
        monkeypatch, hull_models, fresh_s, 6, 15, grip_delay_after_stall=0.004
    )
    assert equal["disarmed"] and not equal["pose_gap"]
    assert equal["drift"] > 0.01, equal


def test_a_long_frame_stall_ends_republishing_and_the_arm_settles_before_engaging(
    monkeypatch, hull_models
) -> None:
    """P1: 17 frames lost (250 ms). webxr republishes the stale pose for POSE_TIMEOUT and then
    stops, so with POSE_TIMEOUT <= fresh_s the clutch sees a pose gap longer than fresh_s
    and stays locked until the pose has settled, grip released or not. With POSE_TIMEOUT
    0.2 the republished pose hid the gap and the arm was free to take at once."""
    settle_ms = ClutchParams().settle_s * 1000.0
    run = _stalled_frames_through_webxr(monkeypatch, hull_models, P1_POSE_TIMEOUT, 17, 500)
    assert run["disarmed"] and run["pose_gap"]
    assert run["states"][0][0] in (STALE, LOCKED)
    assert next(ms for s, ms in run["states"] if s == RELEASED) >= settle_ms, run
    assert run["engaged_ms"] is not None and run["drift"] < 0.001, run

    old = _stalled_frames_through_webxr(monkeypatch, hull_models, OLD_POSE_TIMEOUT, 17, 500)
    assert old["disarmed"] and not old["pose_gap"]
    assert next(ms for s, ms in old["states"] if s == RELEASED) < 50, old


# --- reset --------------------------------------------------------------------


def test_locked_after_reset_until_grip_released_and_repressed(hull_models, home) -> None:
    rig = _rig(hull_models)
    rig.engage("right")
    rig.move("right", [0.1, 0.0, 0.05])
    rig.clutch.reset()
    result = rig.step()
    assert result.states["right"] == LOCKED
    _pose_equal(result.targets["right"].astype(float), home["right"])
    assert result.targets["right"][7] == 0.0
    result = rig.move("right", [0.0, 0.0, 0.1])
    assert result.states["right"] == LOCKED
    _pose_equal(result.targets["right"].astype(float), home["right"])
    rig.grips["right"] = 0.0
    assert rig.step().states["right"] == RELEASED
    rig.grips["right"] = 1.0
    result = rig.step()
    assert result.states["right"] == ENGAGED
    _pose_equal(result.targets["right"].astype(float), home["right"])
    # The released left arm re-arms at once: its grip is already open.
    assert result.states["left"] == RELEASED


def test_reset_position_holds_fk_pose_and_locks(hull_models) -> None:
    rig = _rig(hull_models)
    rig.engage("right")
    qpos16 = np.array(
        [0.2, 0.1, -0.1, 1.2, 0.1, 0.2, -0.3, 0.3] + [-0.2, -0.1, 0.1, 1.4, -0.1, 0.1, 0.2, 0.5],
        dtype=np.float32,
    )
    assert rig.clutch.reset_position(qpos16)
    result = rig.step()
    expected = model_info.ee_poses_from_qpos(qpos16)
    for side in ("right", "left"):
        _pose_equal(result.targets[side].astype(float), expected[side], atol=1e-6)
    assert result.states["right"] == LOCKED
    assert result.targets["right"][7] == 0.0  # +0.3 clipped into [-0.7854, 0]
    assert result.targets["left"][7] == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [np.zeros(8), np.full(16, np.nan), "home", None])
def test_reset_position_malformed_is_ignored(hull_models, bad) -> None:
    rig = _rig(hull_models)
    rig.engage("right")
    before = rig.step()
    assert not rig.clutch.reset_position(bad)
    after = rig.step()
    assert after.states["right"] == ENGAGED
    np.testing.assert_array_equal(after.targets["right"], before.targets["right"])


# --- clamps -----------------------------------------------------------------


def _lowest_tip_z(target: np.ndarray, side: str) -> float:
    """Independent check with the MJCF hull (verified against MuJoCo FK in test_model_info)."""
    points = model_info.fingertip_points(side, float(target[7]))
    world = target[:3] + points @ frames.quat_to_matrix(target[3:7]).T
    return float(world[:, 2].min())


@pytest.mark.parametrize("side", ["right", "left"])
def test_table_clamp_keeps_fingertips_above_table_pointing_down(hull_models, side: str) -> None:
    rng = np.random.default_rng(11 if side == "right" else 12)
    table = model_info.table_top_z_arm_origin()
    low, high = model_info.gripper_ranges()[side]
    down = np.array([1.0, 0.0, 0.0, 0.0])  # identity: the fingers (site -z) point along world -z
    for trial in range(40):
        # Half the trials without the reach clamp, so the table is the only limit.
        params = ClutchParams(clamp_reach=trial % 2 == 0)
        rig = _rig(hull_models, params)
        tilt = frames.quat_from_axis_angle(rng.normal(size=3), rng.uniform(0, np.deg2rad(40)))
        yaw = frames.quat_from_axis_angle([0, 0, 1], rng.uniform(-np.pi, np.pi))
        start = model_info.home_ee_poses()[side]
        start_hold = np.concatenate(
            [start[:3], frames.quat_multiply(yaw, frames.quat_multiply(tilt, down))]
        )
        # Hold and hand at the same, random opening, so the gripper follows from the start.
        gripper = rng.uniform(low, high)
        rig.clutch.arms[side].reset(start_hold, gripper)
        rig.hands[side][7] = gripper
        rig.engage(side)
        result = rig.move(side, [rng.uniform(-0.05, 0.15), 0.0, -0.5], n=40)
        target = result.targets[side].astype(float)
        assert result.states[side] == ENGAGED
        lowest = _lowest_tip_z(target, side)
        assert lowest >= table + 0.005 - 1e-5
        if not params.clamp_reach:
            assert lowest == pytest.approx(table + 0.005, abs=1e-5)  # pressed to the limit


def test_table_clamp_point_model_and_disabled(home) -> None:
    table = model_info.table_top_z_arm_origin()
    point_models = arm_models_from_scene("cell", "point")
    down = np.array([1.0, 0.0, 0.0, 0.0])
    offset = model_info.fingertip_offset("right")
    for params, clamped in ((ClutchParams(clamp_reach=False), True), (NO_CLAMPS, False)):
        rig = _rig(point_models, params)
        rig.clutch.arms["right"].reset(np.concatenate([home["right"][:3], down]))
        rig.engage("right")
        target = rig.move("right", [0, 0, -0.5]).targets["right"].astype(float)
        tip_z = (target[:3] + frames.quat_rotate(target[3:7], offset))[2]
        if clamped:
            assert tip_z == pytest.approx(table + 0.005, abs=1e-5)
        else:
            assert tip_z < table - 0.1


def test_reach_clamp(hull_models) -> None:
    params = ClutchParams(clamp_table=False, clamp_midline=False)
    rig = _rig(hull_models, params)
    rig.engage("right")
    target = rig.move("right", [1.0, 0.0, 0.0]).targets["right"].astype(float)
    shoulder = model_info.shoulder_positions()["right"]
    assert np.linalg.norm(target[:3] - shoulder) == pytest.approx(params.reach_max, abs=1e-5)


def test_midline_clamp(hull_models) -> None:
    params = ClutchParams(clamp_table=False, clamp_reach=False)
    rig = _rig(hull_models, params)
    rig.engage("right")
    rig.engage("left")
    rig.move("right", [0.0, 0.5, 0.0], n=10)
    result = rig.move("left", [0.0, -0.5, 0.0], n=10)
    assert result.targets["right"][1] == pytest.approx(0.05)
    assert result.targets["left"][1] == pytest.approx(-0.05)


@pytest.mark.parametrize(
    ("side", "hand_gripper", "expected"),
    [
        ("right", 0.1745, 0.0),  # webxr's closed overshoot
        ("right", -1.0, -0.7854),
        ("right", -0.3, -0.3),
        ("left", -0.1745, 0.0),
        ("left", 1.0, 0.7854),
        ("left", 0.3, 0.3),
    ],
)
def test_gripper_clip_per_side(
    hull_models, side: str, hand_gripper: float, expected: float
) -> None:
    rig = _rig(hull_models, NO_CLAMPS)
    rig.engage(side)  # hand gripper 0 = the held (closed) gripper: follows at once
    rig.hands[side][7] = hand_gripper
    target = rig.step().targets[side]
    assert target[7] == pytest.approx(expected, abs=1e-6)


# --- gripper pickup (CLUTCH-2) ----------------------------------------------------


def test_reengaging_with_the_trigger_relaxed_keeps_a_closed_gripper_on_the_table(
    hull_models,
) -> None:
    """The review's engage_gripper3.py: re-engaging must not drop the held object."""
    rig = _rig(hull_models, ClutchParams())
    table = model_info.table_top_z_arm_origin()
    rig.engage("right")  # trigger pulled: hand gripper 0, closed
    q0 = rig.hands["right"][3:7].copy()
    for k in range(1, 101):  # roll the hand 90 degrees about the forward axis
        rig.hands["right"][3:7] = frames.quat_multiply(
            frames.quat_from_axis_angle([1, 0, 0], np.pi / 2 * k / 100), q0
        )
        rig.step()
    held = rig.move("right", [0.0, 0.0, -0.4], n=200).targets["right"].copy()
    assert held[7] == 0.0
    assert _lowest_tip_z(held, "right") == pytest.approx(table + 0.005, abs=1e-5)  # on the table

    rig.grips["right"] = 0.0
    rig.step(5)
    rig.hands["right"][7] = -0.7854  # trigger relaxed while released
    rig.step(5)
    rig.grips["right"] = 1.0
    first = rig.step()
    assert first.states["right"] == ENGAGED
    # Was: gripper 0 -> -0.785 (object dropped) and z +45 mm in one tick.
    np.testing.assert_allclose(first.targets["right"], held, atol=1e-6)

    # Pulling the trigger does nothing until the hand's gripper reaches the held angle...
    for value in np.linspace(-0.7854, -0.06, 20):
        rig.hands["right"][7] = value
        np.testing.assert_allclose(rig.step().targets["right"], held, atol=1e-6)
    rig.hands["right"][7] = -0.04  # ...within gripper_pickup of it: picked up
    assert rig.step().targets["right"][7] == pytest.approx(-0.04)
    # Then it follows, and the hull table clamp still lifts an opening gripper.
    rig.hands["right"][7] = -0.7854
    opened = rig.step().targets["right"].astype(np.float64)
    assert opened[7] == pytest.approx(-0.7854)
    assert opened[2] > held[2] + 0.01
    assert _lowest_tip_z(opened, "right") >= table + 0.005 - 1e-5


def test_gripper_is_picked_up_when_the_hand_crosses_the_held_angle(hull_models, home) -> None:
    rig = _rig(hull_models)
    rig.clutch.arms["right"].reset(home["right"], -0.4)
    rig.hands["right"][7] = -0.7
    assert rig.engage("right").targets["right"][7] == pytest.approx(-0.4)
    assert rig.step().targets["right"][7] == pytest.approx(-0.4)
    rig.hands["right"][7] = -0.1  # jumps past -0.4 without ever being within the band
    assert rig.step().targets["right"][7] == pytest.approx(-0.1)
    rig.hands["right"][7] = -0.3
    assert rig.step().targets["right"][7] == pytest.approx(-0.3)
    # Every engagement picks up afresh: released at -0.3, re-engaged with the hand at 0.
    rig.grips["right"] = 0.0
    rig.step()
    rig.hands["right"][7] = 0.0
    rig.grips["right"] = 1.0
    assert rig.step().targets["right"][7] == pytest.approx(-0.3)


def test_gripper_within_the_pickup_band_follows_at_once(hull_models, home) -> None:
    rig = _rig(hull_models)
    rig.clutch.arms["left"].reset(home["left"], 0.4)
    rig.hands["left"][7] = 0.44
    assert rig.engage("left").targets["left"][7] == pytest.approx(0.44)


# --- table plane (REAL-1) -----------------------------------------------------------


def test_table_z_replaces_the_scene_table(home) -> None:
    models = arm_models_from_scene("cell", "point", table_z=-0.25)
    assert {m.table_z for m in models.values()} == {-0.25}
    down = np.array([1.0, 0.0, 0.0, 0.0])
    offset = model_info.fingertip_offset("right")
    rig = _rig(models, ClutchParams(clamp_reach=False))
    rig.clutch.arms["right"].reset(np.concatenate([home["right"][:3], down]))
    rig.engage("right")
    target = rig.move("right", [0, 0, -0.5]).targets["right"].astype(float)
    tip_z = (target[:3] + frames.quat_rotate(target[3:7], offset))[2]
    assert tip_z == pytest.approx(-0.25 + 0.005, abs=1e-5)
    with pytest.raises(ValueError):
        arm_models_from_scene("cell", "point", table_z=float("nan"))


def test_table_lift_flags_a_plane_above_the_start_pose(hull_models, home) -> None:
    params = ClutchParams()
    hold = np.concatenate([home["right"], [0.0]])
    assert table_lift(hold, hull_models["right"], params) == 0.0
    lowest = _lowest_tip_z(hold, "right")
    high = arm_models_from_scene("cell", "hull", table_z=lowest + 0.02)["right"]
    assert table_lift(hold, high, params) == pytest.approx(0.025, abs=1e-6)
    assert table_lift(hold, high, ClutchParams(clamp_table=False)) == 0.0


# --- malformed input ------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_pose",
    [
        np.full(8, np.nan, dtype=np.float32),
        np.array([0.3, 0, 0, 1, 0, 0, 0], dtype=np.float32),  # 7 values
        np.array([0.3, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),  # zero quaternion
        np.array([0.3, 0, np.inf, 1, 0, 0, 0, 0], dtype=np.float32),
        "pose",
    ],
)
def test_malformed_pose_is_ignored_and_arm_holds(hull_models, bad_pose) -> None:
    rig = _rig(hull_models)
    rig.engage("right")
    held = rig.step().targets["right"].copy()
    for _ in range(5):
        rig.t += DT
        assert not rig.clutch.on_pose("right", rig.t, bad_pose)
        assert rig.clutch.on_grip("right", rig.t, 1.0)
        result = rig.clutch.tick(rig.t)
        np.testing.assert_array_equal(result.targets["right"], held)
    # Persistently malformed input is no input: the arm goes stale and stays put.
    for _ in range(30):
        rig.t += DT
        rig.clutch.on_pose("right", rig.t, bad_pose)
        rig.clutch.on_grip("right", rig.t, 1.0)
        result = rig.clutch.tick(rig.t)
    assert result.states["right"] == STALE
    np.testing.assert_array_equal(result.targets["right"], held)


@pytest.mark.parametrize("bad_grip", [float("nan"), None, "1.0x", float("inf")])
def test_malformed_grip_is_ignored(hull_models, bad_grip) -> None:
    rig = _rig(hull_models)
    rig.engage("right")
    assert not rig.clutch.on_grip("right", rig.t, bad_grip)
    assert rig.step().states["right"] == ENGAGED


def test_params_validation() -> None:
    with pytest.raises(ValueError):
        ClutchParams(engage_grip=0.4, release_grip=0.4)
    with pytest.raises(ValueError):
        ClutchParams(fresh_s=0.0)
    with pytest.raises(ValueError):
        ClutchParams(scale=float("nan"))
    with pytest.raises(ValueError):
        ClutchParams(settle_s=-0.1)
    with pytest.raises(ValueError):
        ClutchParams(engage_max_speed=0.0)
    with pytest.raises(ValueError):
        ClutchParams(engage_max_angular_speed=float("nan"))
    with pytest.raises(ValueError):
        ClutchParams(gripper_pickup=0.0)
    with pytest.raises(ValueError):
        ClutchParams(resume_gap_s=-0.1)
    with pytest.raises(ValueError):
        ClutchParams(resume_gap_s=float("inf"))
    with pytest.raises(ValueError):
        ClutchParams(resume_max_move=-0.01)
    with pytest.raises(ValueError):
        ClutchParams(resume_max_turn=float("nan"))


# --- node event loop ----------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.t = 50.0

    def __call__(self) -> float:
        return self.t


def _input(input_id: str, value: pa.Array) -> dict:
    return {"type": "INPUT", "id": input_id, "value": value, "metadata": {}}


def _grip(value: float) -> pa.Array:
    return pa.array([value], type=pa.float32())


def _script(clock: FakeClock, frames_: list[list[dict]]) -> Iterator[dict]:
    """Each inner list is one 5 ms slot: inputs first, then the tick."""
    for slot in frames_:
        clock.t += DT
        yield from slot
        yield _input("tick", pa.array([0], type=pa.uint64()))


def _slot(
    right_pose: np.ndarray, right_grip: float, left_pose: np.ndarray, left_grip: float
) -> list[dict]:
    return [
        _input("pose_right", arrow.pose_array(right_pose)),
        _input("grip_right", _grip(right_grip)),
        # webxr's struct form and the flat form must both be accepted.
        _input("pose_left", pa.array(left_pose, type=pa.float32())),
        _input("grip_left", _grip(left_grip)),
    ]


def _run(events, clutch: Clutch, clock: FakeClock, **options):
    sent: list[tuple[str, pa.Array, dict]] = []
    logs: list[str] = []
    reason = clutch_node.run(
        events,
        lambda o, v, m: sent.append((o, v, m)),
        clutch,
        clock=clock,
        log=logs.append,
        **options,
    )
    return reason, sent, logs


def test_node_publishes_both_targets_every_tick_and_state_on_change(hull_models) -> None:
    clock = FakeClock()
    right, left = hand((0.3, -0.2, 0.0)), hand((0.3, 0.2, 0.0))
    slots = [_slot(right, 0.0, left, 0.0)] * SETTLE_TICKS + [_slot(right, 1.0, left, 0.0)] * 3
    moved = right.copy()
    moved[0] += 0.02
    slots += [_slot(moved, 1.0, left, 0.0)]
    # Lazily chained: the script advances the clock as the loop consumes it.
    events = chain(_script(clock, slots), [_input("control", arrow.string_array("quit"))])
    reason, sent, _ = _run(events, Clutch(hull_models, NO_CLAMPS), clock)

    assert reason == "control=quit"
    targets = [s for s in sent if s[0].startswith("target_")]
    assert len(targets) == 2 * len(slots)
    assert [s[0] for s in targets[:2]] == ["target_right", "target_left"]
    assert targets[0][1].type == arrow.POSE_TYPE
    assert isinstance(targets[0][2]["timestamp"], int)
    states = [json.loads(arrow.text(s[1])) for s in sent if s[0] == "clutch_state"]
    assert states == [
        {"right": "locked", "left": "locked"},  # settling after the first poses
        {"right": "released", "left": "released"},
        {"right": "engaged", "left": "released"},
    ]
    home_right = arrow.values(targets[0][1], "pose")
    last_right = arrow.values(targets[-2][1], "pose")
    np.testing.assert_allclose(last_right[:3], home_right[:3] + [0.02, 0, 0], atol=1e-6)


def test_node_logs_why_an_arm_lost_and_regained_the_hand(hull_models) -> None:
    """`clutch state right=stale` alone is the same line whether the pose stopped, the
    grip did, or the arm is about to take its own engagement back, which is what made the
    field reports undiagnosable. The published JSON keeps the four states exactly."""
    clock = FakeClock()
    right, left = hand((0.3, -0.2, 0.0)), hand((0.3, 0.2, 0.0))
    silent = [
        _input("pose_left", pa.array(left, type=pa.float32())),
        _input("grip_left", _grip(0)),
    ]
    slots = (
        [_slot(right, 0.0, left, 0.0)] * SETTLE_TICKS
        + [_slot(right, 1.0, left, 0.0)] * 3
        + [silent] * 30  # 150 ms with the right controller's frames gone
        + [_slot(right, 1.0, left, 0.0)] * 3
    )
    _, sent, logs = _run(_script(clock, slots), Clutch(hull_models, NO_CLAMPS), clock)

    states = [json.loads(arrow.text(s[1])) for s in sent if s[0] == "clutch_state"]
    assert [s["right"] for s in states] == ["locked", "released", "engaged", "stale", "engaged"]
    assert {v for s in states for v in s.values()} <= {ENGAGED, RELEASED, STALE, LOCKED}
    clutch_logs = [line for line in logs if line.startswith("clutch state")]
    lost = next(line for line in clutch_logs if "right=stale" in line)
    assert "ms late, keeping the grip)" in lost, lost  # says it will take the grip back
    back = clutch_logs[-1]
    assert "right=engaged (kept the grip across a 15" in back, back
    assert "right=locked (settling," in clutch_logs[0], clutch_logs[0]


def test_node_control_reset_and_malformed_inputs(hull_models) -> None:
    clock = FakeClock()
    right, left = hand((0.3, -0.2, 0.0)), hand((0.3, 0.2, 0.0))
    slots = [_slot(right, 0.0, left, 0.0)] * SETTLE_TICKS + [_slot(right, 1.0, left, 0.0)] * 2
    bad = [
        _input("control", arrow.string_array("reset")),
        _input("pose_right", pa.array(["garbage"])),
        _input("grip_right", pa.array([], type=pa.float32())),
        _input("reset_position", pa.array([1.0, 2.0], type=pa.float32())),
        {"type": "ERROR", "error": "boom"},
    ]
    events = chain(_script(clock, slots), bad, _script(clock, [_slot(right, 1.0, left, 0.0)]))
    reason, sent, logs = _run(events, Clutch(hull_models, NO_CLAMPS), clock)

    assert reason == "event stream ended"
    states = [json.loads(arrow.text(s[1])) for s in sent if s[0] == "clutch_state"]
    assert states[-1] == {"right": "locked", "left": "released"}
    assert any("malformed" in line for line in logs)


def test_node_reset_position_input(hull_models) -> None:
    clock = FakeClock()
    qpos16 = np.zeros(16, dtype=np.float32)
    qpos16[3] = qpos16[11] = 1.0
    events = [_input("reset_position", arrow.qpos_array(qpos16))]
    events += list(_script(clock, [[]]))
    _, sent, _ = _run(
        events, Clutch(hull_models, NO_CLAMPS, fk=model_info.ee_poses_from_qpos), clock
    )
    expected = model_info.ee_poses_from_qpos(qpos16)
    got = arrow.values(sent[0][1], "pose")
    assert sent[0][0] == "target_right"
    _pose_equal(got.astype(float), expected["right"], atol=1e-6)


@pytest.mark.parametrize(
    ("closing", "reason"),
    [
        (["control"], "control input closed"),
        (["pose_right", "pose_left", "grip_right", "grip_left"], "webxr inputs closed"),
    ],
)
def test_node_exits_when_inputs_close(hull_models, closing: list[str], reason: str) -> None:
    clock = FakeClock()
    events = [{"type": "INPUT_CLOSED", "id": i} for i in closing]
    events += list(_script(clock, [[]]))
    got, sent, _ = _run(events, Clutch(hull_models), clock)
    assert got == reason
    assert sent == []


def test_node_keeps_running_when_only_one_controller_stream_closes(hull_models) -> None:
    clock = FakeClock()
    events = [{"type": "INPUT_CLOSED", "id": "pose_left"}, {"type": "STOP"}]
    got, _, _ = _run(events, Clutch(hull_models), clock)
    assert got == "stop"


def test_tick_monitor_warns_on_rate_mismatch() -> None:
    monitor = clutch_node.TickMonitor(200.0, window_s=1.0)
    warnings = [monitor.tick(i * 0.005) for i in range(400)]
    assert not any(warnings)
    slow = clutch_node.TickMonitor(200.0, window_s=1.0)
    warnings = [slow.tick(i * 0.008) for i in range(300)]
    assert any(w and "125.0 Hz" in w for w in warnings)


def test_cli_defaults_match_design() -> None:
    args = clutch_node.build_parser().parse_args([])
    params = clutch_node.params_from_args(args)
    assert params == ClutchParams()
    assert (params.engage_grip, params.release_grip, params.fresh_s) == (0.7, 0.4, 0.1)
    assert args.tick_hz == 200.0
    assert (params.settle_s, params.engage_max_speed, params.gripper_pickup) == (0.3, 0.05, 0.05)
    assert (params.resume_gap_s, params.resume_max_move) == (0.5, 0.02)
    assert math.degrees(params.resume_max_turn) == pytest.approx(5.0, abs=0.02)
    # The resume window is the one gap rule that is not coupled to webxr's timings,
    # but it must outlast the freshness window it bridges.
    assert params.resume_gap_s > params.fresh_s
    assert args.table_z is None
    off = clutch_node.params_from_args(clutch_node.build_parser().parse_args(["--no-clamps"]))
    assert not (off.clamp_table or off.clamp_reach or off.clamp_midline)


def test_cli_settle_pickup_and_table_z() -> None:
    parser = clutch_node.build_parser()
    args = parser.parse_args(
        [
            "--settle-ms",
            "500",
            "--engage-max-speed",
            "0.1",
            "--engage-max-angular-speed",
            "inf",
            "--gripper-pickup",
            "0.1",
            "--table-z",
            "-0.30",
            "--resume-gap-ms",
            "0",
            "--resume-max-move",
            "0.03",
            "--resume-max-turn",
            "0.2",
        ]
    )
    params = clutch_node.params_from_args(args)
    assert (params.settle_s, params.engage_max_speed, params.gripper_pickup) == (0.5, 0.1, 0.1)
    assert params.engage_max_angular_speed == float("inf")
    assert (params.resume_gap_s, params.resume_max_move, params.resume_max_turn) == (0.0, 0.03, 0.2)
    assert args.table_z == -0.30
    with pytest.raises(SystemExit):
        parser.parse_args(["--table-z", "nan"])


def test_startup_warnings_for_the_table_plane(hull_models, home) -> None:
    clutch = Clutch(hull_models, ClutchParams())
    assert clutch_node.startup_warnings(clutch, None) == []
    off = Clutch(hull_models, ClutchParams(clamp_table=False))
    (warning,) = clutch_node.startup_warnings(off, None)
    assert "table clamp OFF" in warning and "--table-z" in warning
    lowest = _lowest_tip_z(np.concatenate([home["right"], [0.0]]), "right")
    high = Clutch(arm_models_from_scene("cell", "hull", table_z=lowest + 0.05), ClutchParams())
    warnings = clutch_node.startup_warnings(high, lowest + 0.05)
    assert len(warnings) == 2 and all("engaging will lift" in w for w in warnings)


# --- the robot's measured state ----------------------------------------------

CELL_START = np.array([0.0, 0.0, 0.0, 1.7707963, 0.0, 0.0, 0.0, -0.4])  # openarm_driver


def test_first_robot_state_seeds_the_hold_at_the_arm(hull_models) -> None:
    """The Cell starts at J4 1.77 rad, not the scene's 1.57: the IK must start there."""
    rig = _rig(hull_models)
    assert rig.clutch.on_robot_state("right", CELL_START, 1) is True
    result = rig.step()
    qpos16 = np.concatenate([CELL_START, np.zeros(8)])
    expected = model_info.ee_poses_from_qpos(qpos16)["right"]
    _pose_equal(result.targets["right"].astype(float), expected, atol=1e-6)
    assert result.targets["right"][7] == pytest.approx(-0.4)


def test_later_states_of_the_same_start_do_not_move_the_hold(hull_models) -> None:
    """Holding the measured pose every tick would let the hold sag with the arm."""
    rig = _rig(hull_models)
    rig.clutch.on_robot_state("right", CELL_START, 1)
    held = rig.step().targets["right"].copy()
    sagged = CELL_START.copy()
    sagged[1] -= 0.1
    assert rig.clutch.on_robot_state("right", sagged, 1) is False
    assert rig.clutch.on_robot_state("right", sagged, None) is False
    np.testing.assert_array_equal(rig.step().targets["right"], held)


def test_a_new_start_epoch_reseeds_and_locks_an_engaged_arm(hull_models) -> None:
    rig = _rig(hull_models)
    rig.clutch.on_robot_state("right", CELL_START, 1)
    rig.engage("right")
    moved = CELL_START.copy()
    moved[0] = 0.3
    assert rig.clutch.on_robot_state("right", moved, 2) is True
    result = rig.step()
    expected = model_info.ee_poses_from_qpos(np.concatenate([moved, np.zeros(8)]))["right"]
    _pose_equal(result.targets["right"].astype(float), expected, atol=1e-6)
    # Still squeezing: locked until the grip is released, so it does not jump.
    assert result.states["right"] == LOCKED


def test_robot_state_of_one_arm_leaves_the_other(hull_models) -> None:
    rig = _rig(hull_models)
    before = rig.step().targets["left"].copy()
    rig.clutch.on_robot_state("right", CELL_START, 1)
    np.testing.assert_array_equal(rig.step().targets["left"], before)


@pytest.mark.parametrize("bad", [np.zeros(3), np.full(8, np.nan), "x", None])
def test_malformed_robot_state_is_rejected(hull_models, bad) -> None:
    rig = _rig(hull_models)
    assert rig.clutch.on_robot_state("right", bad, 1) is None
    # ...and does not count as the first state.
    assert rig.clutch.on_robot_state("right", CELL_START, 1) is True


# --- fixed grippers and the intro (docs/teleop.md, Intro) --------------------


def _start_qpos(gripper: float) -> np.ndarray:
    """One arm's 8 joints at dora-openarm's start pose (J4 at 90 deg)."""
    qpos = np.zeros(8)
    qpos[3] = math.pi / 2
    qpos[7] = gripper
    return qpos


def test_a_fixed_gripper_closes_until_the_arm_aligns_then_holds_its_opening(
    hull_models,
) -> None:
    """dora-openarm takes no command until it is sent a closed gripper (ALIGNED)."""
    params = ClutchParams(
        clamp_table=False, clamp_reach=False, clamp_midline=False, fixed_grippers=(("right", 0.3),)
    )
    rig = _rig(hull_models, params)
    opening = 0.3 * hull_models["right"].gripper_range[0]  # the right one opens towards -
    assert rig.clutch.on_robot_state("right", _start_qpos(-0.4), epoch=1)  # Start Arm
    rig.step(2)
    rig.grips["right"] = 1.0
    rig.hands["right"][7] = -0.78  # the trigger relaxed all the way open
    result = rig.step(100)  # 0.5 s, and the arm has not closed it: not aligned
    assert result.states["right"] == ENGAGED
    assert result.targets["right"][7] == 0.0

    rig.clutch.on_robot_state("right", _start_qpos(-0.01), epoch=1)  # now it has
    for gripper in (0.0, -0.4, -0.78):
        rig.hands["right"][7] = gripper
        result = rig.step(3)
        assert result.states["right"] == ENGAGED
        assert result.targets["right"][7] == pytest.approx(opening)
    rig.grips["right"] = 0.0
    assert rig.step(3).targets["right"][7] == pytest.approx(opening)  # released too

    # Pause Arm, Start Arm: closed again until the arm has closed it.
    assert rig.clutch.on_robot_state("right", _start_qpos(-0.4), epoch=2)
    assert rig.step(100).targets["right"][7] == 0.0


def test_a_fixed_gripper_opens_only_after_it_was_commanded_closed_for_a_while(
    hull_models,
) -> None:
    """A gripper already closed at the start still gets a closed command to align on."""
    params = ClutchParams(
        clamp_table=False, clamp_reach=False, clamp_midline=False, fixed_grippers=(("right", 0.3),)
    )
    rig = _rig(hull_models, params)
    assert rig.clutch.on_robot_state("right", _start_qpos(0.0), epoch=1)
    ticks = round(params.fixed_close_s / DT)
    closed = [rig.step().targets["right"][7] for _ in range(ticks + 2)]
    assert closed[: ticks - 1] == [0.0] * (ticks - 1)
    assert closed[-1] == pytest.approx(0.3 * hull_models["right"].gripper_range[0])


@pytest.mark.parametrize(
    "fixed", [(("middle", 0.3),), (("right", 1.5),), (("right", 0.3), ("right", 0.2))]
)
def test_fixed_grippers_are_validated(fixed) -> None:
    with pytest.raises(ValueError):
        ClutchParams(fixed_grippers=fixed)


def test_the_intro_side_is_validated() -> None:
    with pytest.raises(ValueError):
        Clutch(arm_models_from_scene("cell", "point"), intro="middle")


class _ArmEcho:
    """The followers: each tick, the measured gripper is the commanded one."""

    def __init__(self, rig: Rig) -> None:
        self.rig = rig
        self.left_states: list[str] = []

    def step(self, n: int = 1):
        result = None
        for _ in range(n):
            result = self.rig.step()
            for side in ("right", "left"):
                qpos = _start_qpos(float(result.targets[side][7]))
                self.rig.clutch.on_robot_state(side, qpos, epoch=1)
            self.left_states.append(result.states["left"])
        return result

    def until_left(self, state: str, limit_s: float = 30.0):
        for _ in range(round(limit_s / DT)):
            result = self.step()
            if result.states["left"] == state:
                return result
        raise AssertionError(f"left never became {state}: {self.left_states[-1]}")


def test_the_left_intro_runs_after_start_arm_then_the_clutch_takes_over(hull_models) -> None:
    clutch = Clutch(hull_models, NO_CLAMPS, fk=model_info.ee_poses_from_qpos, intro="left")
    rig = Rig(clutch)
    # Squeezed all along: the intro ignores the hand, and the clutch only takes
    # the arm once the grip has been released afterwards.
    rig.grips["left"] = 1.0
    assert clutch.on_robot_state("right", _start_qpos(-0.4), epoch=1)
    assert clutch.on_robot_state("left", _start_qpos(0.4), epoch=1)
    assert clutch.intro_running("left") and not clutch.intro_running("right")
    arms = _ArmEcho(rig)

    start = model_info.ee_poses_from_qpos(np.concatenate([_start_qpos(0), _start_qpos(0)]))
    waiting = arms.until_left("waiting")
    assert set(arms.left_states[:-1]) == {"setup"}
    assert waiting.states["right"] == RELEASED  # the right arm teleoperates meanwhile
    up = frames.quat_to_matrix(waiting.targets["left"][3:7].astype(float))[:, 1]
    np.testing.assert_allclose(up, [0, 0, 1], atol=1e-6)  # site +y up: right finger down
    assert waiting.targets["left"][7] == pytest.approx(0.3927, abs=1e-4)
    np.testing.assert_allclose(waiting.targets["left"][:3], start["left"][:3], atol=1e-6)

    assert clutch.proceed()
    handed = arms.until_left(LOCKED)
    assert not clutch.intro_running("left")
    down = frames.quat_to_matrix(handed.targets["left"][3:7].astype(float))[:, 1]
    np.testing.assert_allclose(down, [0, 0, -1], atol=1e-6)  # left finger down
    assert handed.targets["left"][7] == 0.0
    assert not clutch.proceed()  # nothing waits any more

    # Release, squeeze again: it follows the hand from where the intro left it.
    rig.grips["left"] = 0.0
    assert arms.until_left(RELEASED).targets["left"][7] == 0.0
    rig.grips["left"] = 1.0
    arms.step(2)
    moved = rig.move("left", (0.0, 0.0, 0.05))
    assert moved.states["left"] == ENGAGED
    np.testing.assert_allclose(
        moved.targets["left"][:3], handed.targets["left"][:3] + [0, 0, 0.05], atol=1e-6
    )
    np.testing.assert_allclose(moved.targets["left"][3:7], handed.targets["left"][3:7], atol=1e-6)


def test_each_start_arm_runs_the_intro_again_and_reset_drops_it(hull_models) -> None:
    clutch = Clutch(hull_models, NO_CLAMPS, fk=model_info.ee_poses_from_qpos, intro="left")
    rig = Rig(clutch)
    clutch.on_robot_state("left", _start_qpos(0.4), epoch=1)
    arms = _ArmEcho(rig)
    arms.until_left("waiting")
    clutch.reset()
    assert not clutch.intro_running("left")
    assert clutch.on_robot_state("left", _start_qpos(0.4), epoch=2)  # Pause, Start Arm
    assert clutch.intro_running("left")
    assert arms.step().states["left"] == "setup"


def test_an_episode_end_sends_the_intro_arm_back_and_runs_it_again(hull_models) -> None:
    clutch = Clutch(hull_models, NO_CLAMPS, fk=model_info.ee_poses_from_qpos, intro="left")
    rig = Rig(clutch)
    rig.grips["left"] = 1.0  # squeezed: the intro hands the arm over locked
    assert clutch.restart_intro() == []  # before the arms' first start
    clutch.on_robot_state("right", _start_qpos(-0.4), epoch=1)
    clutch.on_robot_state("left", _start_qpos(0.4), epoch=1)
    arms = _ArmEcho(rig)
    first = arms.until_left("waiting")
    start_right = first.targets["right"].astype(np.float64)
    assert clutch.restart_intro() == ["right"]  # the left one is still in its intro
    clutch._intros.pop("right")  # (undone: only the end of the test sends it back)
    clutch.proceed()
    arms.until_left(LOCKED)
    # The hand takes the arm 10 cm up and away.
    rig.grips["left"] = 0.0
    arms.until_left(RELEASED)
    rig.grips["left"] = 1.0
    arms.step(2)
    moved = rig.move("left", (0.05, 0.0, 0.10))
    assert moved.states["left"] == ENGAGED

    rig.grips["right"] = 1.0
    arms.step(2)
    right_moved = rig.move("right", (0.05, 0.0, 0.10))
    assert right_moved.states["right"] == ENGAGED

    assert clutch.restart_intro() == ["right", "left"]
    back = arms.step()
    assert back.states["left"] == "setup" and back.states["right"] == "setup"
    np.testing.assert_allclose(
        back.targets["right"][:3], right_moved.targets["right"][:3], atol=2e-3
    )
    # It leaves from where the hand left it, not from the start pose.
    np.testing.assert_allclose(back.targets["left"][:3], moved.targets["left"][:3], atol=2e-3)
    again = arms.until_left("waiting")
    np.testing.assert_allclose(again.targets["left"], first.targets["left"], atol=1e-5)
    # The right arm is back at its start pose, held there until the grip is released.
    assert again.states["right"] == LOCKED
    np.testing.assert_allclose(again.targets["right"][:7], start_right[:7], atol=1e-5)
    assert clutch.proceed()
    assert arms.until_left(LOCKED).targets["left"][7] == 0.0


def test_intro_return_eases_back_at_the_return_speed() -> None:
    start = np.array([0.2, 0.15, -0.2, 1.0, 0.0, 0.0, 0.0, 0.0])
    current = start.copy()
    current[:3] += [0.0, 0.0, 0.1]
    intro = Intro(start, (0.0, 0.7854), IntroParams(return_speed=0.1), current=current)
    assert intro.phase == "return"
    np.testing.assert_allclose(intro.step(0.0, 0.0)[:3], current[:3])
    np.testing.assert_allclose(intro.step(0.5, 0.0)[:3], start[:3] + [0, 0, 0.05], atol=1e-9)
    intro.step(1.0, 0.0)  # 0.1 m at 0.1 m/s: back
    assert intro.phase != "return"


def test_node_restarts_the_intro_on_episode_end_and_x_proceeds(hull_models) -> None:
    clock = FakeClock()
    clutch = Clutch(hull_models, NO_CLAMPS, fk=model_info.ee_poses_from_qpos, intro="left")
    state = arrow.state_array(_start_qpos(0.4), np.zeros(8), np.zeros(8))
    x = {False: pa.array([False]), True: pa.array([True])}
    clutch.on_robot_state("left", _start_qpos(0.4), epoch=1)
    clutch._intros.clear()  # as if its first intro had finished
    events = [
        {"type": "INPUT", "id": "state_left", "value": state, "metadata": {"start_epoch": 1}},
        _input("control", arrow.string_array("success")),
        _input("button_x", x[True]),
        _input("button_x", x[True]),  # held: one press
        _input("button_x", x[False]),
        *_script(clock, [[]]),
    ]
    _, _, logs = _run(events, clutch, clock, intro_every_episode=True)
    assert (
        "control=success: left going back to the start pose, the left arm then runs its intro"
        in logs
    )
    assert logs.count("button X ignored: no intro is waiting for it") == 1
    assert clutch.intro_running("left")

    off = Clutch(hull_models, NO_CLAMPS, fk=model_info.ee_poses_from_qpos, intro="left")
    off.on_robot_state("left", _start_qpos(0.4), epoch=1)
    off._intros.clear()
    _run([_input("control", arrow.string_array("fail"))], off, FakeClock())
    assert not off.intro_running("left")


def test_node_starts_the_intro_and_passes_enter_on(hull_models) -> None:
    clock = FakeClock()
    clutch = Clutch(hull_models, NO_CLAMPS, fk=model_info.ee_poses_from_qpos, intro="left")
    state = arrow.state_array(_start_qpos(0.4), np.zeros(8), np.zeros(8))
    events = [
        {"type": "INPUT", "id": "state_left", "value": state, "metadata": {"start_epoch": 1}},
        _input("control", arrow.string_array("proceed")),
        *_script(clock, [[]]),
    ]
    _, sent, logs = _run(events, clutch, clock)

    assert "state_left (start_epoch 1): starting the left arm's intro" in logs
    assert "control=proceed ignored: no intro is waiting for it" in logs
    states = [json.loads(arrow.text(s[1])) for s in sent if s[0] == "clutch_state"]
    assert states[-1]["left"] == "setup"


def test_node_args_read_the_intro_and_fixed_grippers_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("INTRO_ARM", "left")
    monkeypatch.setenv("FIXED_GRIPPERS", "right=0.3, left=0")
    args = clutch_node.build_parser().parse_args([])
    assert args.intro == "left"
    assert clutch_node.params_from_args(args).fixed_grippers == (("right", 0.3), ("left", 0.0))
    args = clutch_node.build_parser().parse_args(["--intro", "none", "--fixed-grippers", "none"])
    assert args.intro is None and args.fixed_grippers == ()
    for name, bad in (
        ("INTRO_ARM", "middle"),
        ("FIXED_GRIPPERS", "right=2"),
        ("FIXED_GRIPPERS", "right"),
    ):
        monkeypatch.setenv(name, bad)
        with pytest.raises(SystemExit):
            clutch_node.build_parser().parse_args([])
        monkeypatch.delenv(name)


def _finger_down_rig(hull_models, fixed=(("right", 0.6),)):
    params = ClutchParams(
        clamp_table=False,
        clamp_reach=False,
        clamp_midline=False,
        fixed_grippers=fixed,
        finger_down=("right",),
    )
    clutch = Clutch(hull_models, params, fk=model_info.ee_poses_from_qpos)
    return Rig(clutch), params


def _up(quat) -> float:
    return float(frames.quat_to_matrix(np.asarray(quat, dtype=float))[:, 1] @ [0, 0, 1])


def test_the_finger_down_turn_waits_for_the_arm_then_eases_in(hull_models) -> None:
    rig, params = _finger_down_rig(hull_models)
    clutch = rig.clutch
    assert clutch.on_robot_state("right", _start_qpos(-0.4), epoch=1)  # Start Arm
    start = clutch.arms["right"].hold
    for _ in range(100):  # 0.5 s with the gripper not closing: the arm is not aligned
        result = rig.step()
    assert result.states["right"] == "setup"
    assert result.targets["right"][7] == 0.0
    np.testing.assert_allclose(result.targets["right"][3:7], start[3:7], atol=1e-6)

    arms = _ArmEcho(rig)  # now the arm closes it: aligned, the turn starts
    targets = []
    for _ in range(round(5.0 / DT)):
        result = arms.step()
        targets.append(result.targets["right"].astype(float))
        if result.states["right"] != "setup":
            break
    assert result.states["right"] in (LOCKED, RELEASED)
    assert _up(result.targets["right"][3:7]) == pytest.approx(1.0, abs=1e-5)
    turns = frames.quat_angle(
        frames.quat_delta(np.array(targets[:-1])[:, 3:7], np.array(targets[1:])[:, 3:7])
    )
    assert float(np.max(turns)) <= 1.5 * params.turn_speed * DT * 1.05
    opening = 0.6 * hull_models["right"].gripper_range[0]
    assert result.targets["right"][7] == pytest.approx(opening)  # aligned: open to 60%


def test_a_finger_down_arm_ignores_the_hands_roll_but_follows_its_tilt(hull_models) -> None:
    rig, _ = _finger_down_rig(hull_models)
    rig.clutch.on_robot_state("right", _start_qpos(-0.4), epoch=1)
    arms = _ArmEcho(rig)
    arms.until_left(RELEASED)  # the left arm is not constrained; just tick past the turn
    while arms.step().states["right"] != RELEASED:
        pass
    held = rig.clutch.arms["right"].hold
    rig.grips["right"] = 1.0
    arms.step(2)

    def turned(axis, degrees):
        q = frames.quat_from_axis_angle(axis, math.radians(degrees))
        rig.hands["right"][3:7] = frames.quat_multiply(q, [1.0, 0.0, 0.0, 0.0])
        return arms.step(3).targets["right"].astype(float)

    # A roll about the fingers (forward, +x) is dropped: nothing turns.
    rolled = turned([1.0, 0.0, 0.0], 40)
    assert rig.clutch.states()["right"] == ENGAGED
    np.testing.assert_allclose(np.abs(rolled[3:7] @ held[3:7]), 1.0, atol=1e-6)
    # Tilting the fingers down is followed, with the finger on the right kept lowest.
    tilted = turned([0.0, 1.0, 0.0], 30)
    approach = -frames.quat_to_matrix(tilted[3:7])[:, 2]
    assert approach[2] == pytest.approx(-math.sin(math.radians(30)), abs=1e-5)
    assert _up(tilted[3:7]) == pytest.approx(math.cos(math.radians(30)), abs=1e-5)


def test_finger_down_is_validated(hull_models) -> None:
    with pytest.raises(ValueError):
        ClutchParams(finger_down=("middle",))
    with pytest.raises(ValueError):
        ClutchParams(turn_speed=0.0)
    with pytest.raises(ValueError):
        Clutch(hull_models, ClutchParams(finger_down=("left",)), intro="left")


def test_node_args_read_finger_down_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("FINGER_DOWN", "right")
    args = clutch_node.build_parser().parse_args([])
    assert clutch_node.params_from_args(args).finger_down == ("right",)
    monkeypatch.setenv("FINGER_DOWN", "up")
    with pytest.raises(SystemExit):
        clutch_node.build_parser().parse_args([])


def test_a_closed_fixed_gripper_stays_closed_after_the_intro(hull_models) -> None:
    """FIXED_GRIPPERS left=0: the intro opens it halfway and closes it, then nothing opens it."""
    params = ClutchParams(
        clamp_table=False, clamp_reach=False, clamp_midline=False, fixed_grippers=(("left", 0.0),)
    )
    clutch = Clutch(hull_models, params, fk=model_info.ee_poses_from_qpos, intro="left")
    rig = Rig(clutch)
    clutch.on_robot_state("left", _start_qpos(0.4), epoch=1)
    arms = _ArmEcho(rig)
    assert arms.until_left("waiting").targets["left"][7] == pytest.approx(0.3927, abs=1e-4)
    clutch.proceed()
    arms.until_left(RELEASED)
    rig.grips["left"] = 1.0
    arms.step(2)
    # Pulled all the way (which would pick the gripper up) and then let go open.
    for gripper in (0.0, 0.4, 0.78):
        rig.hands["left"][7] = gripper
        result = arms.step(5)
        assert result.states["left"] == ENGAGED
        assert result.targets["left"][7] == 0.0
