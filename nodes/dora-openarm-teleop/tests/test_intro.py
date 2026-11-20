"""An arm's opening sequence (docs/teleop.md, Intro), checked against the MJCF's fingers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dora_openarm_teleop import frames, model_info
from dora_openarm_teleop.intro import (
    CLOSE,
    DONE,
    FLIP,
    GRIP,
    OPEN,
    PRESENT,
    SETUP,
    WAIT,
    WAITING,
    Intro,
    IntroParams,
    finger_down,
    level_approach,
)

SCENE = "demo"  # modules/leader-webxr.yaml's
DT = 0.004  # 250 Hz
ECHO = "echo"  # the measured gripper follows the target, as a free gripper does


def start_pose(side: str = "left", gripper: float = 0.4) -> np.ndarray:
    """POSE8 at dora-openarm's start pose (J4 at 90 deg, the rest 0)."""
    qpos16 = np.zeros(16)
    qpos16[3] = qpos16[11] = math.pi / 2
    return np.concatenate([model_info.ee_poses_from_qpos(qpos16, SCENE)[side], [gripper]])


def make(side: str = "left", params: IntroParams | None = None) -> Intro:
    return Intro(start_pose(side), model_info.gripper_ranges(SCENE)[side], params)


class Run:
    """Steps an intro at 250 Hz with a fake clock."""

    def __init__(self, intro: Intro, measured_start: float = 0.4) -> None:
        self.intro = intro
        self.t = 10.0
        self.targets: list[np.ndarray] = []
        self.measured_start = measured_start

    def step(self, measured: float | str | None = ECHO) -> np.ndarray:
        self.t += DT
        if measured == ECHO:
            measured = self.targets[-1][7] if self.targets else self.measured_start
        target = self.intro.step(self.t, measured)
        self.targets.append(target)
        return target

    def until(self, phase: str, measured: float | str | None = ECHO, limit_s: float = 30.0):
        end = self.t + limit_s
        while self.intro.phase != phase:
            assert self.t < end, f"stuck in {self.intro.phase} ({self.intro.detail})"
            self.step(measured)
        return self.targets[-1]


def fingers(side: str, target: np.ndarray) -> dict[str, np.ndarray]:
    """World offset of each finger's hull centroid from the site, at the target."""
    rotation = frames.quat_to_matrix(target[3:7])
    hulls = model_info.finger_hulls(side, SCENE)
    return {
        name: rotation @ hull.points_in_site(float(target[7])).mean(axis=0)
        for name, hull in zip(("inner", "outer"), hulls, strict=True)
    }


def normally(side: str, where: str) -> str:
    """The finger on the robot's `where` ("right": -y) at the start pose."""
    at_start = fingers(side, start_pose(side, 0.0))
    pick = min if where == "right" else max
    return pick(at_start, key=lambda name: at_start[name][1])


def lowest(side: str, target: np.ndarray) -> str:
    offsets = fingers(side, target)
    return min(offsets, key=lambda name: offsets[name][2])


def assert_gap_level(side: str, target: np.ndarray) -> None:
    """The fingers sit one above the other: the gap between them is level."""
    offsets = fingers(side, target)
    across = offsets["inner"] - offsets["outer"]
    assert abs(across[2]) / np.linalg.norm(across) > 0.999, across


def test_present_turns_the_finger_on_the_right_down_and_opens_halfway() -> None:
    run = Run(make())
    target = run.until(WAIT)
    start = start_pose()

    assert lowest("left", target) == normally("left", "right")
    assert_gap_level("left", target)
    np.testing.assert_allclose(target[:3], start[:3], atol=1e-9)
    _, high = model_info.gripper_ranges(SCENE)["left"]
    assert target[7] == pytest.approx(0.5 * high)
    assert run.intro.state == WAITING


def test_waits_for_enter_then_grips_and_turns_the_finger_on_the_left_down() -> None:
    run = Run(make())
    waiting = run.until(WAIT)
    for _ in range(500):  # 2 s: nothing moves while it waits
        np.testing.assert_array_equal(run.step(), waiting)
    assert run.intro.proceed()

    run.until(FLIP)
    gripped = run.targets[-2]
    assert gripped[7] == pytest.approx(0.0)  # closed before turning
    np.testing.assert_allclose(gripped[3:7], waiting[3:7], atol=1e-12)
    final = run.until(DONE)

    assert lowest("left", final) == normally("left", "left")
    assert_gap_level("left", final)
    assert final[7] == pytest.approx(0.0)
    np.testing.assert_allclose(final[:3], start_pose()[:3], atol=1e-9)
    assert run.intro.done


def test_the_flip_turns_back_through_the_start_orientation() -> None:
    """Only that way keeps J5 inside its range; the shortest arc is ambiguous at 180 deg."""
    run = Run(make())
    run.until(WAIT)
    run.intro.proceed()
    run.until(FLIP)
    flip = []
    while run.intro.phase == FLIP:
        flip.append(run.step())
    middle = flip[len(flip) // 2]
    turned = frames.quat_angle(frames.quat_delta(start_pose()[3:7], middle[3:7]))
    assert math.degrees(float(turned)) < 2.0


def test_the_target_moves_smoothly_at_the_turn_speed() -> None:
    params = IntroParams()
    run = Run(make(params=params))
    run.until(WAIT)
    run.intro.proceed()
    run.until(DONE)
    targets = np.array(run.targets)
    turns = frames.quat_angle(frames.quat_delta(targets[:-1, 3:7], targets[1:, 3:7]))
    # Smoothstep peaks at 1.5x the average speed.
    assert float(np.max(turns)) <= 1.5 * params.turn_speed * DT * 1.01
    low, high = model_info.gripper_ranges(SCENE)["left"]
    assert float(np.max(np.abs(np.diff(targets[:, 7])))) <= (high - low) * DT / params.gripper_s
    assert float(np.max(np.abs(np.diff(targets[:, :3], axis=0)))) == 0.0


def test_nothing_turns_until_the_measured_gripper_has_closed() -> None:
    """dora-openarm executes nothing until the commanded gripper closes (ALIGNED)."""
    run = Run(make())
    for _ in range(500):
        target = run.step(measured=0.4)  # the arm is not taking commands
    assert run.intro.phase == CLOSE
    assert "waiting for the arm to close the gripper" in run.intro.detail
    np.testing.assert_allclose(target[3:7], start_pose()[3:7], atol=1e-12)
    assert target[7] == pytest.approx(0.0)
    run.step(measured=0.01)
    assert run.intro.phase == PRESENT


def test_before_the_first_state_it_waits_too() -> None:
    run = Run(make())
    for _ in range(200):
        run.step(measured=None)
    assert run.intro.phase == CLOSE


def test_a_grip_that_closes_on_something_flips_after_the_timeout() -> None:
    params = IntroParams()
    run = Run(make(params=params))
    run.until(WAIT)
    run.intro.proceed()
    started = run.t
    run.until(FLIP, measured=0.3)  # held open by an object
    assert run.t - started == pytest.approx(params.close_timeout_s, abs=2 * DT)


def test_enter_is_ignored_unless_it_waits() -> None:
    run = Run(make())
    run.step()
    for phase in (CLOSE, PRESENT, OPEN):
        run.until(phase)
        assert not run.intro.proceed()
        assert run.intro.phase == phase
        assert run.intro.state == SETUP
    run.until(WAIT)
    assert run.intro.proceed()
    assert run.intro.phase == GRIP
    assert not run.intro.proceed()


def test_open_fraction_and_the_right_arm() -> None:
    """The right gripper opens towards negative angles; its finger on the right goes down."""
    run = Run(make("right", IntroParams(open_fraction=0.25)), measured_start=-0.4)
    target = run.until(WAIT)
    low, _ = model_info.gripper_ranges(SCENE)["right"]
    assert target[7] == pytest.approx(0.25 * low)
    assert lowest("right", target) == normally("right", "right")


def test_a_tilted_start_is_levelled() -> None:
    start = start_pose()
    tilt = frames.quat_from_axis_angle([0.0, 1.0, 0.0], math.radians(15))
    start[3:7] = frames.quat_multiply(tilt, start[3:7])
    intro = Intro(start, model_info.gripper_ranges(SCENE)["left"])
    target = Run(intro).until(WAIT)
    assert_gap_level("left", target)
    approach = -frames.quat_to_matrix(target[3:7])[:, 2]
    assert abs(approach[2]) < 1e-9
    np.testing.assert_allclose(approach, level_approach(start[3:7]), atol=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"open_fraction": 1.5},
        {"turn_speed": 0.0},
        {"gripper_s": float("nan")},
        {"closed_tol": -0.1},
        {"close_timeout_s": 0.1},
    ],
)
def test_params_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        IntroParams(**kwargs)


def test_start_must_be_a_finite_pose8() -> None:
    with pytest.raises(ValueError):
        Intro(np.zeros(7), (0.0, 0.7854))
    with pytest.raises(ValueError):
        Intro(np.full(8, np.nan), (0.0, 0.7854))


# --- finger_down (docs/teleop.md, Finger down) --------------------------------


def _site_y_up(quat: np.ndarray) -> float:
    return float(frames.quat_to_matrix(quat)[:, 1] @ [0.0, 0.0, 1.0])


def _approach(quat: np.ndarray) -> np.ndarray:
    return -frames.quat_to_matrix(quat)[:, 2]


@pytest.mark.parametrize("side", ["right", "left"])
def test_finger_down_turns_the_finger_on_the_right_to_the_bottom(side: str) -> None:
    start = start_pose(side, 0.0)
    turned = finger_down(start[3:7])
    assert _site_y_up(turned) == pytest.approx(1.0)
    np.testing.assert_allclose(_approach(turned), _approach(start[3:7]), atol=1e-12)
    assert lowest(side, np.concatenate([start[:3], turned, [0.0]])) == normally(side, "right")


def test_finger_down_keeps_a_tilted_approach_and_turns_as_far_as_it_allows() -> None:
    start = start_pose("right", 0.0)[3:7]
    tilt = frames.quat_from_axis_angle([0.0, 1.0, 0.0], math.radians(45))  # fingers 45 deg down
    tilted = frames.quat_multiply(tilt, start)
    turned = finger_down(tilted)
    np.testing.assert_allclose(_approach(turned), _approach(tilted), atol=1e-12)
    assert _site_y_up(turned) == pytest.approx(math.cos(math.radians(45)))


def test_finger_down_keeps_the_hands_roll_pointing_straight_down() -> None:
    start = start_pose("right", 0.0)[3:7]
    down = frames.quat_multiply(frames.quat_from_axis_angle([0.0, 1.0, 0.0], math.pi / 2), start)
    np.testing.assert_allclose(np.abs(finger_down(down) @ down), 1.0, atol=1e-9)


def test_finger_down_is_continuous_through_vertical() -> None:
    start = start_pose("right", 0.0)[3:7]
    previous = None
    for degrees in np.linspace(0.0, 120.0, 1201):  # forward, straight down, and past it
        tilt = frames.quat_from_axis_angle([0.0, 1.0, 0.0], math.radians(degrees))
        turned = finger_down(frames.quat_multiply(tilt, start))
        if previous is not None:
            step = math.degrees(float(frames.quat_angle(frames.quat_delta(previous, turned))))
            assert step < 2.0, f"{degrees:.1f} deg: jumped {step:.2f} deg"
        previous = turned
