"""Gravity compensation and soft gains for dora-openarm (docs/teleop.md, Impedance)."""

from __future__ import annotations

import math

import numpy as np
import openarm_can as oa
import pytest
from openarm_driver import Config

from dora_openarm_teleop import impedance_arm
from dora_openarm_teleop.gravity import ArmGravity
from dora_openarm_teleop.impedance_arm import (
    DEFAULT_FRICTION,
    DEFAULT_KP,
    FRICTION_SPEED,
    TAU_LIMIT,
    VELOCITY_LIMIT,
    CommandVelocity,
    FeedforwardOpenArm,
    ImpedanceSettings,
    LatencyProbe,
    feedforward,
    impedance_driver,
)


def start_qpos(gripper: float = 0.0) -> np.ndarray:
    """dora-openarm's start pose: J4 at 90 deg, the rest 0."""
    return np.array([0.0, 0.0, 0.0, math.pi / 2, 0.0, 0.0, 0.0, gripper])


def test_gravity_at_the_start_pose() -> None:
    """The forearm sticks out forward: the shoulder (J1) and the elbow (J4) hold it."""
    left = ArmGravity("left").torques(start_qpos())
    right = ArmGravity("right").torques(start_qpos())
    assert left[0] == pytest.approx(-4.29, abs=0.05)
    assert left[3] == pytest.approx(4.38, abs=0.05)
    # Mirrored arms: J1 turns the other way on the right, J4 bends the same way.
    assert right[0] == pytest.approx(-left[0], abs=1e-6)
    assert right[3] == pytest.approx(left[3], abs=1e-6)


def test_a_hanging_arm_needs_almost_no_torque() -> None:
    """The links' centres of mass sit a little off their axes: under 0.1 Nm at joint zero."""
    np.testing.assert_allclose(ArmGravity("left").torques(np.zeros(8)), 0.0, atol=0.1)


def test_gravity_models_do_not_share_state() -> None:
    a, b = ArmGravity("left"), ArmGravity("left")
    before = a.torques(start_qpos())
    b.torques(np.zeros(8))
    np.testing.assert_allclose(a.torques(start_qpos()), before)


@pytest.mark.parametrize("bad", [np.zeros(6), np.full(8, np.nan)])
def test_gravity_rejects_bad_joints(bad) -> None:
    with pytest.raises(ValueError):
        ArmGravity("left").torques(bad)


def test_feedforward_scales_clips_and_waits_for_a_state() -> None:
    gravity = ArmGravity("left")
    full = feedforward(gravity, ImpedanceSettings(), start_qpos())
    half = feedforward(gravity, ImpedanceSettings(gravity_scale=0.5), start_qpos())
    np.testing.assert_allclose(half, 0.5 * full)
    clipped = feedforward(
        gravity, ImpedanceSettings(tau_limit=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)), start_qpos()
    )
    assert np.max(np.abs(clipped)) == pytest.approx(1.0)
    np.testing.assert_array_equal(feedforward(gravity, ImpedanceSettings(), None), np.zeros(7))


class FakeArm:
    def __init__(self) -> None:
        self.sent: list[list] = []

    def mit_control_all(self, params) -> None:
        self.sent.append(params)

    def get_motors(self) -> str:
        return "motors"


class FakeOpenArm:
    def __init__(self) -> None:
        self.arm = FakeArm()
        self.enabled = False

    def get_arm(self) -> FakeArm:
        return self.arm

    def enable_all(self) -> None:
        self.enabled = True


def test_feedforward_is_added_to_every_mit_command_and_nothing_else_changes() -> None:
    openarm = FakeOpenArm()
    wrapped = FeedforwardOpenArm(openarm, lambda: np.arange(7, dtype=float))
    params = [oa.MITParam(10.0 + i, 0.5, 0.1 * i, 0.0, 0.25) for i in range(7)]
    wrapped.get_arm().mit_control_all(params)
    sent = openarm.arm.sent[-1]
    assert [p.tau for p in sent] == [0.25 + i for i in range(7)]
    assert [(p.kp, p.kd, p.q, p.dq) for p in sent] == [(p.kp, p.kd, p.q, p.dq) for p in params]
    # Everything else is the real object's.
    wrapped.enable_all()
    assert openarm.enabled
    assert wrapped.get_arm().get_motors() == "motors"


class FakeDriver:
    """openarm_driver.SingleArmDriver's constructor and the state it keeps."""

    def __init__(self, arm_side, config=None, kps=None, kds=None, can_interface=None) -> None:
        self.arm_side = arm_side
        self.config = config
        self.kps, self.kds = kps, kds
        self.can_interface = can_interface
        self.openarm = FakeOpenArm()
        self.latest_state = {"qpos": start_qpos(0.4)}
        self.joint_offsets = np.zeros(8)


def test_the_driver_uses_soft_arm_gains_and_keeps_the_gripper_gains() -> None:
    config = Config("openarm_pedestal")
    driver_class = impedance_driver(FakeDriver, ImpedanceSettings(), lambda: config)
    driver = driver_class("left_arm", can_interface="can1")
    np.testing.assert_allclose(driver.kps[:7], DEFAULT_KP)
    assert driver.kps[7] == config.get_default_kps()[7]
    np.testing.assert_allclose(driver.kds, config.get_default_kds())
    assert driver.can_interface == "can1"

    driver.openarm.get_arm().mit_control_all([oa.MITParam(1, 1, 0, 0, 0) for _ in range(7)])
    sent = driver.openarm.arm.sent[-1]
    expected = ArmGravity("left").torques(start_qpos(0.4))
    np.testing.assert_allclose([p.tau for p in sent], expected, atol=1e-5)

    driver.latest_state = None  # before the first state: no feedforward
    driver.openarm.get_arm().mit_control_all([oa.MITParam(1, 1, 0, 0, 0) for _ in range(7)])
    assert [p.tau for p in driver.openarm.arm.sent[-1]] == [0.0] * 7


def test_explicit_gains_win_over_the_settings() -> None:
    config = Config("openarm_pedestal")
    driver = impedance_driver(FakeDriver, ImpedanceSettings(kd=(1,) * 7), lambda: config)(
        "right_arm", config, kps=np.ones(8)
    )
    np.testing.assert_array_equal(driver.kps, np.ones(8))
    np.testing.assert_allclose(driver.kds[:7], 1.0)


def test_args_come_from_the_environment(monkeypatch) -> None:
    args = impedance_arm.build_parser().parse_args([])
    settings = impedance_arm.settings_from_args(args)
    assert settings == ImpedanceSettings()
    assert settings.tau_limit == TAU_LIMIT

    monkeypatch.setenv("IMPEDANCE_KP", "30,30,20,20,5,5,5")
    monkeypatch.setenv("IMPEDANCE_KD", "2,2,1,1,0.5,0.5,0.4")
    monkeypatch.setenv("GRAVITY_SCALE", "0.8")
    settings = impedance_arm.settings_from_args(impedance_arm.build_parser().parse_args([]))
    assert settings.kp == (30, 30, 20, 20, 5, 5, 5)
    assert settings.kd == (2, 2, 1, 1, 0.5, 0.5, 0.4)
    assert settings.gravity_scale == 0.8

    monkeypatch.setenv("IMPEDANCE", "off")
    assert impedance_arm.settings_from_args(impedance_arm.build_parser().parse_args([])) is None

    monkeypatch.setenv("IMPEDANCE_KP", "30,30")
    with pytest.raises(SystemExit):
        impedance_arm.build_parser().parse_args([])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kp": (0.0,) * 7},
        {"kp": (1.0,) * 6},
        {"kd": (-1.0,) * 7},
        {"gravity_scale": 2.0},
        {"tau_limit": (0.0,) * 7},
    ],
)
def test_settings_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        ImpedanceSettings(**kwargs)


# --- velocity feedforward and the latency probe -------------------------------


class Clock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _ramp(velocity: CommandVelocity, clock: Clock, v: float, dt: float, n: int, q0: float = 0.0):
    out = None
    for i in range(1, n + 1):
        clock.t += dt
        out = velocity([q0 + v * dt * i] * 7)
    return out


@pytest.mark.parametrize("dt", [0.004, 0.02])  # the gate's 250 Hz, the driver's 50 Hz moves
def test_the_command_velocity_is_the_rate_the_commands_change_at(dt: float) -> None:
    clock = Clock()
    velocity = CommandVelocity(VELOCITY_LIMIT, clock=clock)
    np.testing.assert_allclose(_ramp(velocity, clock, 0.5, dt, 20), 0.5, rtol=1e-9)


def test_commands_arriving_together_do_not_make_a_spike() -> None:
    clock = Clock()
    velocity = CommandVelocity(VELOCITY_LIMIT, clock=clock)
    _ramp(velocity, clock, 0.5, 0.004, 20)
    q = 0.5 * 0.004 * 20
    clock.t += 0.0001  # the next command a tenth of a millisecond later
    out = velocity([q + 0.5 * 0.004] * 7)
    assert float(np.max(out)) < 0.6


def test_a_pause_starts_the_command_velocity_afresh_and_it_is_clipped() -> None:
    clock = Clock()
    velocity = CommandVelocity(VELOCITY_LIMIT, clock=clock)
    _ramp(velocity, clock, 0.5, 0.004, 20)
    clock.t += 0.2  # 200 ms without a command
    np.testing.assert_array_equal(velocity([5.0] * 7), np.zeros(7))
    fast = _ramp(velocity, clock, 10.0, 0.004, 20, q0=5.0)
    np.testing.assert_allclose(fast, VELOCITY_LIMIT)


def test_the_command_velocity_is_sent_as_dq() -> None:
    openarm = FakeOpenArm()
    wrapped = FeedforwardOpenArm(openarm, lambda: np.zeros(7), lambda q: np.full(7, 0.3))
    wrapped.get_arm().mit_control_all([oa.MITParam(20.0, 2.0, 0.1, 0.0, 0.0) for _ in range(7)])
    sent = openarm.arm.sent[-1]
    assert [p.dq for p in sent] == pytest.approx([0.3] * 7)
    assert [(p.kp, p.kd, p.q) for p in sent] == [(20.0, 2.0, pytest.approx(0.1))] * 7
    half = FeedforwardOpenArm(
        openarm, lambda: np.zeros(7), lambda q: np.full(7, 0.3), velocity_ff=0.5
    )
    half.get_arm().mit_control_all([oa.MITParam(20.0, 2.0, 0.1, 0.0, 0.0) for _ in range(7)])
    assert [p.dq for p in openarm.arm.sent[-1]] == pytest.approx([0.15] * 7)


@pytest.mark.parametrize(("velocity", "sign"), [(0.5, 1.0), (-0.5, -1.0), (0.0, 0.0)])
def test_friction_is_added_along_the_commands_motion(velocity: float, sign: float) -> None:
    openarm = FakeOpenArm()
    gravity = np.arange(7, dtype=float)
    friction = np.array(DEFAULT_FRICTION)
    wrapped = FeedforwardOpenArm(
        openarm, lambda: gravity, lambda q: np.full(7, velocity), friction=friction
    )
    wrapped.get_arm().mit_control_all([oa.MITParam(20.0, 2.0, 0.0, 0.0, 0.0) for _ in range(7)])
    tau = np.array([p.tau for p in openarm.arm.sent[-1]])
    np.testing.assert_allclose(tau, gravity + sign * friction, atol=1e-6)


@pytest.mark.parametrize(
    ("measured", "share"),
    [
        (-0.2, 1.0),  # trails by more than friction / kp (0.52 / 20 = 0.026 rad): all of it
        (-0.013, 0.5),  # half of that: half
        (0.0, 0.0),  # caught up: none
        (0.05, 0.0),  # ahead of the command: none, rather than a push past it
    ],
)
def test_friction_is_added_only_while_the_joint_trails_its_command(
    measured: float, share: float
) -> None:
    openarm = FakeOpenArm()
    friction = np.array(DEFAULT_FRICTION)
    wrapped = FeedforwardOpenArm(
        openarm,
        lambda: np.zeros(7),
        lambda q: np.full(7, 1.0),  # moving up, fast enough for all of tanh
        friction=friction,
        measured=lambda: np.full(7, measured),
    )
    kp = np.array(DEFAULT_KP)
    wrapped.get_arm().mit_control_all([oa.MITParam(k, 1.0, 0.0, 0.0, 0.0) for k in kp])
    tau = np.array([p.tau for p in openarm.arm.sent[-1]])
    expected = friction * np.clip(-measured / (friction / kp), 0.0, 1.0)
    np.testing.assert_allclose(tau, expected, atol=1e-6)
    assert tau[0] == pytest.approx(share * friction[0], abs=1e-3)


def test_friction_waits_for_the_first_state() -> None:
    openarm = FakeOpenArm()
    wrapped = FeedforwardOpenArm(
        openarm,
        lambda: np.zeros(7),
        lambda q: np.ones(7),
        friction=[1.0] * 7,
        measured=lambda: None,
    )
    wrapped.get_arm().mit_control_all([oa.MITParam(1, 1, 0, 0, 0) for _ in range(7)])
    assert [p.tau for p in openarm.arm.sent[-1]] == [0.0] * 7


def test_friction_fades_in_with_the_commanded_speed() -> None:
    openarm = FakeOpenArm()
    wrapped = FeedforwardOpenArm(
        openarm, lambda: np.zeros(7), lambda q: np.full(7, FRICTION_SPEED), friction=[1.0] * 7
    )
    wrapped.get_arm().mit_control_all([oa.MITParam(1, 1, 0, 0, 0) for _ in range(7)])
    assert openarm.arm.sent[-1][0].tau == pytest.approx(math.tanh(1.0), abs=1e-6)


def test_the_driver_sends_no_dq_when_the_velocity_feedforward_is_off() -> None:
    config = Config("openarm_pedestal")
    driver = impedance_driver(FakeDriver, ImpedanceSettings(velocity_ff=0.0), lambda: config)(
        "left_arm"
    )
    for q in (0.0, 0.1, 0.2):
        driver.openarm.get_arm().mit_control_all([oa.MITParam(1, 1, q, 0, 0) for _ in range(7)])
    assert [p.dq for p in driver.openarm.arm.sent[-1]] == [0.0] * 7


class FakeNode:
    def __init__(self, events) -> None:
        self.events = events
        self.sent = []

    def __iter__(self):
        return iter(self.events)

    def send_output(self, *args) -> None:
        self.sent.append(args)


def test_the_latency_probe_reports_how_late_commands_arrive() -> None:
    now_ns = 1_000_000_000_000
    events = [
        {"type": "INPUT", "id": "move_position", "metadata": {"timestamp": now_ns - 5_000_000}},
        {"type": "INPUT", "id": "request_state", "metadata": {}},
        {"type": "INPUT", "id": "move_position", "metadata": {"timestamp": now_ns - 15_000_000}},
    ]
    ticks = iter(range(100))  # each reading of the clock is a millisecond later
    logs: list[str] = []
    node = FakeNode(events)
    probe = LatencyProbe(
        node, 0.0035, logs.append, clock_ns=lambda: now_ns, monotonic=lambda: next(ticks) / 1000
    )
    probe.send_output("state", 1)  # passed through
    assert list(probe) == events
    assert node.sent == [("state", 1)]
    # Reported once the period is over: after the second event, before the third.
    assert len(logs) == 1
    assert "1 move_position, gate -> here median 5.0 ms" in logs[0]
    assert "2 events" in logs[0]


def test_velocity_and_latency_args_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("IMPEDANCE_VELOCITY_FF", "0.5")
    monkeypatch.setenv("LATENCY_LOG_S", "0")
    args = impedance_arm.build_parser().parse_args([])
    assert impedance_arm.settings_from_args(args).velocity_ff == 0.5
    assert args.latency_log_s == 0.0
    with pytest.raises(ValueError):
        ImpedanceSettings(velocity_ff=1.5)
    monkeypatch.setenv("IMPEDANCE_FRICTION", "0,0,0,0,0,0,0")
    args = impedance_arm.build_parser().parse_args([])
    assert impedance_arm.settings_from_args(args).friction == (0.0,) * 7
    assert (
        impedance_arm.settings_from_args(
            impedance_arm.build_parser().parse_args(["--friction", ""])
        ).friction
        == DEFAULT_FRICTION
    )
    for bad in ((-0.1,) * 7, (5.0,) * 7):
        with pytest.raises(ValueError):
            ImpedanceSettings(friction=bad)
