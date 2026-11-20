"""The headset HUD and the clutch's robot-aware table clamp."""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

from dora_openarm_teleop import clutch_node, hud_node
from dora_openarm_teleop.hud import WIDTH, Hud


def lines(message: str | None) -> list[str]:
    assert message is not None
    return json.loads(message)["lines"]


def test_status_follows_the_ui_commands():
    hud = Hud()
    assert lines(hud.message(0.0))[0].startswith("READY")
    hud.on_command("start", {"episode_number": 7}, 1.0)
    first = json.loads(hud.message(1.0))
    assert first["lines"][0] == "REC 00:00  ep 7"
    assert first["recording"] is True and first["haptic"] == "right"
    assert lines(hud.message(66.0))[0] == "REC 01:05  ep 7"
    hud.on_command("cancel", {}, 70.0)
    done = json.loads(hud.message(70.0))
    assert done["lines"][:2] == ["READY  [A] start  hold [B] quit", "ep 7 canceled"]
    assert done["haptic"] == "left"
    # The notice goes after two seconds.
    assert "ep 7 canceled" not in lines(hud.message(72.5))


def test_a_finish_without_a_start_says_nothing():
    hud = Hud()
    hud.message(0.0)
    hud.on_command("success", {}, 1.0)
    assert hud.message(1.0) is None  # unchanged, sent 1 s ago


def test_clutch_words_tell_late_from_regrip():
    hud = Hud()
    hud.on_clutch_state(json.dumps({"right": "stale", "left": "locked"}))
    assert lines(hud.message(0.0))[1] == "L:REGRIP  R:LATE"
    hud.on_clutch_state(json.dumps({"right": "engaged", "left": "released"}))
    assert lines(hud.message(1.0))[1] == "L:HOLD  R:FOLLOW"


def test_clutch_words_for_the_intro():
    hud = Hud()
    hud.on_clutch_state(json.dumps({"right": "released", "left": "setup"}))
    assert lines(hud.message(0.0))[1] == "L:SETUP  R:HOLD"
    hud.on_clutch_state(json.dumps({"right": "released", "left": "waiting"}))
    assert lines(hud.message(1.0))[1] == "L:PRESS X  R:HOLD"


def test_gate_line_only_names_arms_that_are_not_engaged():
    hud = Hud()
    hud.on_gate_state(
        json.dumps(
            {
                "right": {"state": "engaged", "reason": ""},
                "left": {"state": "holding", "reason": "not_tracking"},
            }
        )
    )
    assert lines(hud.message(0.0))[-1] == "GATE L:HOLDING(not_tracking)"
    hud.on_gate_state(json.dumps({"right": {"state": "engaged"}, "left": {"state": "engaged"}}))
    assert not any(line.startswith("GATE") for line in lines(hud.message(1.0)))


def test_rate_limit_and_repeat():
    hud = Hud()
    assert hud.message(0.0) is not None
    hud.on_clutch_state(json.dumps({"right": "engaged"}))
    assert hud.message(0.1) is None  # changed, but within 0.2 s
    assert hud.message(0.25) is not None
    assert hud.message(1.0) is None  # unchanged
    assert hud.message(2.3) is not None  # resent for a page that reconnected


def test_lines_are_ascii_and_fit():
    hud = Hud()
    hud.on_gate_state(
        json.dumps({s: {"state": "holding", "reason": "x" * 60 + "é"} for s in ("right", "left")})
    )
    for line in lines(hud.message(0.0)):
        assert len(line) <= WIDTH and line.isascii()


def test_malformed_state_is_ignored():
    hud = Hud()
    hud.on_clutch_state("not json")
    hud.on_gate_state("[1, 2]")
    assert lines(hud.message(0.0))[1] == "L:--  R:--"


def test_node_quit_forces_the_last_hud_out():
    hud = Hud()
    hud.message(0.0)
    event = {"type": "INPUT", "id": "command", "value": pa.array(["quit"]), "metadata": {}}
    message = hud_node.handle(hud, event, 0.05)
    assert lines(message)[0] == "QUIT" and hud.quitting


@pytest.mark.parametrize(
    ("environ", "clamped", "table_z"),
    [
        ({"ROBOT": "mujoco"}, True, None),
        ({"ROBOT": "pedestal"}, False, None),
        ({}, False, None),
        ({"ROBOT": "pedestal", "TABLE_Z": "-0.30"}, True, -0.30),
        ({"ROBOT": "mujoco", "TABLE_Z": ""}, True, None),
    ],
)
def test_clutch_table_clamp_follows_robot(environ, clamped, table_z):
    args = clutch_node.build_parser().parse_args(["--scene", "demo"])
    clutch_node.apply_robot(args, environ)
    assert (not args.no_table_clamp) is clamped
    assert args.table_z == table_z


def test_an_explicit_table_z_wins_over_the_environment():
    args = clutch_node.build_parser().parse_args(["--table-z", "-0.2"])
    clutch_node.apply_robot(args, {"ROBOT": "cell", "TABLE_Z": "-0.3"})
    assert args.table_z == -0.2 and not args.no_table_clamp
