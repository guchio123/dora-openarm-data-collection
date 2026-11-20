"""What the headset HUD says (docs/teleop.md, "HUD"). Pure logic, injected time.

The data collection UI owns the episodes; this only reads its ``command`` stream
(``start`` with ``episode_number`` metadata, ``success`` / ``fail`` / ``cancel`` /
``quit``) next to the clutch's and the gate's state, so an operator wearing the
headset knows whether they are recording and why an arm is not following without
taking it off. The vocabulary is openarm-quest-teleop's:

* clutch: ``FOLLOW`` engaged, ``HOLD`` released, ``REGRIP`` locked (let go and
  squeeze again), ``LATE`` stale (input is late; keep holding still, the clutch
  resumes by itself), ``--`` nothing received yet. ``LATE`` and ``REGRIP`` ask for
  different things, so they never share a prefix. An arm running its intro shows
  ``SETUP`` while it moves and ``PRESS X`` while it waits for the left
  controller's X (or Enter in the UI).
* gate: only arms that are not engaged, as ``GATE L:HOLDING(jump)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

MAX_LINES = 5
WIDTH = 40
NOTICE_S = 2.0
MIN_INTERVAL_S = 0.2
REPEAT_S = 2.0

CLUTCH_WORDS = {
    "engaged": "FOLLOW",
    "released": "HOLD",
    "locked": "REGRIP",
    "stale": "LATE",
    "setup": "SETUP",
    "waiting": "PRESS X",
}
SIDE_LETTERS = (("left", "L"), ("right", "R"))


def ascii_line(text: str) -> str:
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in text)[:WIDTH]


@dataclass
class Hud:
    """State machine: feed events, ask :meth:`message` on every tick."""

    recording: bool = False
    quitting: bool = False
    episode: int | None = None
    started_at: float = 0.0
    clutch: dict[str, str] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    notice: str = ""
    notice_until: float = 0.0
    haptic: str | None = None
    _sent: str | None = None
    _sent_at: float = float("-inf")

    def on_command(self, command: str, metadata: dict[str, Any], now: float) -> None:
        if command == "start":
            self.recording = True
            self.started_at = now
            number = metadata.get("episode_number")
            self.episode = int(number) if isinstance(number, int) else None
            self.haptic = "right"
        elif command in ("success", "fail", "cancel"):
            if self.recording:
                verb = "canceled" if command == "cancel" else f"saved: {command}"
                self._notify(f"ep {self._number()} {verb}", now)
                self.haptic = "left" if command == "cancel" else "right"
            self.recording = False
        elif command == "quit":
            self.quitting = True
            self.recording = False
            # Also forces the last HUD out before the node exits.
            self.haptic = "both"

    def on_clutch_state(self, text: str) -> None:
        state = _json_object(text)
        if state is not None:
            self.clutch = {k: str(v) for k, v in state.items()}

    def on_gate_state(self, text: str) -> None:
        state = _json_object(text)
        if state is not None:
            self.gate = state

    def lines(self, now: float) -> list[str]:
        lines = [self._status(now)]
        if self.notice and now < self.notice_until:
            lines.append(self.notice)
        lines.append(
            "  ".join(
                f"{letter}:{CLUTCH_WORDS.get(self.clutch.get(side, ''), '--')}"
                for side, letter in SIDE_LETTERS
            )
        )
        gate = self._gate_line()
        if gate:
            lines.append(gate)
        return [ascii_line(line) for line in lines[:MAX_LINES]]

    def message(self, now: float) -> str | None:
        """The HUD JSON to send now, or None (unchanged and sent recently)."""
        document = {"lines": self.lines(now), "recording": self.recording}
        text = json.dumps(document, separators=(",", ":"))
        changed = text != self._sent
        if self.haptic is None:
            if not changed and now - self._sent_at < REPEAT_S:
                return None
            if changed and now - self._sent_at < MIN_INTERVAL_S:
                return None
        else:
            document["haptic"] = self.haptic
            self.haptic = None
        self._sent, self._sent_at = text, now
        return json.dumps(document, separators=(",", ":"))

    def _status(self, now: float) -> str:
        if self.quitting:
            return "QUIT"
        if self.recording:
            elapsed = max(0, int(now - self.started_at))
            return f"REC {elapsed // 60:02d}:{elapsed % 60:02d}  ep {self._number()}"
        return "READY  [A] start  hold [B] quit"

    def _gate_line(self) -> str:
        parts = []
        for side, letter in SIDE_LETTERS:
            arm = self.gate.get(side)
            if not isinstance(arm, dict) or arm.get("state") in (None, "engaged"):
                continue
            reason = arm.get("reason") or ""
            parts.append(f"{letter}:{str(arm['state']).upper()}({reason})")
        return ("GATE " + " ".join(parts)) if parts else ""

    def _notify(self, text: str, now: float) -> None:
        self.notice, self.notice_until = text, now + NOTICE_S

    def _number(self) -> str:
        return "?" if self.episode is None else str(self.episode)


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None
