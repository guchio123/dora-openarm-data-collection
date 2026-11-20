# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""dora-rs node that provides UI to control data collection with OpenArm."""

import argparse
import asyncio
import collections
import dataclasses
import datetime
import json
import os
import pathlib
import time
from collections.abc import AsyncIterable
from contextlib import asynccontextmanager

import dora
import pyarrow as pa
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.templating import Jinja2Templates

base_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=f"{base_dir}/templates")

node = None

auto_open = False
port = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Open a Web browser automatically if requested."""
    if auto_open:
        url = f"http://127.0.0.1:{port}"
        await asyncio.create_subprocess_exec("open", url)
    yield


app = FastAPI(lifespan=_lifespan)


@dataclasses.dataclass
class State:
    """The current state."""

    collecting: bool = False
    running: bool = True
    episode_number: int = 0
    task_index: int = 0
    task_title: str = ""
    arm_status_right: str = "stopped"
    arm_status_left: str = "stopped"


state = State()

_state_changed = asyncio.Condition()

# Monotonically incremented on every state change.
state_version = 0


CAMERA_INPUTS = (
    "camera_wrist_right",
    "camera_wrist_left",
    "camera_head_left",
    "camera_head_right",
    "camera_ceiling",
    "camera_head_depth",
)

# JPEG streams shown as live images, e.g. colorized depth. Only the latest
# frame is kept; the browser pulls it as MJPEG from /preview/<input id>.
PREVIEW_INPUTS = ("camera_head_depth_preview",)
PREVIEW_INTERVAL_S = 0.05

CAMERA_TIMESTAMP_WINDOW = 60
CAMERA_STALE_AFTER_S = 1.0

# dora-openarm status inputs, one per arm. The input id matches the State field
ARM_STATUS_INPUTS = ("arm_status_right", "arm_status_left")

# dora-openarm state inputs, one per arm, carrying per-motor MOSFET and rotor
# temperatures. They arrive at the leader's rate (250 Hz), so the handler only
# stores the latest sample; the browser is fed from /arm-health instead.
ARM_STATE_INPUTS = {"arm_state_right": "right", "arm_state_left": "left"}

# A reading older than this is shown as unknown rather than as the last value,
# so a follower that stopped answering does not look merely cool.
ARM_TEMPERATURE_STALE_AFTER_S = 3.0

# Shortest gap between two parses of an arm's state. The arms report at the
# leader's rate and the browser is fed every 500 ms, so converting every
# message out of Arrow would be ~50x more work than anyone can see.
ARM_STATE_PARSE_INTERVAL_S = 0.2

# The leader's own stream. `ker_metadata` names the device and is sent once
# at startup, so it is remembered rather than polled; `ker_position` is only
# read for its arrival time, the same way the camera rows are.
KER_METADATA_INPUT = "ker_metadata"
KER_POSITION_INPUT = "ker_position"
KER_TIMESTAMP_WINDOW = 120
KER_STALE_AFTER_S = 1.0

# Which inputs have ever delivered something. A dataflow wires up only the
# nodes it uses, and a node is never told what it was wired to, so a panel
# earns its place on screen by having produced data at least once.
seen_inputs: set[str] = set()

# VR packet arrival times (ns) published by udp-receiver as
# `vr_receive_times` or `vr_recv_ts` (deprecated).
VR_RECEIVE_TIMES_INPUTS = ("vr_receive_times", "vr_recv_ts")

VR_TIMESTAMP_WINDOW = 120  # ~1.6 s of history at 72 Hz
VR_STALE_AFTER_S = 1.0


@dataclasses.dataclass
class StreamStats:
    """Rolling rate / jitter stats for one input stream (camera, VR, leader)."""

    fps: float = 0.0
    jitter_ms: float = 0.0


camera_stats: dict[str, StreamStats] = {name: StreamStats() for name in CAMERA_INPUTS}
# Latest JPEG of each preview input and a counter bumped on every frame, so a
# stream sends a frame only when it changed.
previews: dict[str, bytes] = {}
preview_versions: dict[str, int] = {name: 0 for name in PREVIEW_INPUTS}
camera_timestamps: dict[str, collections.deque] = {
    name: collections.deque(maxlen=CAMERA_TIMESTAMP_WINDOW) for name in CAMERA_INPUTS
}

vr_stats = StreamStats()
vr_timestamps: collections.deque = collections.deque(maxlen=VR_TIMESTAMP_WINDOW)

ker_stats = StreamStats()
ker_timestamps: collections.deque = collections.deque(maxlen=KER_TIMESTAMP_WINDOW)
ker_device: dict = {}


@dataclasses.dataclass
class ArmHealth:
    """Latest per-motor and bus readings for one arm.

    Temperatures are in Celsius: `mos` is the driver MOSFET, `rotor` the
    motor itself. Both are kept because either can be the one that
    overheats -- the MOSFET leads under a sustained current, the rotor lags
    but stays hot for longer.

    `motor_status` is what each motor says about itself, or "SILENT" for one
    that stopped answering. A motor that is merely unplugged raises no bus
    error at all, so this is the only place it shows.

    `bus` holds openarm_can's per-interface counters. They latch, so what an
    operator wants -- what happened during this episode -- is the difference
    against `bus_baseline`, which is reset when an episode starts.
    """

    mos: list[int] = dataclasses.field(default_factory=list)
    rotor: list[int] = dataclasses.field(default_factory=list)
    motor_status: list[str] = dataclasses.field(default_factory=list)
    bus: dict = dataclasses.field(default_factory=dict)
    bus_baseline: dict = dataclasses.field(default_factory=dict)
    updated_at: float = 0.0


arm_health: dict[str, ArmHealth] = {
    side: ArmHealth() for side in ARM_STATE_INPUTS.values()
}

# Bus counters worth surfacing, worst last: the badge shows the heaviest one
# that moved, so the order here is the severity order.
BUS_COUNTER_SEVERITY = (
    ("error_warning", "WARNING"),
    ("tx_overflow", "OVERFLOW"),
    ("rx_overflow", "OVERFLOW"),
    ("error_passive", "ERROR-PASSIVE"),
    ("ack_error", "NO ACK"),
    ("bus_off", "BUS-OFF"),
)


# Holding B this long while not recording quits the whole collection. A single
# press used to: B is also "fail" during an episode, so one press too many right
# after an episode ended stopped the dataflow.
QUIT_HOLD_S = 2.0


@dataclasses.dataclass
class HeadsetButtons:
    """Turn the headset's A/B button levels into UI actions.

    The leader sends each button's level with every frame. A and B act on a
    rising edge, except B outside an episode, which quits only once it has been
    held for QUIT_HOLD_S. A press that started an action does nothing else until
    released, so a B that ends an episode (fail) cannot run on into a quit.
    """

    quit_hold_s: float = QUIT_HOLD_S
    _down: dict = dataclasses.field(default_factory=dict)
    _used: set = dataclasses.field(default_factory=set)

    def on_button(self, button, pressed, collecting, now):
        """Return "start" / "success" / "fail" / "quit", or None."""
        was_down = button in self._down
        if not pressed:
            self._down.pop(button, None)
            self._used.discard(button)
            return None
        if not was_down:
            self._down[button] = now
        if button in self._used:
            return None
        if button == "button_a":
            if was_down:
                return None
            self._used.add(button)
            return "success" if collecting else "start"
        if button == "button_b":
            if collecting:
                if was_down:
                    return None
                self._used.add(button)
                return "fail"
            if now - self._down[button] >= self.quit_hold_s:
                self._used.add(button)
                return "quit"
        return None


def _event_ts_to_seconds(ts) -> float:
    """Normalize a dora event timestamp (datetime or ns int) to POSIX seconds."""
    if isinstance(ts, datetime.datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        return ts.timestamp()
    if isinstance(ts, (int, float)):
        return float(ts) / 1e9
    return time.time()


def _fold_arrival(series, stats, ts_s: float, stale_after_s: float) -> None:
    """Fold one arrival time into a rolling rate / jitter estimate.

    A gap longer than `stale_after_s` restarts the window, so the rate
    reported after a stall describes the stream now rather than averaging
    across the outage.
    """
    if series and ts_s - series[-1] > stale_after_s:
        series.clear()
    series.append(ts_s)
    if len(series) < 2:
        return
    span = series[-1] - series[0]
    if span <= 0:
        return
    diffs = [series[i] - series[i - 1] for i in range(1, len(series))]
    stats.fps = (len(series) - 1) / span
    stats.jitter_ms = (max(diffs) - min(diffs)) * 1e3


def _update_camera_stats(event_id: str, ts_s: float) -> None:
    _fold_arrival(
        camera_timestamps[event_id],
        camera_stats[event_id],
        ts_s,
        CAMERA_STALE_AFTER_S,
    )


def _update_arm_health(side: str, value) -> None:
    """Record one dora-openarm `state` message.

    The payload is a length-1 StructArray. `tmos`, `trotor` and
    `motor_status` are one entry per motor, in the order the driver reports
    them (joints, then gripper); `bus` covers the interface as a whole.
    """
    health = arm_health[side]
    now = time.time()
    if now - health.updated_at < ARM_STATE_PARSE_INTERVAL_S:
        return
    names = value.type.names if pa.types.is_struct(value.type) else ()
    try:
        if "tmos" in names:
            health.mos = [int(v) for v in value.field("tmos")[0].as_py()]
            health.rotor = [int(v) for v in value.field("trotor")[0].as_py()]
        if "motor_status" in names:
            health.motor_status = list(value.field("motor_status")[0].as_py())
        if "bus" in names:
            health.bus = dict(value.field("bus")[0].as_py() or {})
            if not health.bus_baseline:
                health.bus_baseline = dict(health.bus)
        health.updated_at = now
    except Exception:
        # An older (or differently shaped) dora-openarm state must never take
        # the whole node down: _main_dora's event loop has no guard around
        # its dispatch, so an exception here kills every panel, not just this
        # one. Broad on purpose -- some pyarrow errors (ArrowIndexError,
        # ArrowNotImplementedError) fall outside AttributeError/KeyError/
        # TypeError/ValueError, so a narrower catch would still let a
        # malformed message escape.
        pass


def _reset_bus_baselines() -> None:
    """Make the bus badges read "during this episode" from here on.

    The counters latch by design, so that a bus-off the driver recovers from
    within milliseconds is not missed. Without a baseline the badge would go
    red once and stay red for the rest of the session.
    """
    for health in arm_health.values():
        health.bus_baseline = dict(health.bus)


def _bus_state(health: ArmHealth) -> dict:
    """Worst bus condition for one arm, and how often it happened.

    `carrier` is the only instantaneous reading. Everything else is counted,
    so a fault that has since recovered reports as RECOVERED rather than as
    though it were still happening.
    """
    if not health.bus:
        return {"state": "UNKNOWN", "count": 0, "severity": "unknown"}
    if not health.bus.get("carrier", True):
        # IFF_UP is clear only when the interface was taken down; a bus-off
        # with no auto-restart leaves it up but without carrier.
        down = health.bus.get("net_down", 0) - health.bus_baseline.get("net_down", 0)
        return {
            "state": "IF DOWN" if down > 0 else "NO CARRIER",
            "count": 0,
            "severity": "error",
        }

    worst, count = None, 0
    for name, label in BUS_COUNTER_SEVERITY:
        delta = health.bus.get(name, 0) - health.bus_baseline.get(name, 0)
        if delta > 0:
            worst, count = label, delta
    if worst is None:
        return {"state": "OK", "count": 0, "severity": "ok"}
    # It happened, but the link is carrying traffic again.
    return {"state": f"RECOVERED ({worst})", "count": count, "severity": "warn"}


def _update_ker_device(value) -> None:
    """Remember what the leader said about itself.

    Sent once when the leader starts, so a UI that connected later simply
    never sees it. That only costs the device name; the panel itself is
    earned by the position stream.
    """
    try:
        payload = value[0].as_py()
        ker_device.update(json.loads(payload) if isinstance(payload, str) else payload)
    except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        pass


def _update_ker_stats(ts_s: float) -> None:
    """Fold one leader sample arrival time into the rolling stats."""
    _fold_arrival(ker_timestamps, ker_stats, ts_s, KER_STALE_AFTER_S)


def _update_vr_stats(ts_s: float) -> None:
    """Fold one real VR packet arrival time (POSIX seconds) into the rolling stats."""
    series = vr_timestamps
    if series and ts_s - series[-1] > VR_STALE_AFTER_S:
        series.clear()
    series.append(ts_s)
    if len(series) < 2:
        return
    span = series[-1] - series[0]
    if span <= 0:
        return
    vr_stats.fps = (len(series) - 1) / span
    diffs = [series[i] - series[i - 1] for i in range(1, len(series))]
    vr_stats.jitter_ms = (max(diffs) - min(diffs)) * 1e3


async def _notify_state_changed() -> None:
    global state_version
    async with _state_changed:
        state_version += 1
        _state_changed.notify_all()


def next_task():
    """Update the state with the next task."""
    state.task_index += 1
    if state.task_index >= len(tasks):
        state.task_index = 0
    state.task_title = tasks[state.task_index]["prompt"]


def _command_start():
    """Start a new episode."""
    node.send_output(
        "command",
        pa.array(["start"]),
        {
            "episode_number": state.episode_number,
            "task_index": state.task_index,
        },
    )
    # From here the bus badges report this episode, not the whole session.
    _reset_bus_baselines()
    state.collecting = True


def _command_success():
    """Finish the current episode successfully."""
    node.send_output("command", pa.array(["success"]))
    state.collecting = False
    state.episode_number += 1
    next_task()


def _command_fail():
    """Finish the current episode unsuccessfully."""
    node.send_output("command", pa.array(["fail"]))
    state.collecting = False
    state.episode_number += 1
    next_task()


def _command_quit():
    """Quit this data collection."""
    node.send_output("command", pa.array(["quit"]))
    state.running = False


def _command_arm_start():
    """Start (power on) the arm(s)."""
    node.send_output("arm_command", pa.array(["start"]))


def _command_arm_stop():
    """Pause (stop) the arm(s)."""
    node.send_output("arm_command", pa.array(["stop"]))


def _command_proceed():
    """Move on an arm's intro that waits for Enter (the WebXR leader's clutch)."""
    node.send_output("command", pa.array(["proceed"]))


@app.get("/", response_class=HTMLResponse)
def _root(request: Request):
    """Render the main HTML."""
    return templates.TemplateResponse(
        request=request,
        name="root.html",
        context={"state": state, "state_version": state_version},
    )


@app.post("/start")
def _start(request: Request):
    _command_start()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/skip")
def _skip(request: Request):
    """Skip the next task."""
    next_task()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/success")
def _success(request: Request):
    _command_success()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/fail")
def _fail(request: Request):
    _command_fail()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/cancel")
def _cancel(request: Request):
    """Cancel the current episode."""
    node.send_output("command", pa.array(["cancel"]))
    state.collecting = False
    state.episode_number += 1
    return RedirectResponse(request.url_for("_root"), 303)


@app.get("/events", response_class=EventSourceResponse)
async def _events(request: Request) -> AsyncIterable[ServerSentEvent]:
    try:
        last_version = int(request.query_params.get("since"))
    except (TypeError, ValueError):
        last_version = state_version
    while state.running:
        async with _state_changed:
            await _state_changed.wait_for(
                lambda: state_version != last_version or not state.running
            )
        if not state.running:
            break
        last_version = state_version
        yield ServerSentEvent(
            data={
                "collecting": state.collecting,
                "episode_number": state.episode_number,
                "task_index": state.task_index,
                "arm_status_right": state.arm_status_right,
                "arm_status_left": state.arm_status_left,
            },
            id=str(state_version),
        )


@app.get("/stats", response_class=EventSourceResponse)
async def _stats() -> AsyncIterable[ServerSentEvent]:
    """Push camera FPS / jitter snapshots to the browser every 500 ms."""
    while state.running:
        now = time.time()
        snapshot = {}
        for name, s in camera_stats.items():
            if name not in seen_inputs:
                # Not wired in this dataflow; the row stays hidden rather
                # than sitting at "-- Hz" for the whole session.
                continue
            series = camera_timestamps[name]
            if not series or now - series[-1] > CAMERA_STALE_AFTER_S:
                snapshot[name] = {"fps": 0.0, "jitter_ms": 0.0}
            else:
                snapshot[name] = {"fps": s.fps, "jitter_ms": s.jitter_ms}
        yield ServerSentEvent(data=snapshot)
        await asyncio.sleep(0.5)


@app.get("/preview/{name}")
async def _preview(name: str):
    """Stream the latest frames of a preview input as MJPEG."""
    if name not in PREVIEW_INPUTS:
        return HTMLResponse(status_code=404)

    async def frames():
        sent_version = -1
        while state.running:
            version = preview_versions[name]
            frame = previews.get(name)
            if frame is not None and version != sent_version:
                sent_version = version
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    + frame
                    + b"\r\n"
                )
            await asyncio.sleep(PREVIEW_INTERVAL_S)

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/vr-stats", response_class=EventSourceResponse)
async def _vr_stats() -> AsyncIterable[ServerSentEvent]:
    """Push the real VR stream Hz / jitter snapshot to the browser every 500 ms."""
    while state.running:
        now = time.time()
        if not vr_timestamps or now - vr_timestamps[-1] > VR_STALE_AFTER_S:
            snapshot = {"fps": 0.0, "jitter_ms": 0.0}
        else:
            snapshot = {"fps": vr_stats.fps, "jitter_ms": vr_stats.jitter_ms}
        snapshot["present"] = any(i in seen_inputs for i in VR_RECEIVE_TIMES_INPUTS)
        yield ServerSentEvent(data=snapshot)
        await asyncio.sleep(0.5)


@app.get("/ker-stats", response_class=EventSourceResponse)
async def _ker_stats() -> AsyncIterable[ServerSentEvent]:
    """Push the leader stream rate and the device it came from every 500 ms."""
    while state.running:
        now = time.time()
        if not ker_timestamps or now - ker_timestamps[-1] > KER_STALE_AFTER_S:
            snapshot = {"fps": 0.0, "jitter_ms": 0.0}
        else:
            snapshot = {"fps": ker_stats.fps, "jitter_ms": ker_stats.jitter_ms}
        snapshot["present"] = KER_POSITION_INPUT in seen_inputs
        snapshot["device"] = ker_device
        yield ServerSentEvent(data=snapshot)
        await asyncio.sleep(0.5)


@app.get("/arm-health", response_class=EventSourceResponse)
async def _arm_health() -> AsyncIterable[ServerSentEvent]:
    """Push per-motor temperatures, motor status and bus state every 500 ms.

    Sampled rather than pushed on change: the arms report at 250 Hz, and
    none of this needs to reach an operator sooner than this.
    """
    while state.running:
        now = time.time()
        snapshot = {}
        for side, health in arm_health.items():
            fresh = health.updated_at and now - health.updated_at <= (
                ARM_TEMPERATURE_STALE_AFTER_S
            )
            snapshot[side] = {
                "present": f"arm_state_{side}" in seen_inputs,
                "mos": health.mos if fresh else [],
                "rotor": health.rotor if fresh else [],
                "motor_status": health.motor_status if fresh else [],
                "bus": _bus_state(health)
                if fresh
                else {"state": "UNKNOWN", "count": 0, "severity": "unknown"},
            }
        yield ServerSentEvent(data=snapshot)
        await asyncio.sleep(0.5)


@app.post("/quit")
def _quit(request: Request):
    _command_quit()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/arm/start")
def _arm_start(request: Request):
    """Start (power on) the arm(s)."""
    _command_arm_start()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/arm/stop")
def _arm_stop(request: Request):
    """Pause (stop) the arm(s)."""
    _command_arm_stop()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/proceed")
def _proceed(request: Request):
    """Enter: move on an arm's intro that waits for it."""
    _command_proceed()
    return RedirectResponse(request.url_for("_root"), 303)


def _merge(base, override):
    """Merge ``override`` into ``base``: dicts recursively, others replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path, robot=None):
    """Load a metadata YAML file.

    A file with ``robots:`` holds every robot: the common part with
    ``robots[robot]`` merged over it. A file without it is used as is.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    robots = data.pop("robots", None)
    if robots is None:
        return data
    if robot not in robots:
        raise SystemExit(
            f"{path}: no robots.{robot}; choose one of: {', '.join(robots)}"
        )
    return _merge(data, robots[robot] or {})


async def _main_uvicorn(server):
    await server.serve()


async def _main_dora(server):
    """Quit the Web application when this dataflow is stopped."""
    # Bring the arm(s) up on boot. dora-openarm no longer auto-starts
    _command_arm_start()
    buttons = HeadsetButtons()
    while state.running:
        if node.is_empty():
            await asyncio.sleep(0.001)
            continue
        event = node.next()
        if event["type"] == "STOP":
            state.running = False
        elif event["type"] == "INPUT":
            event_id = event["id"]
            seen_inputs.add(event_id)
            if event_id == KER_METADATA_INPUT:
                _update_ker_device(event["value"])
                continue
            if event_id == KER_POSITION_INPUT:
                _update_ker_stats(
                    _event_ts_to_seconds(event["metadata"].get("timestamp"))
                )
                continue
            if event_id in CAMERA_INPUTS:
                _update_camera_stats(
                    event_id,
                    _event_ts_to_seconds(event["metadata"].get("timestamp")),
                )
                continue
            if event_id in PREVIEW_INPUTS:
                # Copy out of dora's shared memory; a JPEG is small.
                previews[event_id] = event["value"].to_numpy().tobytes()
                preview_versions[event_id] += 1
                continue
            if event_id in ARM_STATUS_INPUTS:
                value = event["value"][0].as_py()
                # Only notify on an actual change. The follower may publish repeated (heartbeat) status values;
                if getattr(state, event_id) != value:
                    setattr(state, event_id, value)
                    await _notify_state_changed()
                continue
            if event_id in ARM_STATE_INPUTS:
                # Store only. Notifying here would push an SSE frame 250
                # times a second per arm; /arm-health samples instead.
                _update_arm_health(ARM_STATE_INPUTS[event_id], event["value"])
                continue
            if event_id in VR_RECEIVE_TIMES_INPUTS:
                for ts_ns in event["value"].to_pylist():
                    _update_vr_stats(float(ts_ns) / 1e9)
                continue
            if event_id not in ("button_a", "button_b"):
                continue

            action = buttons.on_button(
                event_id,
                bool(event["value"][0].as_py()),
                state.collecting,
                time.monotonic(),
            )
            if action is None:
                continue
            {
                "start": _command_start,
                "success": _command_success,
                "fail": _command_fail,
                "quit": _command_quit,
            }[action]()

            await _notify_state_changed()
    server.should_exit = True


async def _main_async():
    config = uvicorn.Config(app, port=port, log_level="info")
    server = uvicorn.Server(config)

    task_uvicorn = asyncio.create_task(_main_uvicorn(server))
    task_dora = asyncio.create_task(_main_dora(server))

    await task_uvicorn
    # Process may linger when dora exits via SIGTERM,
    # as _main_dora() may not receive a STOP event.
    # Set `state.running = False` when task_uvicorn exits
    # so that _main_dora() also exits.
    state.running = False
    await task_dora


def _select_task(tasks, selector):
    """Return the index of the task that ``selector`` names.

    ``selector`` is empty (the first task), an index, or a prompt: an exact
    match first, then a unique substring, both case-insensitive. Anything
    else exits with the list of tasks to choose from.
    """
    selector = (selector or "").strip()
    if not selector:
        return 0
    if selector.isdigit() and int(selector) < len(tasks):
        return int(selector)
    wanted = selector.casefold()
    prompts = [str(task["prompt"]).strip().casefold() for task in tasks]
    if wanted in prompts:
        return prompts.index(wanted)
    partial = [i for i, prompt in enumerate(prompts) if wanted in prompt]
    if len(partial) == 1:
        return partial[0]
    choices = "\n".join(f"  {i}: {task['prompt']}" for i, task in enumerate(tasks))
    reason = "matches several tasks" if partial else "matches no task"
    raise SystemExit(f"TASK={selector!r} {reason}. Choose one of:\n{choices}")


def main():
    """Run data collection control Web application."""
    global node
    global tasks

    parser = argparse.ArgumentParser(description="Record data as OpenArm dataset")
    parser.add_argument(
        "--metadata-file",
        default=os.getenv("METADATA_FILE"),
        help="The metadata file",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--auto-open",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("AUTO_OPEN", "") == "yes",
        help="Open a Web browser automatically",
    )
    default_port = 8000
    parser.add_argument(
        "--port",
        default=int(os.getenv("PORT", default_port)),
        help=f"The port for UI ({default_port})",
        type=int,
    )
    parser.add_argument(
        "--task",
        default=os.getenv("TASK", ""),
        help=(
            "The task to start with: its index in the metadata's tasks, or "
            "its prompt (exact, or a unique part of it, case-insensitive). "
            "Default: the first task"
        ),
    )
    args = parser.parse_args()
    global auto_open
    auto_open = args.auto_open
    global port
    port = args.port
    metadata = load_yaml(args.metadata_file, os.getenv("ROBOT"))
    tasks = metadata["tasks"]
    state.task_index = _select_task(tasks, args.task)
    state.task_title = tasks[state.task_index]["prompt"]
    print(f"Task {state.task_index}: {state.task_title}", flush=True)

    node = dora.Node()
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
