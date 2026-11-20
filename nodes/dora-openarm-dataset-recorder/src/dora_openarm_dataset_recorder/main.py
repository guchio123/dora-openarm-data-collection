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

"""Node to record data from OpenArm and cameras as a LeRobot dataset (v3.0).

While an episode is recorded its data goes to a raw episode directory under
``<dataset>/.staging/episodes/<n>``: arm states kept in memory and written as
parquet when it ends, camera frames written as they arrive, one file each.
When the episode ends with success or fail, a background thread appends it to
the LeRobot dataset (see lerobot_v3) and deletes the raw directory; cancel
deletes it at once. Episodes still in the staging area at startup (the node
stopped before converting them) are converted first.
"""

import argparse
import datetime
import json
import math
import os
import pathlib
import queue
import shutil
import threading
import traceback
from dataclasses import dataclass, field

import dora
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from numpy.typing import ArrayLike

from dora_openarm_dataset_recorder.lerobot_v3 import DatasetMismatch, LeRobotV3Dataset

# The raw episode's own record of how it ended (task, success), written last:
# a staging directory without it was never finished.
EPISODE_INFO_FILE = "episode.json"


@dataclass
class Episode:
    """Episode related data."""

    number: int = 0
    success: bool = False
    task_index: int = 0

    right_action_timestamps: ArrayLike = field(default_factory=list)
    right_actions: ArrayLike = field(default_factory=list)
    right_observation_timestamps: ArrayLike = field(default_factory=list)
    right_observations: ArrayLike = field(default_factory=list)
    left_action_timestamps: ArrayLike = field(default_factory=list)
    left_actions: ArrayLike = field(default_factory=list)
    left_observation_timestamps: ArrayLike = field(default_factory=list)
    left_observations: ArrayLike = field(default_factory=list)
    elevation_action_timestamps: ArrayLike = field(default_factory=list)
    elevation_actions: ArrayLike = field(default_factory=list)
    elevation_observation_timestamps: ArrayLike = field(default_factory=list)
    elevation_observations: ArrayLike = field(default_factory=list)


def extract_values(value: pa.Array, key: str) -> np.ndarray:
    """Read `key` from a length-1 StructArray, or a flat array as-is."""
    if pa.types.is_struct(value.type):
        value = value.field(key)[0].values
    return np.array(value, dtype=np.float32)


class EpisodeWriter:
    """Writer an episode."""

    def __init__(self, directory, episode):
        """Initialize variables."""
        self._directory = directory
        self._episode = episode
        self._base_directory = self._directory / "episodes" / str(self._episode.number)

    @property
    def directory(self):
        """The raw episode directory."""
        return self._base_directory

    def write_camera_image(self, name, image, timestamp, format):
        """Write an image from a camera."""
        output_path = self._base_directory / "cameras" / name / f"{timestamp}.{format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as output:
            # Using pa.PythonFile here is for zero-copy. We can't
            # write pa.Buffer data to Python's IO directory. If we use
            # Python's IO, we need to copy data in pa.Buffer as
            # Python's bytes. We want to avoid it.
            with pa.PythonFile(output) as pa_output:
                pa_output.write(image.buffers()[1])

    def finish(self):
        """Write all pending data, then the episode's info file."""
        self._write_states()
        self._base_directory.mkdir(parents=True, exist_ok=True)
        info = {
            "number": self._episode.number,
            "success": self._episode.success,
            "task_index": self._episode.task_index,
        }
        (self._base_directory / EPISODE_INFO_FILE).write_text(json.dumps(info))

    def _write_states(self):
        if self._episode.right_actions:
            self._write_kinematic_state(
                self._base_directory / "action" / "arms" / "right",
                self._episode.right_action_timestamps,
                self._episode.right_actions,
            )
        if self._episode.right_observations:
            self._write_kinematic_state(
                self._base_directory / "obs" / "arms" / "right",
                self._episode.right_observation_timestamps,
                self._episode.right_observations,
            )
        if self._episode.left_actions:
            self._write_kinematic_state(
                self._base_directory / "action" / "arms" / "left",
                self._episode.left_action_timestamps,
                self._episode.left_actions,
            )
        if self._episode.left_observations:
            self._write_kinematic_state(
                self._base_directory / "obs" / "arms" / "left",
                self._episode.left_observation_timestamps,
                self._episode.left_observations,
            )
        if self._episode.elevation_actions:
            self._write_positions(
                self._base_directory / "action" / "lifter" / "elevation.parquet",
                self._episode.elevation_action_timestamps,
                self._episode.elevation_actions,
            )
        if self._episode.elevation_observations:
            self._write_positions(
                self._base_directory / "obs" / "lifter" / "elevation.parquet",
                self._episode.elevation_observation_timestamps,
                self._episode.elevation_observations,
            )

    def cancel(self):
        """Cancel this episode."""
        shutil.rmtree(self._base_directory, ignore_errors=True)

    def _write_positions(self, output_path, timestamps, positions):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        list_type = pa.list_(pa.float32())
        table = pa.table(
            {
                "timestamp": pa.array(timestamps, type=pa.timestamp("ns")),
                "value": pa.array(positions, type=list_type),
            }
        )
        pq.write_table(table, output_path)

    def _write_kinematic_state(self, base_path, timestamps, states):
        output_path = base_path / "state.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        list_type = pa.list_(pa.float32())
        first = states[0]

        if isinstance(first, pa.StructArray):
            field_names = first.type.names
            available_field_names = [
                "qpos",
                "qvel",
                "qtorque",
                "pose",
            ]  # currently only support these fields
            state_fields = {"timestamp": pa.array(timestamps, type=pa.timestamp("ns"))}
            for field_name in field_names:
                if field_name not in available_field_names:
                    continue
                state_fields[field_name] = pa.array(
                    [extract_values(s, field_name) for s in states],
                    type=list_type,
                )
            table = pa.table(state_fields)
        else:
            # if the observation is not a struct, it should be qpos.
            table = pa.table(
                {
                    "timestamp": pa.array(timestamps, type=pa.timestamp("ns")),
                    "qpos": pa.array(states, type=list_type),
                }
            )
        pq.write_table(table, output_path)


class EpisodeConverter:
    """Appends finished raw episodes to the LeRobot dataset on a background thread.

    One at a time, so that the next episode can start meanwhile.
    """

    def __init__(self, dataset, staging):
        """Start the thread."""
        self._dataset = dataset
        self._staging = staging
        self._queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="lerobot-writer", daemon=True
        )
        self._thread.start()

    def submit(self, raw_directory):
        """Convert this raw episode directory after the ones before it."""
        self._queue.put(raw_directory)

    def call(self, function):
        """Run ``function(dataset)`` on the thread, between episodes."""
        self._queue.put(function)

    def close(self):
        """Convert what is queued, then stop."""
        pending = self._queue.qsize()
        if pending:
            print(
                f"Writing {pending} pending item(s) to the LeRobot dataset", flush=True
            )
        self._queue.put(None)
        self._thread.join()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            if callable(item):
                item(self._dataset)
                continue
            self._convert(item)

    def _convert(self, raw):
        started = datetime.datetime.now()
        try:
            info = json.loads((raw / EPISODE_INFO_FILE).read_text())
            index = self._dataset.add_episode(raw, info["task_index"], info["success"])
        except Exception:  # noqa: BLE001 - any failure keeps the raw episode
            failed = self._staging / "failed" / raw.name
            failed.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(failed, ignore_errors=True)
            shutil.move(str(raw), str(failed))
            print(
                f"Could not add raw episode {raw.name} to the LeRobot dataset; kept it in "
                f"{failed}:\n{traceback.format_exc()}",
                flush=True,
            )
            return
        shutil.rmtree(raw, ignore_errors=True)
        seconds = (datetime.datetime.now() - started).total_seconds()
        if index is None:
            print(
                f"Raw episode {raw.name} has no frame with every stream; dropped",
                flush=True,
            )
        else:
            print(
                f"Episode {index} written ({self._dataset.episodes[-1]['length']} frames, "
                f"{seconds:.1f} s; {self._dataset.total_episodes} episodes in "
                f"{self._dataset.root})",
                flush=True,
            )


class DatasetWriter:
    """Write a LeRobot dataset through a staging area."""

    def __init__(self, directory, name, metadata, fps=30, converter_class=None):
        """Open (or create) the dataset; refuse one whose metadata differs."""
        self._base_directory = directory / name
        self._staging = self._base_directory / ".staging"
        self._metadata = metadata
        tasks = [
            str(task.get("prompt") if isinstance(task, dict) else task)
            for task in metadata.get("tasks", []) or []
        ]
        try:
            dataset = LeRobotV3Dataset(self._base_directory, fps, tasks, metadata)
        except DatasetMismatch as error:
            raise ValueError(str(error)) from error
        self._converter = (converter_class or EpisodeConverter)(dataset, self._staging)
        self._submit_pending()

    def _submit_pending(self):
        episodes = self._staging / "episodes"
        if not episodes.exists():
            return
        for raw in sorted(episodes.iterdir(), key=lambda p: p.stat().st_mtime):
            if (raw / EPISODE_INFO_FILE).exists():
                print(
                    f"Converting raw episode {raw.name} left from an earlier run",
                    flush=True,
                )
                self._converter.submit(raw)
            else:
                print(
                    f"Raw episode {raw} was never finished (the node stopped while recording); "
                    "left as it is",
                    flush=True,
                )

    def create_episode_writer(self, episode):
        """Create a writer for the given episode, in a fresh staging directory."""
        episodes_directory = self._staging / "episodes"
        used = set()
        for directory in (episodes_directory, self._staging / "failed"):
            if directory.exists():
                used |= {p.name for p in directory.iterdir()}
        if str(episode.number) in used:
            numbers = [int(i) for i in used if i.isdigit()]
            episode.number = max(numbers, default=-1) + 1
        return EpisodeWriter(self._staging, episode)

    def finish_episode(self, episode_writer):
        """Hand a finished episode to the converter."""
        self._converter.submit(episode_writer.directory)

    def set_leader_ker_metadata(self, ker_metadata):
        """Record KER leader device metadata under equipment.leader.ker."""

        def update(dataset):
            equipment = dataset.metadata.setdefault("equipment", {})
            leader = equipment.setdefault("leader", {})
            ker = leader.setdefault("ker", {})
            ker["id"] = "OpenArmKER"
            ker["firmware_version"] = ker_metadata.get("fw")
            ker["hardware_version"] = ker_metadata.get("hw")

        self._converter.call(update)

    def close(self):
        """Wait until every finished episode is in the dataset."""
        self._converter.close()


class FrequencyDetector:
    """Detect frequency of an input."""

    def __init__(self, configs):
        """Initialize with node configurations."""
        self._configs = configs

    def detect(self, input):
        """Detect frequency of the given input."""
        if isinstance(input, dict):
            input = input["source"]
        node_id, name = input.split("/", 1)
        for config in self._configs:
            if config["id"] != node_id:
                continue
            inputs = config["inputs"]
            next_input = (
                inputs.get("tick")
                or inputs.get("request_state")
                or inputs.get("request_position")
            )
            if next_input is None:
                continue
            if next_input.startswith("dora/timer/"):
                unit, value = next_input.split("/")[2:4]
                if unit == "secs":
                    return 1.0 / int(value)
                elif unit == "millis":
                    return 1_000.0 / int(value)
                else:
                    return None
            else:
                return self.detect(next_input)


class EventReader:
    """Read events from a node on a background thread.

    Writing an episode blocks the main loop for a while. If nobody reads
    the node meanwhile, its event channel fills up and dora drops the
    inputs that arrive, including the next `start` command. This keeps
    reading and queues the events in the order they arrived.

    Call `finish()` before exiting. Python crashes with SIGSEGV if it
    shuts down while the thread is still inside the node.
    """

    _END = object()
    _TIMEOUT = 0.1

    def __init__(self, node):
        """Start reading events from the given node."""
        self._events = queue.Queue()
        self._error = None
        self._finishing = threading.Event()
        self._thread = threading.Thread(
            target=self._read, args=(node,), name="event-reader", daemon=True
        )
        self._thread.start()

    def __iter__(self):
        """Yield the read events until the node has no more events."""
        while True:
            event = self._events.get()
            if event is self._END:
                break
            yield event
        if self._error is not None:
            raise self._error

    def finish(self):
        """Stop reading events and wait for the thread to end."""
        self._finishing.set()
        self._thread.join()

    def _read(self, node):
        try:
            while not self._finishing.is_set():
                # Wait with a timeout to check _finishing regularly. This
                # returns an ERROR event on timeout; it's ignored like any
                # other non-INPUT event.
                event = node.next(timeout=self._TIMEOUT)
                if event is None:
                    break
                self._events.put(event)
        except BaseException as error:
            self._error = error
        finally:
            self._events.put(self._END)


def _merge(base, override):
    """Merge ``override`` into ``base``: dicts recursively, others replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_metadata(path, robot=None):
    """Load a metadata file.

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


def _collect_dynamic_metadata(metadata, args, node):
    metadata["operation_type"] = args.operation_type
    if args.operation_type == "teleop":
        if "equipment" not in metadata:
            metadata["equipment"] = {}
        if "embodiments" not in metadata["equipment"]:
            metadata["equipment"]["embodiments"] = {}
        # equipment.leader.ker is filled at runtime from the KER node's
        # metadata reply (see main()'s "ker_metadata" handling).
    elif args.operation_type == "rollout":
        if "model" not in metadata:
            metadata["model"] = {}
        if args.docker_image:
            metadata["model"]["docker_image"] = args.docker_image

    metadata["frequencies"] = {
        "action": {
            "arms": {},
        },
        "obs": {
            "arms": {},
        },
        "cameras": {},
    }
    frequency_detector = FrequencyDetector(node.dataflow_descriptor()["nodes"])
    for name, input in node.node_config()["inputs"].items():
        frequency = frequency_detector.detect(input)
        if not frequency:
            continue
        if name.startswith("arm_"):
            # arm_right_action -> right, action
            side, type = name.split("_")[1:3]
            if type == "observation":
                type = "obs"
            metadata["frequencies"][type]["arms"][side] = frequency
        elif name.startswith("camera_"):
            # camera_wrist_right -> wrist_right
            camera_name = name.removeprefix("camera_")
            metadata["frequencies"]["cameras"][camera_name] = frequency


def main():
    """Collect data and record them."""
    parser = argparse.ArgumentParser(description="Record data as OpenArm dataset")
    parser.add_argument(
        "--directory",
        default=os.getenv("DIRECTORY", os.getcwd()),
        help="The output directory",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--docker-image",
        default=os.getenv("DOCKER_IMAGE", os.getenv("IMAGE")),
        help="The Docker image used for this rollout",
        type=str,
    )
    parser.add_argument(
        "--metadata-file",
        default=os.getenv("METADATA_FILE"),
        help="The metadata file",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--name",
        default=os.getenv("NAME", "lerobot"),
        help="The dataset name: the LeRobot dataset's directory under --directory",
        type=str,
    )
    parser.add_argument(
        "--fps",
        default=int(os.getenv("FPS", "30")),
        help="Frames per second of the LeRobot dataset",
        type=int,
    )
    parser.add_argument(
        "--operation-type",
        choices=["teleop", "rollout"],
        default=os.getenv("OPERATION_TYPE", "teleop"),
        help="The operation type",
        type=str,
    )
    args = parser.parse_args()

    node = dora.Node()
    if args.metadata_file is None:
        metadata = {}
    else:
        metadata = _load_metadata(args.metadata_file, os.getenv("ROBOT"))
    _collect_dynamic_metadata(metadata, args, node)
    dataset_writer = DatasetWriter(args.directory, args.name, metadata, args.fps)
    event_reader = EventReader(node)
    try:
        _record_events(event_reader, dataset_writer)
    finally:
        event_reader.finish()
        dataset_writer.close()


def _record_events(event_reader, dataset_writer):
    """Record events read by event_reader until they end or quit is received."""
    episode = None
    episode_writer = None
    arm_observation_timestamp_key = None

    for event in event_reader:
        if event["type"] != "INPUT":
            continue

        event_id = event["id"]
        if event_id == "command":
            command = event["value"][0].as_py()
            if command == "start":
                episode = Episode()
                episode.number = event["metadata"].get("episode_number", 0)
                episode.task_index = event["metadata"].get("task_index", 0)
                episode_writer = dataset_writer.create_episode_writer(episode)
            elif command in ("success", "fail", "cancel") and episode is None:
                print(f"Ignored {command}: no episode is being recorded", flush=True)
            elif command in ("success", "fail"):
                if command == "success":
                    episode.success = True
                episode_writer.finish()
                dataset_writer.finish_episode(episode_writer)
                episode = None
                episode_writer = None
            elif command == "cancel":
                episode_writer.cancel()
                episode = None
                episode_writer = None
            elif command == "quit":
                if episode is not None:
                    episode_writer.finish()
                    dataset_writer.finish_episode(episode_writer)
                    episode = None
                    episode_writer = None
                break
            continue

        if event_id == "ker_metadata":
            # KER leader device metadata (JSON) from the KER node.
            ker_metadata = json.loads(event["value"][0].as_py())
            dataset_writer.set_leader_ker_metadata(ker_metadata)
            continue

        timestamp_key = "timestamp"
        if event_id in ("arm_right_observation", "arm_left_observation"):
            if arm_observation_timestamp_key is None:
                arm_observation_timestamp_key = (
                    "observation_timestamp"
                    if "observation_timestamp" in event["metadata"]
                    else "timestamp"
                )
                print(
                    f"Arm observation timestamp field: {arm_observation_timestamp_key}",
                    flush=True,
                )
            timestamp_key = arm_observation_timestamp_key
            if timestamp_key not in event["metadata"]:
                raise ValueError(
                    f"{event_id} is missing locked timestamp field {timestamp_key!r}"
                )

        # Main process
        if episode is None:
            continue
        timestamp = event["metadata"][timestamp_key]
        if isinstance(timestamp, datetime.datetime):
            # Added by dora-rs automatically.
            # Convert to POSIX timestamp in nanosecond.
            timestamp = math.ceil(timestamp.timestamp() * 1_000_000_000)
        if event_id.startswith("arm_"):
            value = event["value"]
            if isinstance(value, pa.StructArray) and "new_position" in value.type.names:
                value = value.field("new_position")

            # arm_right_action ->
            # right_action
            key_prefix = event_id.removeprefix("arm_")
            # right_action ->
            # right_actions
            values_key = f"{key_prefix}s"
            getattr(episode, values_key).append(value)
            # right_action ->
            # right_action_timestamps
            timestamps_key = f"{key_prefix}_timestamps"
            getattr(episode, timestamps_key).append(timestamp)
        elif event_id.startswith("elevation_"):
            # elevation_observation -> elevation_observations, elevation_observation_timestamps
            # elevation_action -> elevation_actions, elevation_action_timestamps
            getattr(episode, f"{event_id}s").append(event["value"])
            getattr(episode, f"{event_id}_timestamps").append(timestamp)
        elif event_id.startswith("camera_"):
            name = event_id.removeprefix("camera_")
            image = event["value"]
            format = event["metadata"]["encoding"]
            episode_writer.write_camera_image(name, image, timestamp, format)


if __name__ == "__main__":
    main()
