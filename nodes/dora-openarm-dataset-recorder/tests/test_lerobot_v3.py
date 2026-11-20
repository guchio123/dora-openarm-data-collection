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

"""Raw episodes appended to a LeRobot v3.0 dataset (lerobot_v3)."""

import json

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import dora_openarm_dataset_recorder.main as recorder
from dora_openarm_dataset_recorder.lerobot_v3 import DatasetMismatch, LeRobotV3Dataset

# The real one: conftest stands a raw-keeping one in for it in the node tests.
EpisodeConverter = recorder.EpisodeConverter

T0 = 1_800_000_000_000_000_000
SECOND = 1_000_000_000
TASKS = ["Pick a spoon.", "Pick a cup."]


def _write_state(path, timestamps, **fields):
    path.mkdir(parents=True, exist_ok=True)
    columns = {"timestamp": pa.array(timestamps, type=pa.timestamp("ns"))}
    for name, values in fields.items():
        columns[name] = pa.array([list(v) for v in values], type=pa.list_(pa.float32()))
    pq.write_table(pa.table(columns), path / "state.parquet")


def make_raw(directory, seconds=1.0, offset=0.0, depth=True, cameras=("head_left",)):
    """A raw episode: arms at 100 Hz, JPEG cameras and a 16-bit depth PNG at 20 Hz."""
    arm_times = T0 + (np.arange(int(seconds * 100) + 1) * SECOND // 100)
    for kind in ("obs", "action"):
        for side_index, side in enumerate(("right", "left")):
            qpos = [
                np.full(8, side_index * 10 + (t - T0) / SECOND + offset)
                for t in arm_times
            ]
            fields = {"qpos": qpos}
            if kind == "obs":
                fields["qvel"] = [q * 0 + 1 for q in qpos]
                fields["qtorque"] = [q * 0 + 2 for q in qpos]
            _write_state(directory / kind / "arms" / side, arm_times, **fields)
    camera_times = T0 + 5_000_000 + (np.arange(int(seconds * 20) + 1) * SECOND // 20)
    for name in cameras:
        folder = directory / "cameras" / name
        folder.mkdir(parents=True)
        for i, t in enumerate(camera_times):
            image = np.full((48, 64, 3), (i * 10) % 256, dtype=np.uint8)
            cv2.imwrite(str(folder / f"{t}.jpeg"), image)
    if depth:
        folder = directory / "cameras" / "head_depth"
        folder.mkdir(parents=True)
        for i, t in enumerate(camera_times):
            cv2.imwrite(
                str(folder / f"{t}.png"), np.full((24, 32), 500 + i, dtype=np.uint16)
            )
    return directory


def test_an_episode_becomes_v3_rows_videos_and_metadata(tmp_path):
    raw = make_raw(tmp_path / "raw")
    dataset = LeRobotV3Dataset(tmp_path / "ds", 30, TASKS, {"location": "Lab"})
    assert dataset.add_episode(raw, task_index=1, success=True) == 0

    root = tmp_path / "ds"
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["fps"] == 30
    # Every stream has data from the first camera frame (5 ms in) to the last
    # arm sample (1 s): frames at 0, 1/30, ... within that span.
    assert info["total_episodes"] == 1 and info["total_frames"] == 30
    features = info["features"]
    assert features["observation.state"]["shape"] == [16]
    assert features["observation.state"]["names"][0] == "right_joint1.pos"
    assert features["observation.state"]["names"][8] == "left_joint1.pos"
    assert features["observation.images.head_left"]["dtype"] == "video"
    assert features["observation.images.head_depth"] == {
        "dtype": "image",
        "shape": [24, 32, 1],
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": True, "depth_unit": "mm"},
    }

    data = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
    assert data.column_names == [
        "observation.state",
        "observation.velocity",
        "observation.effort",
        "action",
        "observation.images.head_depth",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    assert data.schema.field("observation.state").type == pa.list_(pa.float32(), 16)
    assert data.schema.field("timestamp").type == pa.float32()
    np.testing.assert_allclose(
        data.column("timestamp").to_numpy(), np.arange(30) / 30, atol=1e-6
    )
    assert data.column("task_index").to_pylist() == [1] * 30
    # The latest sample at or before each frame: frame k is at 5 ms + k/30 s, and
    # the right arm's qpos is its time in seconds, sampled every 10 ms.
    state = np.array(data.column("observation.state").to_pylist())
    expected = np.floor((0.005 + np.arange(30) / 30) * 100) / 100
    np.testing.assert_allclose(state[:, 0], expected, atol=1e-5)
    np.testing.assert_allclose(state[:, 8], expected + 10, atol=1e-5)
    # Depth survives exactly, as a 16-bit TIFF.
    first = data.column("observation.images.head_depth")[0].as_py()
    depth = cv2.imdecode(np.frombuffer(first["bytes"], np.uint8), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16 and depth.shape == (24, 32) and depth[0, 0] == 500
    assert first["path"] == "frame-000000.tiff"

    video = (
        root / "videos" / "observation.images.head_left" / "chunk-000" / "file-000.mp4"
    )
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        times = [float(f.pts * f.time_base) for f in container.decode(stream)]
    np.testing.assert_allclose(times, np.arange(30) / 30, atol=1e-6)

    episodes = pq.read_table(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    row = episodes.to_pylist()[0]
    assert row["tasks"] == ["Pick a cup."] and row["length"] == 30
    assert (row["dataset_from_index"], row["dataset_to_index"]) == (0, 30)
    assert row["videos/observation.images.head_left/to_timestamp"] == pytest.approx(1.0)
    assert row["stats/observation.state/count"] == [30]
    assert np.array(row["stats/observation.images.head_left/mean"]).shape == (3, 1, 1)

    tasks = pq.read_table(root / "meta" / "tasks.parquet")
    assert tasks.column("task").to_pylist() == TASKS
    pandas = json.loads(tasks.schema.metadata[b"pandas"])
    assert pandas["index_columns"] == ["task"]

    stats = json.loads((root / "meta" / "stats.json").read_text())
    assert stats["observation.state"]["count"] == [30]
    assert np.array(stats["observation.images.head_depth"]["max"]).shape == (1, 1, 1)


def test_episodes_append_and_a_reopened_dataset_carries_on(tmp_path):
    first = LeRobotV3Dataset(tmp_path / "ds", 30, TASKS)
    first.add_episode(make_raw(tmp_path / "a"), 0, True)
    again = LeRobotV3Dataset(tmp_path / "ds", 30, TASKS)
    assert again.total_episodes == 1
    assert (
        again.add_episode(make_raw(tmp_path / "b", seconds=2.0, offset=5), 1, False)
        == 1
    )

    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2 and info["total_frames"] == 30 + 60
    second = pq.read_table(tmp_path / "ds" / "data" / "chunk-000" / "file-001.parquet")
    assert second.column("index").to_pylist() == list(range(30, 90))
    assert second.column("episode_index").to_pylist() == [1] * 60
    stats = json.loads((tmp_path / "ds" / "meta" / "stats.json").read_text())
    assert stats["observation.state"]["count"] == [90]
    # The last frame of the second: 5 ms + 59/30 s, sampled every 10 ms, plus 5.
    assert stats["observation.state"]["max"][0] == pytest.approx(6.97, abs=1e-5)
    openarm = yaml.safe_load((tmp_path / "ds" / "meta" / "openarm.yaml").read_text())
    assert [e["success"] for e in openarm["episodes"]] == [True, False]


def test_an_episode_with_other_cameras_or_settings_is_refused(tmp_path):
    dataset = LeRobotV3Dataset(tmp_path / "ds", 30, TASKS, {"location": "Lab"})
    dataset.add_episode(make_raw(tmp_path / "a"), 0, True)
    with pytest.raises(DatasetMismatch, match="features differ"):
        dataset.add_episode(make_raw(tmp_path / "b", depth=False), 0, True)
    with pytest.raises(DatasetMismatch, match="fps"):
        LeRobotV3Dataset(tmp_path / "ds", 50, TASKS)
    with pytest.raises(DatasetMismatch, match="location"):
        LeRobotV3Dataset(tmp_path / "ds", 30, TASKS, {"location": "Home"})


def test_the_converter_writes_finished_episodes_and_keeps_failed_ones(tmp_path):
    dataset = LeRobotV3Dataset(tmp_path / "ds", 30, TASKS)
    staging = tmp_path / "ds" / ".staging"
    good = make_raw(staging / "episodes" / "0")
    (good / recorder.EPISODE_INFO_FILE).write_text(
        json.dumps({"number": 0, "success": True, "task_index": 0})
    )
    bad = staging / "episodes" / "1"
    (bad / "cameras").mkdir(parents=True)
    (bad / recorder.EPISODE_INFO_FILE).write_text(
        json.dumps({"number": 1, "success": False, "task_index": 0})
    )
    converter = EpisodeConverter(dataset, staging)
    converter.submit(good)
    converter.submit(bad)
    converter.close()
    assert dataset.total_episodes == 1
    assert not good.exists() and not bad.exists()
    assert (staging / "failed" / "1" / recorder.EPISODE_INFO_FILE).exists()
