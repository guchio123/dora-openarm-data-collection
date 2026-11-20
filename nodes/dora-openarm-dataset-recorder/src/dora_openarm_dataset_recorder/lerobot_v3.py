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

"""Append recorded episodes to a LeRobot dataset, format v3.0 (what lerobot 0.7 reads).

No lerobot or torch here: pyarrow writes the parquet files, PyAV the videos and
OpenCV reads the camera frames. The layout follows lerobot 0.7.0's own writer:

    meta/info.json            fps, features, totals, path templates
    meta/stats.json           per-feature stats over every episode
    meta/tasks.parquet        task string (pandas index) -> task_index
    meta/episodes/chunk-000/file-000.parquet   one row per episode
    meta/openarm.yaml         the OpenArm metadata, and success per episode
    data/chunk-CCC/file-FFF.parquet            one file per episode
    videos/<key>/chunk-CCC/file-FFF.mp4        one file per episode and camera

An episode comes in as a raw episode directory (what the recorder writes while
recording, see main.EpisodeWriter): arm states as parquet with Unix-ns
timestamps, camera frames as one file per frame named by timestamp. It is
resampled onto a fixed grid of ``fps`` frames per second: frame k is at
``t0 + k / fps``, where t0 is the first instant every stream has data, and
each stream contributes its latest sample at or before that instant.

- ``observation.state`` / ``observation.velocity`` / ``observation.effort``:
  the measured qpos / qvel / qtorque, right arm then left (8 + 8).
- ``action``: the commanded qpos, right then left.
- JPEG cameras become AV1 videos (libsvtav1, yuv420p, GOP 2, CRF 30, as
  lerobot encodes them), ``observation.images.<camera>``.
- 16-bit PNG cameras (the ZED depth, mm) become ``image`` features stored in
  the data parquet as 16-bit TIFF (LZW), flagged ``is_depth_map`` with
  ``depth_unit: mm``, which is how lerobot 0.7 stores depth as images.
"""

from __future__ import annotations

import copy
import fractions
import json
import os
import pathlib
from dataclasses import dataclass

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# SVT-AV1 prints its configuration for every video it encodes; errors only.
os.environ.setdefault("SVT_LOG", "1")

CODEBASE_VERSION = "v3.0"
CHUNKS_SIZE = 1000
DATA_FILES_SIZE_IN_MB = 100
VIDEO_FILES_SIZE_IN_MB = 200
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-000/file-000.parquet"
OPENARM_METADATA_PATH = "meta/openarm.yaml"

SIDES = ("right", "left")
ARM_JOINTS = [f"joint{j}" for j in range(1, 8)] + ["gripper"]
# Raw state field -> (feature, name suffix).
OBSERVATION_FIELDS = {
    "qpos": ("observation.state", "pos"),
    "qvel": ("observation.velocity", "vel"),
    "qtorque": ("observation.effort", "effort"),
}
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
STAT_NAMES = ("min", "max", "mean", "std", *QUANTILES, "count")
# Video encoding: lerobot 0.7's defaults (configs/video.py).
VIDEO_CODEC = "libsvtav1"
VIDEO_OPTIONS = {"crf": "30", "preset": "12", "svtav1-params": "fast-decode=0"}
VIDEO_GOP = 2
# Image stats: like lerobot, about this many frames, shrunk to at most this side.
STATS_FRAMES = 100
STATS_SIDE = 150
# Data rows written to parquet at a time (depth frames make rows ~150 kB).
ROW_BATCH = 64


class DatasetMismatch(ValueError):
    """An episode or the metadata does not fit the dataset already on disk."""


@dataclass
class _Stream:
    timestamps: np.ndarray  # int64 ns, ascending
    values: list  # per sample: an ndarray (arms) or a file path (cameras)

    def at(self, times: np.ndarray) -> np.ndarray:
        """Index of the latest sample at or before each time (the first one before it)."""
        index = np.searchsorted(self.timestamps, times, side="right") - 1
        return np.clip(index, 0, len(self.timestamps) - 1)


def _read_state(path: pathlib.Path) -> dict[str, _Stream] | None:
    if not path.exists():
        return None
    table = pq.read_table(path)
    order = np.argsort(
        table.column("timestamp").cast(pa.int64()).to_numpy(), kind="stable"
    )
    timestamps = table.column("timestamp").cast(pa.int64()).to_numpy()[order]
    streams = {}
    for name in table.column_names:
        if name == "timestamp":
            continue
        values = np.array(table.column(name).to_pylist(), dtype=np.float32)[order]
        streams[name] = _Stream(timestamps, list(values))
    return streams


def _read_camera(directory: pathlib.Path) -> tuple[_Stream, str] | None:
    files = [p for p in directory.iterdir() if p.is_file() and p.stem.isdigit()]
    if not files:
        return None
    files.sort(key=lambda p: int(p.stem))
    formats = {p.suffix.lstrip(".").lower() for p in files}
    if len(formats) != 1:
        raise DatasetMismatch(f"{directory}: mixed image formats {sorted(formats)}")
    timestamps = np.array([int(p.stem) for p in files], dtype=np.int64)
    return _Stream(timestamps, files), formats.pop()


def _is_depth_format(fmt: str) -> bool:
    return fmt == "png"


def _stats(values: np.ndarray, axis: int | tuple = 0) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    stats = {
        "min": values.min(axis=axis),
        "max": values.max(axis=axis),
        "mean": values.mean(axis=axis),
        "std": values.std(axis=axis),
    }
    for name, q in QUANTILES.items():
        stats[name] = np.quantile(values, q, axis=axis)
    return stats


def _image_stats(frames: list[np.ndarray], scale: float) -> dict[str, np.ndarray]:
    """Per-channel stats of (H, W, C) frames as (C, 1, 1) arrays, divided by ``scale``."""
    pixels = np.concatenate(
        [f.reshape(-1, f.shape[-1]).astype(np.float64) / scale for f in frames]
    )
    return {k: v.reshape(-1, 1, 1) for k, v in _stats(pixels).items()}


def _shrink(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(height, width)
    if side <= 2 * STATS_SIDE:
        return image
    factor = STATS_SIDE / side
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return small.reshape(size[1], size[0], -1)


def _stats_sample(n: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(n, STATS_FRAMES)).round().astype(int))


def _read_rgb(path: pathlib.Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DatasetMismatch(f"cannot read image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_depth(path: pathlib.Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
        raise DatasetMismatch(f"{path}: not a single-channel 16-bit image")
    return depth


def encode_video(
    frames: list[pathlib.Path], path: pathlib.Path, fps: int
) -> tuple[dict, dict[str, np.ndarray]]:
    """Encode ``frames`` (one file per output frame, repeats allowed) at ``fps``.

    Frame k gets pts k in a 1/fps time base, so it decodes at exactly k / fps.
    Returns the video info for info.json and the frames' stats.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    time_base = fractions.Fraction(1, fps)
    sample = set(_stats_sample(len(frames)).tolist())
    sampled = []
    first = _read_rgb(frames[0])
    height, width = first.shape[:2]
    if height % 2 or width % 2:
        raise DatasetMismatch(
            f"{frames[0]}: {width}x{height} is odd, yuv420p needs even sides"
        )
    with av.open(
        str(temporary), "w", format="mp4", options={"movflags": "faststart"}
    ) as out:
        stream = out.add_stream(VIDEO_CODEC, rate=fps, options=dict(VIDEO_OPTIONS))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.time_base = time_base
        stream.codec_context.gop_size = VIDEO_GOP
        last_path, image = None, first
        for k, frame_path in enumerate(frames):
            if frame_path != last_path:
                image = first if k == 0 else _read_rgb(frame_path)
                if image.shape[:2] != (height, width):
                    raise DatasetMismatch(
                        f"{frame_path}: size changed within the episode"
                    )
                last_path = frame_path
            if k in sample:
                sampled.append(_shrink(image))
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = k
            frame.time_base = time_base
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    os.replace(temporary, path)
    info = {
        "video.height": height,
        "video.width": width,
        "video.codec": "av1",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
        "is_depth_map": False,
    }
    return info, _image_stats(sampled, 255.0)


def _depth_tiff(depth: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".tiff", depth)
    if not ok:
        raise DatasetMismatch("cannot encode a depth frame as TIFF")
    return data.tobytes()


def _write_parquet_atomic(table: pa.Table, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    pq.write_table(table, temporary)
    os.replace(temporary, path)


def _write_text_atomic(text: str, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _tasks_table(tasks: list[str]) -> pa.Table:
    """meta/tasks.parquet: task_index with the task string as the pandas index.

    lerobot reads it with pandas.read_parquet and looks a task up by
    ``tasks.iloc[task_index].name``, so the pandas metadata has to make
    ``task`` the index.
    """
    table = pa.table(
        {
            "task_index": pa.array(range(len(tasks)), type=pa.int64()),
            "task": pa.array(tasks, type=pa.string()),
        }
    )
    pandas = {
        "index_columns": ["task"],
        "column_indexes": [
            {
                "name": None,
                "field_name": None,
                "pandas_type": "unicode",
                "numpy_type": "object",
                "metadata": {"encoding": "UTF-8"},
            }
        ],
        "columns": [
            {
                "name": "task_index",
                "field_name": "task_index",
                "pandas_type": "int64",
                "numpy_type": "int64",
                "metadata": None,
            },
            {
                "name": "task",
                "field_name": "task",
                "pandas_type": "unicode",
                "numpy_type": "object",
                "metadata": None,
            },
        ],
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.2.3",
    }
    return table.replace_schema_metadata({b"pandas": json.dumps(pandas).encode()})


def _chunk_file(index: int) -> tuple[int, int]:
    return index // CHUNKS_SIZE, index % CHUNKS_SIZE


def _jsonable(stats: dict[str, np.ndarray]) -> dict:
    return {name: np.asarray(value).tolist() for name, value in stats.items()}


def aggregate_stats(episodes: list[dict[str, dict[str, np.ndarray]]]) -> dict:
    """Dataset stats from per-episode stats.

    Count-weighted mean, pooled std, elementwise min and max, and the
    count-weighted mean of the quantiles.
    """
    if not episodes:
        return {}
    out = {}
    for feature in episodes[0]:
        parts = [e[feature] for e in episodes if feature in e]
        counts = np.array([float(np.asarray(p["count"]).reshape(-1)[0]) for p in parts])
        weights = counts / counts.sum()

        def weighted(name, parts=parts, weights=weights):
            return sum(
                w * np.asarray(p[name], dtype=np.float64)
                for w, p in zip(weights, parts)
            )

        mean = weighted("mean")
        variance = sum(
            w
            * (
                np.asarray(p["std"], dtype=np.float64) ** 2
                + (np.asarray(p["mean"]) - mean) ** 2
            )
            for w, p in zip(weights, parts)
        )
        stats = {
            "min": np.minimum.reduce(
                [np.asarray(p["min"], dtype=np.float64) for p in parts]
            ),
            "max": np.maximum.reduce(
                [np.asarray(p["max"], dtype=np.float64) for p in parts]
            ),
            "mean": mean,
            "std": np.sqrt(variance),
            "count": np.array([int(counts.sum())]),
        }
        for name in QUANTILES:
            stats[name] = weighted(name)
        out[feature] = _jsonable(stats)
    return out


class LeRobotV3Dataset:
    """A LeRobot v3.0 dataset on disk that episodes are appended to, one at a time."""

    def __init__(
        self,
        root: pathlib.Path,
        fps: int,
        tasks: list[str],
        metadata: dict | None = None,
        robot_type: str = "openarm",
    ) -> None:
        """Open the dataset at ``root``, or start one there."""
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.root = pathlib.Path(root)
        self.fps = int(fps)
        self.tasks = list(tasks) or ["default task"]
        self.robot_type = robot_type
        self.metadata = copy.deepcopy(metadata or {})
        self.features: dict | None = None
        self.episodes: list[dict] = []
        self.episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
        self.openarm_episodes: list[dict] = []
        self._load()

    # -- existing dataset -------------------------------------------------

    def _load(self) -> None:
        info_path = self.root / "meta" / "info.json"
        openarm_path = self.root / OPENARM_METADATA_PATH
        if openarm_path.exists():
            existing = yaml.safe_load(openarm_path.read_text(encoding="utf-8")) or {}
            self.openarm_episodes = existing.pop("episodes", [])
            existing.pop("version", None)
            mismatched = {
                k: (existing[k], v)
                for k, v in self.metadata.items()
                if k in existing and existing[k] != v
            }
            if mismatched:
                raise DatasetMismatch(
                    f"Existing dataset metadata does not match: {openarm_path}\n"
                    + "\n".join(
                        f"  {k}: existing={old!r}, current={new!r}"
                        for k, (old, new) in mismatched.items()
                    )
                )
        if not info_path.exists():
            return
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("fps") != self.fps:
            raise DatasetMismatch(f"{info_path}: fps {info.get('fps')}, not {self.fps}")
        self.features = info["features"]
        tasks_path = self.root / "meta" / "tasks.parquet"
        if tasks_path.exists():
            existing_tasks = pq.read_table(tasks_path).column("task").to_pylist()
            if existing_tasks != self.tasks[: len(existing_tasks)]:
                raise DatasetMismatch(
                    f"{tasks_path}: tasks {existing_tasks} are not a prefix of {self.tasks}"
                )
        episodes_path = self.root / EPISODES_PATH
        if not episodes_path.exists():
            return
        table = pq.read_table(episodes_path)
        for row in table.to_pylist():
            stats: dict[str, dict[str, np.ndarray]] = {}
            plain = {}
            for column, value in row.items():
                if column.startswith("stats/"):
                    _, feature_stat = column.split("/", 1)
                    feature, stat = feature_stat.rsplit("/", 1)
                    stats.setdefault(feature, {})[stat] = np.asarray(value)
                else:
                    plain[column] = value
            self.episodes.append(plain)
            self.episode_stats.append(stats)
        indices = [e["episode_index"] for e in self.episodes]
        if indices != list(range(len(indices))):
            raise DatasetMismatch(
                f"{episodes_path}: episode indices are not 0..{len(indices) - 1}"
            )

    @property
    def total_episodes(self) -> int:
        """Episodes in the dataset."""
        return len(self.episodes)

    @property
    def total_frames(self) -> int:
        """Frames in the dataset."""
        return sum(int(e["length"]) for e in self.episodes)

    # -- one episode ------------------------------------------------------

    def add_episode(
        self, raw: pathlib.Path, task_index: int = 0, success: bool | None = None
    ) -> int | None:
        """Append the raw episode directory ``raw``.

        Returns its episode_index, or None when it has no instant with every
        stream present (nothing written).
        """
        raw = pathlib.Path(raw)
        if not 0 <= task_index < len(self.tasks):
            raise DatasetMismatch(
                f"task_index {task_index} is not one of {len(self.tasks)} tasks"
            )
        arms = {
            (kind, side): _read_state(raw / kind / "arms" / side / "state.parquet")
            for kind in ("obs", "action")
            for side in SIDES
        }
        missing = [f"{kind}/{side}" for (kind, side), s in arms.items() if not s]
        if missing:
            raise DatasetMismatch(f"{raw}: no arm data for {', '.join(missing)}")
        cameras: dict[str, tuple[_Stream, str]] = {}
        camera_root = raw / "cameras"
        if camera_root.exists():
            for directory in sorted(p for p in camera_root.iterdir() if p.is_dir()):
                read = _read_camera(directory)
                if read is not None:
                    cameras[directory.name] = read

        observation_fields = [
            f for f in OBSERVATION_FIELDS if all(f in arms[("obs", s)] for s in SIDES)
        ]
        if "qpos" not in observation_fields:
            raise DatasetMismatch(f"{raw}: the arm observations have no qpos")
        streams = [arms[("obs", s)]["qpos"] for s in SIDES]
        streams += [arms[("action", s)]["qpos"] for s in SIDES]
        streams += [stream for stream, _ in cameras.values()]
        start = max(int(s.timestamps[0]) for s in streams)
        end = min(int(s.timestamps[-1]) for s in streams)
        if end < start:
            return None
        length = (end - start) * self.fps // 1_000_000_000 + 1
        times = start + (np.arange(length, dtype=np.int64) * 1_000_000_000) // self.fps

        features = self._features(observation_fields, cameras, raw)
        episode_index = self.total_episodes
        first_index = self.total_frames
        chunk_index, file_index = _chunk_file(episode_index)
        stats: dict[str, dict[str, np.ndarray]] = {}
        record: dict = {
            "episode_index": episode_index,
            "tasks": [self.tasks[task_index]],
            "length": int(length),
            "data/chunk_index": chunk_index,
            "data/file_index": file_index,
            "dataset_from_index": first_index,
            "dataset_to_index": first_index + int(length),
        }

        # Vectors, resampled: right arm then left.
        columns: dict[str, np.ndarray] = {}
        for field in observation_fields:
            feature = OBSERVATION_FIELDS[field][0]
            columns[feature] = np.concatenate(
                [
                    np.stack(arms[("obs", s)][field].values)[
                        arms[("obs", s)][field].at(times)
                    ]
                    for s in SIDES
                ],
                axis=1,
            ).astype(np.float32)
        columns["action"] = np.concatenate(
            [
                np.stack(arms[("action", s)]["qpos"].values)[
                    arms[("action", s)]["qpos"].at(times)
                ]
                for s in SIDES
            ],
            axis=1,
        ).astype(np.float32)
        for feature, values in columns.items():
            if values.shape[1] != features[feature]["shape"][0]:
                raise DatasetMismatch(
                    f"{raw}: {feature} has {values.shape[1]} values per frame"
                )
            stats[feature] = {**_stats(values), "count": np.array([length])}

        # Videos.
        for name, (stream, fmt) in cameras.items():
            if _is_depth_format(fmt):
                continue
            key = f"observation.images.{name}"
            files = [stream.values[i] for i in stream.at(times)]
            path = self.root / VIDEO_PATH.format(
                video_key=key, chunk_index=chunk_index, file_index=file_index
            )
            info, image_stats = encode_video(files, path, self.fps)
            shape = [info["video.height"], info["video.width"], 3]
            if features[key]["shape"] != shape:
                raise DatasetMismatch(
                    f"{raw}: {name} is {shape}, the dataset's {features[key]['shape']}"
                )
            features[key].setdefault("info", info)
            stats[key] = {**image_stats, "count": np.array([length])}
            record[f"videos/{key}/chunk_index"] = chunk_index
            record[f"videos/{key}/file_index"] = file_index
            record[f"videos/{key}/from_timestamp"] = 0.0
            record[f"videos/{key}/to_timestamp"] = float(length) / self.fps

        # Data parquet: vectors, depth images, then the default features.
        depth = {
            name: stream
            for name, (stream, fmt) in cameras.items()
            if _is_depth_format(fmt)
        }
        schema = self._schema(features)
        data_path = self.root / DATA_PATH.format(
            chunk_index=chunk_index, file_index=file_index
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = data_path.with_name(f".{data_path.name}.{os.getpid()}")
        depth_sample = set(_stats_sample(int(length)).tolist())
        depth_sampled: dict[str, list[np.ndarray]] = {name: [] for name in depth}
        depth_rows = {name: stream.at(times) for name, stream in depth.items()}
        with pq.ParquetWriter(temporary, schema, compression="snappy") as writer:
            for begin in range(0, int(length), ROW_BATCH):
                rows = range(begin, min(begin + ROW_BATCH, int(length)))
                batch = {}
                for feature, values in columns.items():
                    width = values.shape[1]
                    batch[feature] = pa.FixedSizeListArray.from_arrays(
                        pa.array(
                            values[rows.start : rows.stop].reshape(-1),
                            type=pa.float32(),
                        ),
                        width,
                    )
                for name, stream in depth.items():
                    key = f"observation.images.{name}"
                    entries = []
                    for k in rows:
                        image = _read_depth(stream.values[depth_rows[name][k]])
                        shape = [image.shape[0], image.shape[1], 1]
                        if features[key]["shape"] != shape:
                            raise DatasetMismatch(
                                f"{raw}: {name} is {shape}, the dataset's {features[key]['shape']}"
                            )
                        if k in depth_sample:
                            depth_sampled[name].append(_shrink(image[:, :, None]))
                        entries.append(
                            {"bytes": _depth_tiff(image), "path": f"frame-{k:06d}.tiff"}
                        )
                    batch[key] = pa.array(entries, type=schema.field(key).type)
                frame_index = np.arange(rows.start, rows.stop, dtype=np.int64)
                batch["timestamp"] = pa.array(
                    (frame_index / self.fps).astype(np.float32)
                )
                batch["frame_index"] = pa.array(frame_index)
                batch["episode_index"] = pa.array(
                    np.full(len(rows), episode_index, dtype=np.int64)
                )
                batch["index"] = pa.array(first_index + frame_index)
                batch["task_index"] = pa.array(
                    np.full(len(rows), task_index, dtype=np.int64)
                )
                writer.write_table(
                    pa.table([batch[f.name] for f in schema], schema=schema)
                )
        os.replace(temporary, data_path)
        for name, frames in depth_sampled.items():
            stats[f"observation.images.{name}"] = {
                **_image_stats(frames, 1.0),
                "count": np.array([length]),
            }

        frame_index = np.arange(length, dtype=np.float64)
        for feature, values in {
            "timestamp": frame_index / self.fps,
            "frame_index": frame_index,
            "episode_index": np.full(length, episode_index, dtype=np.float64),
            "index": first_index + frame_index,
            "task_index": np.full(length, task_index, dtype=np.float64),
        }.items():
            stats[feature] = {**_stats(values[:, None]), "count": np.array([length])}

        # Metadata last: the episode exists once these say so.
        self.features = features
        self.episodes.append(record)
        self.episode_stats.append(stats)
        self.openarm_episodes.append(
            {
                "episode_index": episode_index,
                "success": success,
                "task_index": task_index,
                "start_ns": int(times[0]),
                "length": int(length),
            }
        )
        self._write_meta()
        return episode_index

    def _features(self, observation_fields, cameras, raw) -> dict:
        features: dict = {}
        for field in observation_fields:
            feature, suffix = OBSERVATION_FIELDS[field]
            features[feature] = self._vector(suffix)
        features["action"] = self._vector("pos")
        for name, (stream, fmt) in cameras.items():
            key = f"observation.images.{name}"
            if _is_depth_format(fmt):
                image = _read_depth(stream.values[0])
                features[key] = {
                    "dtype": "image",
                    "shape": [image.shape[0], image.shape[1], 1],
                    "names": ["height", "width", "channels"],
                    "info": {"is_depth_map": True, "depth_unit": "mm"},
                }
            else:
                image = _read_rgb(stream.values[0])
                features[key] = {
                    "dtype": "video",
                    "shape": [image.shape[0], image.shape[1], 3],
                    "names": ["height", "width", "channels"],
                }
        features.update(copy.deepcopy(DEFAULT_FEATURES))
        if self.features is not None:
            old = {
                k: {kk: vv for kk, vv in v.items() if kk != "info"}
                for k, v in self.features.items()
            }
            new = {
                k: {kk: vv for kk, vv in v.items() if kk != "info"}
                for k, v in features.items()
            }
            if old != new:
                raise DatasetMismatch(
                    f"{raw}: its features differ from the dataset's: "
                    f"only in the dataset {sorted(set(old) - set(new))}, "
                    f"only in the episode {sorted(set(new) - set(old))}, "
                    f"changed {sorted(k for k in set(old) & set(new) if old[k] != new[k])}"
                )
            return copy.deepcopy(self.features)
        return features

    @staticmethod
    def _vector(suffix: str) -> dict:
        names = [f"{side}_{joint}.{suffix}" for side in SIDES for joint in ARM_JOINTS]
        return {"dtype": "float32", "shape": [len(names)], "names": names}

    @staticmethod
    def _schema(features: dict) -> pa.Schema:
        fields = []
        for key, feature in features.items():
            dtype = feature["dtype"]
            if dtype == "video":
                continue
            if dtype == "image":
                fields.append(
                    pa.field(
                        key, pa.struct([("bytes", pa.binary()), ("path", pa.string())])
                    )
                )
            elif feature["shape"] == [1]:
                fields.append(pa.field(key, pa.from_numpy_dtype(np.dtype(dtype))))
            else:
                fields.append(
                    pa.field(
                        key,
                        pa.list_(
                            pa.from_numpy_dtype(np.dtype(dtype)), feature["shape"][0]
                        ),
                    )
                )
        return pa.schema(fields)

    # -- metadata ---------------------------------------------------------

    def _write_meta(self) -> None:
        meta = self.root / "meta"
        rows = []
        for record, stats in zip(self.episodes, self.episode_stats):
            row = dict(record)
            for feature, values in stats.items():
                for name in STAT_NAMES:
                    value = np.asarray(values[name])
                    row[f"stats/{feature}/{name}"] = (
                        value.astype(np.int64).reshape(-1).tolist()
                        if name == "count"
                        else value.astype(np.float64).tolist()
                    )
            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = 0
            rows.append(row)
        _write_parquet_atomic(pa.Table.from_pylist(rows), self.root / EPISODES_PATH)
        _write_parquet_atomic(_tasks_table(self.tasks), meta / "tasks.parquet")
        _write_text_atomic(
            json.dumps(aggregate_stats(self.episode_stats), indent=4),
            meta / "stats.json",
        )
        has_video = any(f["dtype"] == "video" for f in self.features.values())
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "total_tasks": len(self.tasks),
            "chunks_size": CHUNKS_SIZE,
            "data_files_size_in_mb": DATA_FILES_SIZE_IN_MB,
            "video_files_size_in_mb": VIDEO_FILES_SIZE_IN_MB,
            "fps": self.fps,
            "splits": {"train": f"0:{self.total_episodes}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if has_video else None,
            "features": self.features,
        }
        _write_text_atomic(json.dumps(info, indent=4), meta / "info.json")
        openarm = copy.deepcopy(self.metadata)
        openarm["episodes"] = self.openarm_episodes
        _write_text_atomic(
            yaml.dump(openarm, default_flow_style=False, allow_unicode=True),
            self.root / OPENARM_METADATA_PATH,
        )
