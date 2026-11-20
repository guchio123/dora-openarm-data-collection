# dora-openarm-dataset-recorder

A [Dora](https://dora-rs.ai/) node that records data as a
[LeRobot](https://github.com/huggingface/lerobot) dataset, format v3.0 (what
lerobot 0.7 reads), in `$DIRECTORY/$NAME` (`NAME` defaults to `lerobot`).

## How an episode is written

- **While recording**, the episode goes to a raw directory,
  `<dataset>/.staging/episodes/<n>`. Camera frames are written as they
  arrive, one file each, named by their Unix-ns timestamp. Arm states are
  kept in memory and written as parquet when the episode ends, followed by
  `episode.json` (task, success).
- **On success or fail**, a background thread appends the episode to the
  dataset and deletes the raw directory (`lerobot_v3.py`). So the next episode
  can start meanwhile. On cancel, the raw directory is deleted at once.
- **At startup**, raw episodes with an `episode.json` are converted first.
  Ones without it were never finished and are left alone.
- **On failure**, the raw directory moves to `.staging/failed/`.

The conversion resamples every stream to `FPS` (30) frames per second. Frame k
is at `t0 + k / fps`, where t0 is the first instant every stream has data, and
each stream gives its latest sample at or before that instant.

| Feature | From |
|---|---|
| `observation.state`, `observation.velocity`, `observation.effort` | measured qpos, qvel, qtorque; right arm then left, 8 each |
| `action` | commanded qpos, right then left |
| `observation.images.<camera>` (video) | JPEG cameras, encoded with libsvtav1, yuv420p, GOP 2 and CRF 30 (lerobot's defaults) |
| `observation.images.<camera>` (image) | 16-bit PNG cameras (depth, mm), stored in the data parquet as 16-bit TIFF, with `is_depth_map: true` and `depth_unit: mm` |

Success or fail, and the metadata file's contents, are in `meta/openarm.yaml`.
Data and videos are one file per episode. The dataset refuses an episode whose
features (cameras, sizes) differ from the ones already in it.

## Timestamps

The first `arm_left_observation` or `arm_right_observation` selects the timestamp
field for both arms for the recorder process lifetime, including before recording
starts. It selects `metadata.observation_timestamp` when present, otherwise
`metadata.timestamp` for older producers, and logs the choice once.

The selection persists across episode completion, cancellation, and subsequent
starts. Later messages must contain the selected field; a missing field raises
an error. If `timestamp` was selected, later `observation_timestamp` fields are
ignored. Restart the recorder to select a different field.

Integer timestamps are Unix nanoseconds. Actions, lifter observations, and camera
images continue to use `metadata.timestamp` independently of the arm selection.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
