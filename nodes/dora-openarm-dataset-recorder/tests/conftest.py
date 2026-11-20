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

import json

import pytest

import dora_openarm_dataset_recorder.main as recorder


class KeepRawConverter:
    """Stands in for the LeRobot conversion: keeps each raw episode as it is
    and records what was handed over, so tests can look at the raw data."""

    def __init__(self, dataset, staging):
        self.dataset = dataset
        self.staging = staging
        self.episodes = []
        KeepRawConverter.last = self

    def submit(self, raw):
        info = json.loads((raw / recorder.EPISODE_INFO_FILE).read_text())
        self.episodes.append(
            {
                "id": str(info["number"]),
                "success": info["success"],
                "task_index": info["task_index"],
            }
        )

    def call(self, function):
        function(self.dataset)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def keep_raw_episodes(monkeypatch):
    monkeypatch.setattr(recorder, "EpisodeConverter", KeepRawConverter)
    return KeepRawConverter
