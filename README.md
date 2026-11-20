# OpenArm データ収集（dora-rs）

[OpenArm](https://openarm.dev/) を [dora-rs](https://dora-rs.ai/) で動かし、Meta Quest 3（WebXR）で遠隔操作してデータを収集する手順です。

[enactic/dora-openarm-data-collection](https://github.com/enactic/dora-openarm-data-collection) のフォークです。
[guchio123 のフォーク](https://github.com/guchio123/dora-openarm-data-collection)の運用（この README の手順、
タスクのメタデータ、UI、pedestal/ZED、USB 接続）を引き継ぎ、WebXR テレオペを作り直しています。

- **ラグの原因だった IK の 1 ステップあたりの誤差制限を外しました。** 0.2 m/s・90°/s の手の動きで、
  手首の追従誤差は 23° → 0.8°、止めてから追いつくまで 525 ms → 6 ms（MuJoCo での実測、[docs/teleop.md](docs/teleop.md#measured)）。
- **グリップを握っている間だけ腕が手の動きに追従します（クラッチ）。** 離せば腕はその場で止まるので、手を楽な位置に戻して握り直せます。
- **速度・関節範囲・追従を監視する安全ゲート**が、IK とロボットの間に入っています。
- ゴーグル内に状態表示（HUD）が出ます。B は長押しでのみ終了します。接続にはペアリングトークンが必要です。

仕組みと数値の根拠は [docs/teleop.md](docs/teleop.md)（英語）にあります。

- [0. インストール](#0-インストール)
- [1. タスクとメタデータ](#1-タスクとメタデータ)
- [2. 起動手順](#2-起動手順)
- [3. 収集の操作](#3-収集の操作)
- [4. 終了手順](#4-終了手順)
- [5. 収録したデータを確認する](#5-収録したデータを確認する)
- [6. トラブルシューティング](#6-トラブルシューティング)
- [Configurations](#configurations)（英語）

## 0. インストール

```bash
sudo apt install libcli11-dev coturn   # coturn は USB 接続のときだけ必要
git clone --recursive https://github.com/Raptor-zip/dora-openarm-data-collection.git
cd dora-openarm-data-collection
uv sync
uv run scripts/webxr-setup             # 証明書とペアリングトークン（tls/）。最初の 1 回だけ
```

`scripts/webxr-setup` が表示する `https://…:8443/#k=…` が Quest で開く URL です。`#k=` 以降がトークンで、
これを知っている端末だけがロボットを動かせます。**他人に見せないでください。** 漏れたら
`uv run scripts/webxr-setup --rotate-token` で作り直します（保存済みの URL はすべて無効になります）。

## 1. タスクとメタデータ

タスク名、場所、オペレーターなどのメタデータは、すべて [`metadata.yaml`](metadata.yaml) の **1 ファイル**にまとめて書きます。

```yaml
location: Lab
operator: Tester

tasks:
  - prompt: Pick a spoon.          # UI に表示されるタスク名
    description: Pick a spoon on a table.
  - prompt: Pick a cup.
    description: Pick a cup on a table.
```

## 2. 起動手順

### かんたん起動（2-2 から 2-3 をまとめて）

CAN を立ち上げ（2-1）、Quest を USB-C でつないだら、リポジトリのフォルダで次の 1 行を実行します。

```bash
scripts/teleop
```

- Quest の USB 中継（`adb reverse` と TURN サーバー）と、ZED に必要な `zed` グループの扱いを、自動で済ませてから dataflow を起動します。
- **起動するとすぐ腕が動きます。** 腕の周りに人や物がない状態で実行してください。
- **このターミナルで Enter を押すと、UI の Enter（左腕の初期動作を進める。左手の X ボタンと同じ）になります。**
- Quest のブラウザで `https://localhost:8443/` を開き、ロボットと同じ方向を向いて Start を押します。
- Ctrl+C で止まります（腕は終了姿勢へ動いてから脱力します）。
- `ROBOT=pedestal TABLE_Z=-0.3 scripts/teleop` のように、環境変数はそのまま渡せます。

以下は、手順を 1 つずつ行う場合です。

### 2-1. CAN を立ち上げる

```bash
for i in can0 can1; do
  sudo ip link set $i down
  sudo ip link set $i type can bitrate 1000000 dbitrate 5000000 fd on
  sudo ip link set $i txqueuelen 1000
  sudo ip link set $i up
done
ip -br link | grep can     # can0 と can1 が UP になっていれば OK
```

### 2-2. ターミナル 1：USB 中継を起動する

Quest を USB-C でつないだ状態で、次を実行します。

```bash
cd ~/dora-openarm-data-collection
scripts/webxr-usb/start.sh
```

### 2-3. ターミナル 2：dataflow を起動する

```bash
cd ~/dora-openarm-data-collection
source .venv/bin/activate
TASK="Pick a spoon." LEADER=webxr ROBOT=pedestal-zed scripts/dora-envsubst run dataflow.yaml --uv
```

- PC のブラウザで http://localhost:8000/ を開くと、データ収集 UI が表示されます。
- **起動するとすぐ腕が動きます。** UI が起動時に Start Arm を送るので、両腕に力が入って開始姿勢へ移動し、左腕の初期動作も始まります。ロボットのそばで、腕が動いてよい状態にしてから起動してください。
- 腕は、重力補償付きの柔らかい制御（インピーダンス制御）で動きます。壁などに触れても強く押し付けにくくなります。ただし、腕を伸ばして指先で正面から突く方向は硬いままです。従来の制御に戻すときは `IMPEDANCE=off` を、硬さを変えるときは `IMPEDANCE_KP=25,25,15,15,10,10,10` のように 7 関節分の kp を付けて起動します（[docs/teleop.md](docs/teleop.md#impedance)）。
- 開始姿勢では、両手首を従来より 10 cm 体に近づけています。高さとグリッパーの向きは従来と同じです（`configs/openarm_pedestal_teleop.yaml` と `configs/openarm_pedestal_soft_gripper.yaml` の `start`。[docs/teleop.md](docs/teleop.md#start-pose)）。
- 左グリッパーのつかむ力は、従来の約 8 分の 1 にしています（`configs/openarm_pedestal_soft_gripper.yaml` の `gripper_posforce_limits`）。

起動前に、テレオペのノードどうしの設定（タイミング、周期、速度上限）の食い違いが自動で検査されます。
食い違いがあると起動せず、どこが違うかを表示します（[docs/teleop.md](docs/teleop.md#checks)）。

実機で机の高さを使う場合は、測った値を `TABLE_Z=-0.30`（m、`arm_origin` 基準）のように付けて起動します。
付けないと、実機では机への押し付け防止（指先を机より上に保つ制限）が**無効**になります（起動時に表示されます）。

### 2-4. Quest でテレオペを開始する

1. Quest のブラウザで、`scripts/webxr-setup` が表示した **`https://localhost:8443/#k=…`** を開きます。
   - 証明書の警告が出たら、「詳細設定」→「localhost にアクセスする」で進みます。
2. 数秒待ってから、**Start** を **1 回だけ**押します。
3. UI で **Start Arm** を押すと、**左腕が初期動作をします**（Start Arm のたびと、エピソードが終わるたびに行います）。
   1. グリッパーを閉じます。これで UI の Arm 表示が `ALIGNED` になります。
   2. 通常は右側にある爪が下になるまで手先を回し（爪同士の隙間が水平になります）、グリッパーを半分ほど開きます。
   3. ゴーグル内の表示が `L:PRESS X` になったら、**左手コントローラーの X ボタン**を押します
      （PC の UI（http://localhost:8000/）の Enter や Continue ボタンでも同じです）。
   4. グリッパーを閉じてから、通常は左側にある爪が下になるまで手先を回して止まります。表示が `L:HOLD` になれば追従待機です。左グリッパーは以後、閉じたままです（トリガーでは開きません）。

   右腕はその場で止まったまま、グリッパーをいったん閉じて `ALIGNED` になり、そのあと 60% 開きます。
   右グリッパーは以後、トリガーに関係なく 60% 開いたままです。
   （`FINGER_DOWN=right` を付けて起動すると、右手の「通常は右側にある爪」が常に下を向くよう、手首のひねりを固定します。既定では無効です。）
4. **グリップ（中指）を握ると**、その腕が手の動きに追従します。離すとその場で止まります。
   - 握る前に、手を 0.3 秒ほど止めてください。動かしながら握っても追従は始まりません。
   - ゴーグル内の表示が `R:FOLLOW` になっていれば右腕が追従中です。

## 3. 収集の操作

### コントローラー

| 操作 | 動作 |
|---|---|
| **グリップ**（中指）を握ったまま手を動かす | その腕の手先が、握った時点からの手の動きの分だけ動く |
| グリップを離す | その腕はその場で止まる。手を楽な位置に戻して、握り直せばまた動く |
| **トリガー**（人差し指） | 何もしない。右手のグリッパーは常に 60% 開いたまま、左手のグリッパーは初期動作のあと常に閉じたまま |
| **A ボタン**（右手） | 待機中：エピソードを**開始**<br>収録中：**成功**として保存 |
| **B ボタン**（右手） | 収録中：**失敗**として保存<br>待機中：**2 秒長押し**で終了（dataflow を止める） |
| **X ボタン**（左手） | `L:PRESS X` のとき：左腕の初期動作を先に進める（グリッパーを閉じて、左側の爪を下へ回す） |

頭を動かしたり体を傾けたりしても、腕は動きません。
コントローラーを見失った手は、その場で一時停止します。0.5 秒以内に同じ位置へ戻れば、握り直さずに再開します。

### ゴーグル内の表示（HUD）

| 表示 | 意味 |
|---|---|
| `READY` / `REC 01:05  ep 7` | 待機中 / 収録中（経過時間、エピソード番号） |
| `R:FOLLOW` | 右腕が手に追従している |
| `R:HOLD` | グリップを離していて、腕は止まっている |
| `R:REGRIP` | いったんグリップを離してから握り直す必要がある（起動直後、Start Arm の後など） |
| `R:LATE` | コントローラーの入力が途切れている |
| `L:SETUP` / `R:SETUP` | 左腕が初期動作で、右腕が爪を下に向けるために動いている（手の動きには追従しない） |
| `L:PRESS X` | 左腕が初期動作の途中で、X ボタン（または PC の UI の Enter）を待っている |
| `GATE R:HOLDING(理由)` | 安全ゲートが右腕を止めている。理由は [docs/teleop.md](docs/teleop.md#gate-dora-openarm-gate) |

開始・成功・失敗で右手、取り消しで左手、終了で両手のコントローラーが振動します。

### PC の UI（http://localhost:8000/）

| ボタン | キー | 動作 |
|---|---|---|
| Start! | `c` | エピソードを開始する |
| Success! | `c`（収録中） | 成功として保存する |
| Fail! | `a`（収録中） | 失敗として保存する |
| Cancel! | `a` 長押し（収録中） | 保存せずに破棄する |
| Skip task | `c` 長押し | 次のタスクに切り替える |
| Start Arm / Pause Arm | `b` 長押し | 腕の起動 / 停止 |
| Continue [Enter] | `Enter` | 左腕の初期動作を先に進める（`L:PRESS X` のときだけ効く。X ボタンと同じ） |
| Quit | `b` | 終了する |

### 1 エピソードの流れ

1. 腕が `ALIGNED` で、ゴーグル内に `GATE` の行が出ておらず、物体が初期位置に置かれていることを確認します。
2. **A**（または UI の Start!）を押して、収録を開始します。
3. タスクを実行します。
4. 成功したら **A**、失敗したら **B** を押します。撮り直す場合は、UI の Cancel! で破棄します。
5. エピソードが終わると、**両腕が自動で開始姿勢へ戻ります**（手を離した位置から、平均 0.1 m/s）。続けて左腕は初期動作をやり直します（爪を右下へ回して、グリッパーを半分開く）。その間に物体を初期位置に戻し、左グリッパーに渡すものがあれば渡します。
6. 表示が `L:PRESS X` になったら **X** を押します。左グリッパーが閉じ、左側の爪が下になるまで回って止まります。
7. グリップを握り直してから、1 に戻ります（戻った腕は、グリップをいったん離して握り直すまで動きません）。

腕を毎回戻さず、Start Arm のときだけ初期動作をさせたいときは、`INTRO_EVERY_EPISODE=off` を付けて起動します。

> 開始してすぐに A を押すと、0.2 秒ほどのごく短いエピソードが保存されます。押す前に、いま収録中かどうかを UI で確認してください。

### 腕を初期姿勢に戻す

UI で **Pause Arm** を押してから **Start Arm** を押します。

- **Pause Arm**：終了姿勢（肘を 90°）へ 2 秒かけて移動したあと、**モーターの力が抜けます**。重力で腕が落ちることがあるので、**支えられる状態で押してください**。
- **Start Arm**：モーターに力を入れ、開始姿勢へ移動します。そのあと左腕が初期動作をやり直すので（[2-4](#2-4-quest-でテレオペを開始する) の 3）、Enter を押してから、グリップを握り直して操作を再開します。

## 4. 終了手順

1. 収録中でないことを確認してから、**B を 2 秒長押し**します（または UI の Quit）。
2. 腕が終了姿勢に移動して止まったら、ターミナル 2 で dataflow が終了したことを確認します。終了しない場合は Ctrl+C で止めます。
3. ターミナル 1 の USB 中継を Ctrl+C で止めます。
4. 腕を支えてから、ロボットの電源を切ります。

## 5. 収録したデータ

データは [LeRobot](https://github.com/huggingface/lerobot) Dataset **v3.0** 形式で、`data/<LEADER>-<ROBOT>/lerobot/` に保存されます（LeRobot 0.7 で読み込めることを確認済み）。

```
data/<LEADER>-<ROBOT>/lerobot/
├── meta/
│   ├── info.json            # fps（30）、特徴量の定義、総エピソード数・フレーム数
│   ├── stats.json           # 特徴量ごとの統計（正規化用）
│   ├── tasks.parquet        # タスク文（metadata.yaml の prompt）→ task_index
│   ├── episodes/chunk-000/file-000.parquet   # エピソードごとの長さ、動画の位置、統計
│   └── openarm.yaml         # metadata.yaml の内容と、エピソードごとの成功/失敗
├── data/chunk-000/file-NNN.parquet           # 1 エピソード 1 ファイル（30 fps の各フレーム）
└── videos/observation.images.<カメラ名>/chunk-000/file-NNN.mp4   # AV1
```

| 特徴量 | 中身 |
|---|---|
| `observation.state` | 実際の関節角（右 J1〜J7、右グリッパー、左 J1〜J7、左グリッパーの 16 個） |
| `observation.velocity` / `observation.effort` | 実際の関節速度 / トルク（同じ並び） |
| `action` | 指令値（安全ゲートの出力。同じ並び） |
| `observation.images.head_left` など | カラーカメラの動画（AV1、30 fps） |
| `observation.images.head_depth` | 深度画像（16 bit、mm、0 は深度なし）。動画ではなく、parquet の中に TIFF で入ります |

- 各フレームには、その時刻以前で最新のデータが入ります（腕は 250 Hz、カメラは 30 fps で届いたものを 30 fps にそろえます）。
- 成功/失敗は LeRobot の形式に項目がないので、`meta/openarm.yaml` の `episodes` に入ります。
- 収録中のデータは `.staging/` に一時保存され、成功/失敗を押したあと裏で v3 に変換されます（18 秒のエピソードで約 14 秒）。変換が終わると一時データは消えます。破棄（Cancel）したエピソードはすぐ消えます。
- 変換の前に dataflow が止まっても、次の起動時に残りを変換します。変換できなかったエピソードは `.staging/failed/` に残ります。
- エピソード番号（`episode_index`）は 0 から連続で振られます。UI に表示される番号とは一致しないことがあります。
- 容量の目安は、5 カメラ＋深度で 1 分あたり約 300 MB です。
- 深度の特徴量があるデータセットを LeRobot で読み込むには、読み込む側の PyAV に HEVC エンコーダー（libx265）が必要です（LeRobot 0.7 が読み込み時に確認します）。
- `scripts/view-episode.py` は以前の形式（`data/<LEADER>-<ROBOT>/dataset/`）用で、v3 のデータは読めません。

## 6. トラブルシューティング

| 症状 | 原因と対処 |
|---|---|
| `uv: command not found` | [uv](https://docs.astral.sh/uv/getting-started/installation/) をインストールします。 |
| Quest でページが開かない / `wrong or missing pairing token` と出て接続できない | URL に `#k=…` が付いているか確認します（`scripts/webxr-setup` が表示したもの）。トークンを作り直した場合は新しい URL を開きます。USB の場合は、`adb devices` が `device` になっているか確認します。`unauthorized` なら、ゴーグル内で USB デバッグを許可します。そのあと `scripts/webxr-usb/start.sh` を起動し直します。 |
| Start を押しても反応しない | 前回の XR セッションが残っています（コンソールに `already an active, immersive XRSession` と出ます）。Quest の**ブラウザアプリを完全に終了**して開き直します。それでも直らなければ、Quest を再起動します。 |
| Start は押せるのに腕が動かない | ターミナル 1 の USB 中継（TURN）が止まっていないか確認します。Wi-Fi では端末どうしの通信が遮断されていることがあるので、USB 接続を使ってください。 |
| 左腕が `L:SETUP` のまま回り始めない | 初期動作は、左グリッパーが実際に閉じるまで待ちます（閉じたことで腕が指令を受け付けたと確認するため）。クラッチのログ（`out/<実行ID>/log_leader.clutch.jsonl`）に `waiting for the arm to close the gripper` と出たままなら、下の `ALIGNED` にならない場合と同じ原因です。初期動作を使わないときは `INTRO_ARM=none` を付けて起動します。右腕が `R:SETUP` のまま回り始めないときや、右グリッパーが 60% に開かないときも、同じく閉じるのを待っています。 |
| 片方の腕が `ALIGNED` にならない | ロボット側のログに `arm[N] went silent` と出ていれば、N 番（0 から数える）のモーターが応答していません。たとえば `arm[5]` は J6（手首の曲げ）です。その付近のケーブルと電源を確認します。 |
| ログに `write_no_buffer` が大量に出る | CAN の送信キューが小さすぎます。`txqueuelen 1000` を設定します（[2-1](#2-1-can-を立ち上げる) を参照）。 |
| グリップを握っても腕が動かない | ゴーグル内の表示を見ます。`REGRIP` ならいったん離して握り直します。`--` や `LATE` ならコントローラーが追跡されていません。`GATE … HOLDING` なら理由を [docs/teleop.md](docs/teleop.md#gate-dora-openarm-gate) で確認します。手を止めずに握ると追従は始まりません。 |
| 起動時に `dora-envsubst: leader: …` と出て起動しない | テレオペのノードどうしの設定が食い違っています。メッセージの項目を [`modules/leader-webxr.yaml`](modules/leader-webxr.yaml) で直します（[docs/teleop.md](docs/teleop.md#checks)）。 |
| 追従が遅い | IK に `--frame-position-error-limit` などの誤差制限を付けていないか確認します（付けると遅くなります。[docs/teleop.md](docs/teleop.md#measured)）。実機では安全ゲートが関節速度をドライバ上限の 0.5 倍に抑えています。 |
| UI のボタンを押しても反応が遅い | UI のページを再読み込みします（改善済みですが、古いページを開いたままの場合に起こります）。 |
| エピソードが保存されていない | レコーダーのログ（`out/<実行ID>/log_recorder.jsonl`）を確認します。`Episode N written` と出ていれば保存済みです。`metadata does not match` と出ていれば、[1 章](#1-タスクとメタデータ)の注意を参照してください。`Could not add raw episode` と出ていれば、その生データは `.staging/failed/` に残っています。 |
| 深度画像が真っ黒に見える | 正常です。値が mm（数百〜千程度）で、16 bit の最大値 65535 に比べて小さいため、普通の画像ビューアでは黒く見えます。 |

### ログの場所

dataflow のログは、実行ごとに `out/<実行ID>/log_<ノード名>.jsonl` に保存されます。

```bash
d=$(ls -td out/*/ | head -1)       # 最新の実行
tail -5 $d/log_robot.right.jsonl   # 右腕
tail -5 $d/log_recorder.jsonl      # レコーダー
tail -5 $d/log_leader.gate.jsonl   # 安全ゲート（止めた理由）
```

Quest のブラウザ側のエラーは `out/quest.log` に保存されます。

## Configurations

[`dataflow.yaml`](dataflow.yaml) is the only dataflow. The leader and the robot
are modules under [`modules/`](modules) picked by environment variables:

| Variable | Values | Picks | Default |
|---|---|---|---|
| `LEADER` | `ker`, `vr`, `webxr`, `dummy` | `modules/leader-$LEADER.yaml` | `dummy` |
| `ROBOT` | `cell` (real Cell), `pedestal`, `pedestal-zed`, `mujoco`, `dummy` | `modules/robot-$ROBOT.yaml` | `dummy` |

dora doesn't expand environment variables in `module:` paths
([dora-rs/dora#3805](https://github.com/dora-rs/dora/issues/3805)), so run the
dataflow through [`scripts/dora-envsubst`](scripts/dora-envsubst). It expands
`$VAR`, `${VAR}` and `${VAR:-default}` in `module:` paths, writes the result to
`<name>.envsubst.yaml` and runs the given dora command on it.

`ROBOT=cell` and `ROBOT=pedestal-zed` capture the head ZED Mini with the ZED
SDK, which computes depth on the GPU. Install the ZED SDK first; see
[`nodes/dora-openarm-zed`](nodes/dora-openarm-zed/README.md). The depth is
recorded as `cameras/head_depth/<timestamp>.png` (16-bit, mm) and shown in
the UI.

Only the selected leader and robot are in the dataflow, so the others open no
device or port. `LEADER=dummy` with `ROBOT=cell` or `ROBOT=pedestal` moves the
real arms with generated motion. See [`docs/design.md`](docs/design.md) for how
the modules are structured.

With `LEADER=webxr`, `TABLE_Z` (m, in the `arm_origin` frame) is the
measured table for the clutch on a real robot; see [`docs/teleop.md`](docs/teleop.md).
`INTRO_ARM` (default `left`) is the arm that runs the intro after each Start
Arm, and after each episode unless `INTRO_EVERY_EPISODE=off`, and waits for
the left controller's X (or Enter in the UI), `FIXED_GRIPPERS` (default `right=0.6,left=0`)
the grippers held at a fixed opening whatever the trigger does, as a fraction
of their range (the left one closed once its intro is done), and `FINGER_DOWN` (default `none`) the arms whose finger
normally on the right is kept down; `none` turns any of them off
([`docs/teleop.md`](docs/teleop.md#intro)).
`scripts/dora-envsubst` checks the teleop nodes' couplings before `run`,
`start` and `build` (`SKIP_TELEOP_CHECK=1` skips it).

`TASK` picks the task to start with from the `tasks` in [`metadata.yaml`](metadata.yaml),
by index or by its prompt (exact, or a unique part of it, case-insensitive).
It defaults to the first task, and Skip in the UI still moves to the next one.
An unknown or ambiguous `TASK` stops the UI with the list of tasks:

```bash
TASK="Pick a cup." LEADER=webxr ROBOT=mujoco scripts/dora-envsubst run dataflow.yaml --uv
```

All dataset metadata is in one file, [`metadata.yaml`](metadata.yaml): the
location, the operator, the tasks and the equipment common to every robot,
plus a `robots:` section with what each robot adds, picked by `ROBOT`.
Datasets are recorded under `data/$LEADER-$ROBOT/`, e.g. `data/webxr-mujoco/`:

| Robot | Recorded equipment |
|---|---|
| `cell` | Arms, lifter, five cameras, head depth |
| `pedestal`, `pedestal-viz` | Arms only |
| `pedestal-zed` | Arms, wrist cameras, head ZED Mini with depth |
| `mujoco` | Arms, five simulated cameras |
| `dummy` | Arms, five dummy cameras |

The recorder writes a LeRobot v3.0 dataset to `data/$LEADER-$ROBOT/lerobot/`
(`NAME`, `FPS` = 30; see
[`nodes/dora-openarm-dataset-recorder`](nodes/dora-openarm-dataset-recorder/README.md)).
It refuses to append to a dataset recorded with different metadata, e.g.
after editing the tasks, so move `data/$LEADER-$ROBOT/lerobot/` away first.

### KER leader

`LEADER=ker` uses a [KER](https://github.com/enactic/dora-openarm-ker) leader arm
connected over USB. Its joint positions are sent to the follower arms directly.

### VR leader

`LEADER=vr` receives VR controller poses over UDP by
[dora-openarm-vr](https://github.com/enactic/dora-openarm-vr), converts them to
joint positions by inverse kinematics
([dora-openarm-kinematics](https://github.com/enactic/dora-openarm-kinematics))
and sends them to the follower arms.

### WebXR leader

`LEADER=webxr` uses [dora-openarm-webxr](https://github.com/enactic/dora-openarm-webxr).
It starts a Web server and the Web browser on a VR device such as Meta Quest 3
or PICO 4 connects to it to stream controller poses. No native VR application is
needed.

WebXR requires HTTPS, so a TLS certificate is needed. A self-signed certificate is enough; see the [dora-openarm-webxr setup instructions](https://github.com/enactic/dora-openarm-webxr#setup) for how to generate one. Then run:

```bash
./nodes/dora-openarm-webxr/example/prepare_tls.sh $(hostname).local
LEADER=webxr ROBOT=mujoco scripts/dora-envsubst run dataflow.yaml --uv
```

Open http://localhost:8000/ on the local machine for the data collection UI, and open `https://<host name>:8443/` in the Web browser on your VR device (where `<host name>` is the value passed to `prepare_tls.sh`) to start teleoperation.

### Dummy leader

`LEADER=dummy` generates leader motion without any device. With `ROBOT=dummy`,
we can test the dataflow without any hardware.

### Cell robot

`ROBOT=cell` uses real OpenArm units on CAN, the Cell lifter and wrist, head and
ceiling cameras.

The Cell lifter receives the shared 250 Hz tick and records its elevation action
and observation in millimeters. It uses its built-in startup calibration and hold
behavior, and the left joystick of a VR leader moves it. A real lifter is
required on `can2` by default, and startup enables and moves it for calibration.

### Pedestal robot

`ROBOT=pedestal` uses real OpenArm units on a pedestal over CAN. It has no
lifter and no cameras. The followers use the `openarm_pedestal` configuration
with the wrists starting 10 cm closer to the body
([`configs/openarm_pedestal_teleop.yaml`](configs/openarm_pedestal_teleop.yaml);
[docs/teleop.md](docs/teleop.md#start-pose)), the left one also with about an
eighth of the grip force
([`configs/openarm_pedestal_soft_gripper.yaml`](configs/openarm_pedestal_soft_gripper.yaml)).
All pedestal robots run their arms with gravity compensation and soft joint
gains (`IMPEDANCE`, `IMPEDANCE_KP`, `GRAVITY_SCALE`); see
[`docs/teleop.md`](docs/teleop.md#impedance).

### MuJoCo robot

`ROBOT=mujoco` uses a MuJoCo simulation
([dora-openarm-mujoco](https://github.com/enactic/dora-openarm-mujoco)) instead
of real OpenArm units and cameras. We can use this for testing teleoperation and
data collection without real hardware.

### Dummy robot

`ROBOT=dummy` doesn't use any hardware. The arms echo their targets and the
cameras send test images.

## Debugging OpenArm libraries

The OpenArm-specific dependency repositories are checked out under [`lib/`](lib/README.md).
Initialize them with `git submodule update --init --recursive`.

Libraries normally come from the package index at the versions in `uv.lock`.
To use an editable library checkout, run the command for the library you need
from the repository root:

```bash
uv add --editable lib/openarm_driver
# Other optional library checkouts:
# uv add --editable lib/openarm_can/python
# uv add --editable lib/openarm_ker
# uv add --editable lib/openarm_control
# uv add --editable lib/openarm_mujoco
```

`uv add --editable` updates the parent dependency declaration, adds a local path
under `[tool.uv.sources]`, updates the lockfile, and syncs the environment.
Restart the dataflow after editing Python code. CAN bindings use
`lib/openarm_can/python`; changes to their C++ code require rebuilding.

To return to the package-index version, remove that library's entry from
`[tool.uv.sources]`, then run `uv lock` and `uv sync --locked`. Keep required
version constraints such as `openarm-driver>=0.5.1`. Avoid separate `uv pip install`
commands for managed dependencies, since they bypass the project lockfile.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
