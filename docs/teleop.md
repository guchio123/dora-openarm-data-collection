# WebXR teleoperation (`LEADER=webxr`)

How a Meta Quest moves the arms, why each stage exists, and where its numbers
come from. The operator's steps are in the [README](../README.md) (Japanese);
the dataflow rules are in [design.md](design.md).

```
webxr --pose/grip--> clutch --target--> ik --position--> gate --cmd--> robot
 (Quest page)          |  ^                                ^  |
                       |  +----------- state_* ------------+  +-- start_epoch
                       +--clutch_state--> hud <--gate_state
```

All of it is [`modules/leader-webxr.yaml`](../modules/leader-webxr.yaml). The
nodes other than `webxr` are the [`dora-openarm-teleop`](../nodes/dora-openarm-teleop)
package. They are ported from openarm-quest-teleop, a MuJoCo and Quest 3
teleoperation project, where they were developed against measured headset
sessions.

## Controls

| Input | Does |
|---|---|
| **Grip** (middle finger), held | That arm follows the hand's *motion* since the squeeze |
| Grip released | That arm holds where it is |
| **Trigger** (index finger) | Nothing by default: the right gripper stays 60% open and the left one closed (`FIXED_GRIPPERS`). For a gripper not in `FIXED_GRIPPERS`: pulled closes |
| Turning the hand about the fingers | Ignored for an arm in `FINGER_DOWN` (none by default), whose finger normally on the right stays down |
| **A** | Idle: start an episode. Recording: save it as a success |
| **B** | Recording: save it as a failure. Idle: **hold 2 s** to quit |
| **X** (left controller), or **Enter** in the UI (http://localhost:8000/) | The left arm's intro, while it waits: close and flip (see [Intro](#intro)) |

The arm follows the hand's displacement, not its absolute position. So you can
let go, move your hand somewhere comfortable, and squeeze again: the arm doesn't
jump (a "clutch", like lifting a mouse off the desk). Turning or leaning your
head doesn't move an engaged arm (`PIVOT_MODE=latched`).

Quit needs a 2 s hold because B is also the failure button. An operator who
fails an episode and keeps B down for a moment no longer ends the session.

## Stages

### webxr

[`nodes/dora-openarm-webxr`](../nodes/dora-openarm-webxr) is enactic's node
with the patches listed in its [`PATCHES.md`](../nodes/dora-openarm-webxr/PATCHES.md):

- Pose smoothing with a reset after gaps, latched neck pivot.
- The HUD panel and haptics.
- A pairing token in the URL, so that only your headset can drive the robot.
- A log of the page's own errors (`out/quest.log`).
- USB relaying (patch 18).

### Clutch (`dora-openarm-clutch`)

- **Engage.** An arm is armed when its grip has been released and the hand has
  been still for 300 ms (< 0.05 m/s, < 0.5 rad/s). Then the grip engages it at
  0.7 squeeze and releases it at 0.4. The hysteresis means a half-held grip
  doesn't chatter.
- **Gaps.** A hand pose older than 100 ms makes the arm hold. If the input comes
  back within 500 ms, within 20 mm and 5° of where it left, the arm resumes
  without a re-grip. Wi-Fi drops are usually this short.
- **Clamps.** Every tick it clamps the target to the arm's reach, keeps it on its
  own side, and keeps the fingertips above the table.
  - The table is the MuJoCo scene's when `ROBOT` is `mujoco` or `dummy`.
  - On a real robot, give the measured table height as `TABLE_Z`, in meters, in
    the `arm_origin` frame. Without `TABLE_Z` the table clamp is **off**, and
    the node says so at startup. Clamping against the simulated table's height
    would be wrong in either direction.
- **Where it starts.** The first `state_*` of each arm, and the first after each
  follower restart, sets that arm's hold to where the arm actually is, locked
  until the grip is released.
  - The follower's `start_epoch` metadata tells a restart (Pause Arm, then
    Start Arm) apart.
  - Without this, the IK would start at the scene's home pose. The Cell's start
    pose is 0.2 rad away from it in J4 (`openarm_driver` config), which is
    exactly the gate's `jump_tol`, so the gate would never engage.
  - Later states are ignored, because holding the measured pose would make the
    hold sag with the arm.
- **Fixed grippers.** `FIXED_GRIPPERS` (`right=0.6,left=0` in
  `leader-webxr.yaml`; `none` turns it off) holds a gripper at a fixed
  fraction of its range open, whatever the trigger does: the right one at 60%,
  -0.471 rad (27°), the left one closed. The left arm's intro still opens and
  closes its gripper; the fixed opening applies once the intro is done.
  - After each Start Arm it is first commanded closed, until it has been for
    0.3 s and the arm measures it closed. dora-openarm runs with
    `--align-trigger gripper` and executes nothing (`STARTED`, not `ALIGNED`)
    until the commanded gripper is closed, so a gripper held open from the
    start would keep that arm from ever moving.
  - In MuJoCo the right gripper closed 0.4 s after the start and opened right
    after.
- **Finger down.** `FINGER_DOWN` (`none` in `leader-webxr.yaml` for now;
  `FINGER_DOWN=right` turns it on for the right arm) keeps an arm's finger
  that is normally on its right lowest
  (`intro.finger_down`). The target follows the hand's position and the
  direction its fingers point, but its roll about that direction is set: the
  finger on the right straight down when the fingers point level, as far down
  as a tilt allows otherwise. Within 20° of pointing straight up or down the
  roll fades back to the hand's, since neither finger is lower there.
  - It is a constraint on the IK's target: whatever the IK solves for has it.
  - After each Start Arm the arm first waits for the same alignment as a fixed
    gripper, then turns into it at 0.5 rad/s (90° in 3.1 s), shown as `SETUP`.
    It does not follow the hand until the turn is done and the grip has been
    released, as after any start.
  - The pose sits near the joint limits: in MuJoCo the right J5 ended at
    82.9° (gate range ±87°) and J2 at -6.3° (gate range from -7.1°). Some
    hand directions may not be reachable there; the gate then holds
    (`ik_limit`) rather than following.

#### Intro

`INTRO_ARM` (`left` in `leader-webxr.yaml`; `none` turns it off) names an arm
that, after each Start Arm and after each episode, runs a fixed sequence before
the clutch takes it:

0. After an episode only: the end effector goes back from wherever the hand
   left it to the start pose (the FK of where the arm measured after its last
   Start Arm), in a straight line and a slerp, eased, at 0.1 m/s and 0.5 rad/s
   on average, whichever takes longer. The gripper stays as it was.

1. The gripper closes. dora-openarm runs with `--align-trigger gripper`: it
   executes nothing (the UI shows `STARTED`, not `ALIGNED`) until the commanded
   gripper is closed. So nothing turns until the measured gripper has closed,
   which shows that the commands are reaching the arm.
2. The end effector rolls about its approach axis, levelled to horizontal,
   until the finger that is normally on the right points down and the gap
   between the fingers is level. Then the gripper opens halfway.
3. It waits for **X** on the left controller (a press, the edge from released
   to pressed) or **Enter** in the UI, which sends the command `proceed`.
4. The gripper closes, until it measures closed or for at most 2 s, since it
   may be closing on something.
5. The end effector rolls back through the start orientation and on until the
   finger that is normally on the left points down.

Then the clutch holds that pose: release the grip if it is squeezed, and
squeeze to follow, as after any Start Arm. The hand is ignored until then.

- "Normally" is the start pose: the fingers point forward and the
  ee_control_point site's +y is the robot's left for both arms, so the finger on
  the right is on the site's -y side.
- The roll is a Cartesian target that goes through the IK and the gate like a
  followed one. It turns at 0.5 rad/s on average (`--intro-turn-speed`),
  easing in and out, so 90° takes 3.1 s and the flip 6.3 s.
- A roll of 90° is J5's whole range (±1.5708) and the gate keeps 0.05 rad
  inside it, so the IK brings in the shoulder too. In the IK model the turn
  is exact for the first roll and within 1° and 2 mm for the flip, at the edge
  of J2's range. In MuJoCo the gate held nothing during the whole sequence.
- `--intro-open` sets how far the gripper opens (fraction of its range, 0.5).
- The HUD shows `SETUP` while it moves and `PRESS X` while it waits. A Pause
  Arm and Start Arm runs it again from the start pose.
- An episode ends with the UI command `success`, `fail` or `cancel`, which the
  clutch also receives. The sequence then starts again with step 0, so that
  episodes can follow each other without a Pause Arm and Start Arm. The other
  arm goes back to its start pose the same way (step 0 alone, gripper as it
  is), and then holds there, locked until the grip is released.
  `INTRO_EVERY_EPISODE=off` keeps all of it to Start Arm. An end that arrives
  while an arm is still on its way changes nothing for that arm.
- In the MuJoCo robot, X also re-places the scene's objects (dora-openarm-mujoco).

### IK (`dora-openarm-teleop-ik`)

`dora-openarm-teleop-ik` starts enactic's `dora-openarm-ik` with the following:

- **No per-solve frame-error limits.**
  - The noguchi fork passes
    `--frame-position-error-limit 0.001 --frame-orientation-error-limit 0.004`.
    That moves the target at most 1 mm and 0.004 rad per solve, which at 250
    solves/s caps the hand at 0.25 m/s and the wrist at 1 rad/s (57°/s). That
    is the lag operators feel.
  - Speed is bounded where it belongs: in joint space, by the gate.
- **The gate's joint ranges.** A copy of the MJCF with each J1-J7 range
  narrowed to the gate's range minus its margin, so the IK doesn't plan
  configurations that the gate then refuses.
- **Joint speeds at 0.9× the gate's caps** (`--config`). The IK's built-in caps
  are faster than the gate's. The gate would then clip every tick of a fast
  reach, and the arm would lag the IK's plan.
- **`--tick-hz` equal to the leader tick.** The IK integrates its velocity caps
  over 1/tick-hz. If it says 200 while ticks come at 250, the arm moves 25 %
  faster than the caps say. A unit test caught exactly this.
- **Pinned to the fast cores** on a big.LITTLE CPU, read from `cpu_capacity`.
  On a homogeneous CPU it doesn't pin.

The generated MJCF and config are cached under `~/.cache/dora-openarm-teleop/`
and keyed by their settings.

**Known limitation: at a joint limit the IK keeps the position and gives up
the orientation.**

- J2 stops the upper arm 10° past hanging straight, so a hand moved inward
  soon asks for more than the arm can do at that orientation.
- Upstream caps the orientation error each solve uses (0.25 rad), but caps the
  position error only while the target moves faster than 0.6-0.9 m/s. So a
  level hand moved slowly sideways past the limit tilts the gripper. From 36
  recorded poses of each arm, ±0.15 m in 0.8 s tilted it up to 42° (right) and
  44° (left).
- Capping the position error at every speed, with `--ori-cost 3` and caps of
  0.05 m / 0.5 rad, held the tilt to 1.9° / 3.4° in simulation. On the robot
  it was worse to drive. The left arm's wrist sits at J5's limit after its
  intro, so most hand turns can't be reached there. That IK then moved the
  whole arm to turn the gripper: up to 3.6 cm forward or back for a 29° roll
  of the hand, against 0.2 cm here. The operator saw it as the arm
  exaggerating forward and back motion, so it was taken out.

### Gate (`dora-openarm-gate`)

The last stage before the robot. Every tick, each arm's command moves towards
the IK's solution at no more than the joint speed caps:

| Profile | `velocity_scale` | Used when |
|---|---|---|
| sim | 0.8 × `joint_velocity_limits` | `ROBOT` is `mujoco` or `dummy` |
| real | 0.5 × `joint_velocity_limits` (J1 1.0 rad/s, J5-J7 3.15 rad/s) | any other `ROBOT` |

`joint_velocity_limits` comes from
[`joint_limits.yaml`](../nodes/dora-openarm-teleop/src/dora_openarm_teleop/joint_limits.yaml),
which also gives each joint range and its source. The ranges are the narrowest
of openarm_driver's pedestal config and the MJCF. **The Cell configs are not
checked against them.**

The gate holds, and says why, when any of these happens:

- the IK is older than 50 ms;
- the robot's state is older than 50 ms;
- the IK jumps more than 0.2 rad from the command;
- the IK solves a joint past its range;
- the arm doesn't follow the command (more than 0.30 rad off on the real
  robot, 0.25 rad in sim).

It engages by ramping the command towards the IK. It compares the IK with the
command, never with the measured pose: on the real robot the driver sends no
gravity torque, and the arm sags 0.07-0.16 rad below its command.

It seeds the command from the measured pose, at startup and on each new
`start_epoch`, and stamps that epoch on its commands. `dora-openarm` drops a
command addressed to an earlier start.

### HUD (`dora-openarm-teleop-hud`)

Shows in the headset:

- `READY` / `REC mm:ss ep N`;
- each arm's clutch: `FOLLOW` (engaged), `HOLD`, `REGRIP` (locked: release
  and squeeze again), `LATE` (input gap), `SETUP` and `ENTER` (its
  [intro](#intro) or finger-down turn moves, or it waits for Enter in the UI);
- any arm the gate is holding, and why.

It also buzzes: right controller on start/success/fail, left on cancel, both on
quit.

## Start pose

Start Arm moves each pedestal arm to its driver config's start pose, and the
clutch holds the arm there until its grip is squeezed. `openarm_pedestal`'s
start pose has the elbow at 90° and every other joint at 0, so the wrist is
21.6 cm in front of `arm_origin`.

Ours holds each wrist 10 cm closer to the body, at the same height and with
the gripper turned the same way:

| | J1 | J4 | J6 | Other joints |
|---|---|---|---|---|
| Right | -0.465 | 1.928 | -0.109 | 0 |
| Left | 0.465 | 1.928 | 0.109 | 0 |

- The shoulder (J1) swings the upper arm back 26.6°, the elbow (J4) bends to
  110.5°, and J6 keeps the gripper level.
- The values come from the teleop IK, given the old start pose moved 10 cm
  back. Checked by forward kinematics: 10.0 cm back, 0.1 mm off in height,
  0.08° off in orientation.
- Both configs carry it:
  [`configs/openarm_pedestal_teleop.yaml`](../configs/openarm_pedestal_teleop.yaml)
  (right) and
  [`configs/openarm_pedestal_soft_gripper.yaml`](../configs/openarm_pedestal_soft_gripper.yaml)
  (left).
- The stop pose stays `openarm_pedestal`'s.
- The IK still starts from the model's `home` keyframe, the old start pose.
  Given the new pose as its target, it gets there in about 1 s, to within
  0.1°. The gate holds until then.
- The left intro rolls the gripper as before: ±90° about its fingers, within
  1 mm and 1°.

## Impedance

The pedestal robots (`ROBOT=pedestal`, `pedestal-zed`, `pedestal-viz`) run
their arms through `dora-openarm-impedance`, so that an arm can touch a wall
without pressing hard on it.

dora-openarm drives each arm joint with the motor's MIT command,
`kp·(q* − q) + kd·(0 − q̇) + tau`, with the driver config's kp and kd and
`tau = 0`. The springs alone then hold the arm up, and it sags by
gravity / kp: at the start pose J1 holds 4.3 Nm with kp 70, 61 mrad, which
is the sag the gate measured (Gate). So kp has to be stiff to stay near the
target, and a stiff spring pushes hard on whatever the arm touches.

`dora-openarm-impedance` runs dora-openarm unchanged, with one difference in
its driver (`openarm_driver.SingleArmDriver`):

- **Gravity feedforward.** Every MIT command carries the arm's gravity torque
  at the latest measured joints as `tau`. It is computed from the MJCF with
  MuJoCo's inverse dynamics at zero velocity
  ([`gravity.py`](../nodes/dora-openarm-teleop/src/dora_openarm_teleop/gravity.py)),
  9 µs per call. It is clipped per joint to 15 15 8 8 2 2 2 Nm.
- **Softer kp** for joints 1-4: 20 20 12 12 instead of 70 70 70 60. The
  wrist keeps the config's 10 10 10. kd, the gripper and everything else stay
  the config's: the safety checks, the start and stop moves (which get the
  feedforward too) and the measured state.
  - The wrist joints sit at most 0.12 m from the fingertip, so they barely
    change what a fingertip presses with. Softened to 4 they took at most
    0.4 N off a 2 cm push (6.3 N to 5.9 N), and let the gripper swing up to
    16° off its command in a fast move (J6, episode 9).
- **Velocity feedforward.** dora-openarm sends `dq = 0`, so the damping term
  brakes every motion and the arm trails a moving target by kd / kp: 39 ms
  for J1 at the config's gains, 138 ms at the soft ones. The node sends the
  commands' own joint velocity as dq instead: their change over the last
  20 ms, as they arrive, clipped to the config's joint velocity limits and
  zero after a 100 ms pause. Contact compliance is kp's, so it stays.
  - Recorded on the robot, the arm trailed its commands by 145-150 ms (J1)
    at the config's gains and 165-175 ms at the soft ones without it.
  - In a one-joint model of J1 (0.25 kg m², a 0.3 rad minimum-jerk move in
    0.6 s) the lag at mid-move is 51 ms at kp 70 with dq = 0, 157 ms at kp 20,
    and 49 ms at kp 20 with the feedforward. The model's 157 ms matches the
    robot; its 51 ms does not, so ~100 ms of the robot's lag at kp 70 comes
    from somewhere else than the motors' gains.
- **Latency log.** Every `LATENCY_LOG_S` (10 s) each arm node logs how late
  commands reach it from the gate (`metadata["timestamp"]`, which dora-echo
  passes on) and how long dora-openarm spends on each event. On the robot:
  0.2 ms and 0.2 ms (median; 0.3 ms and 0.3 ms p95). Nothing in the software
  waits.
- **Friction compensation.** That ~100 ms is friction. A joint slides only once
  its spring pulls harder than its friction, so it trails by friction / kp.
  While the joints slid with dq = 0 and gravity fed forward, the torque the
  PD put out along the motion was 0.74 0.49 0.54 0.98 0.10 0.19 0.16 Nm
  (J1-J7), the same at any speed: friction. At the soft kp that is 1.4-4.7°.
  The node adds 0.7x of it to tau along the commands' motion
  (`IMPEDANCE_FRICTION`), faded in by tanh(velocity / 0.05 rad/s), and only
  while the joint trails its command that way: in full from friction / kp
  behind, none once it has caught up. Pushing whenever the commands move
  carries the joint past the target once they stop.
  - In the one-joint J1 model with 0.74 Nm of friction (stiction included),
    a 29° move in 1 s: 4.3° worst error and 66 ms lag with the velocity
    feedforward alone, 3.0° and 41 ms with the compensation. Pushing whenever
    the commands move gets 2.8° and 40 ms, but stops 1.6° past the target
    rather than 0.26°.
  - At rest friction still holds a joint anywhere within friction / kp of its
    command; no feedforward changes that, only kp does.
  - On the robot's left-arm flip (the intro, 180° in 6 s): J5 trailed by
    240 ms and up to 8.6° with dq = 0, 60 ms and 4.8° with the velocity
    feedforward.

The model's gravity matches the robot. Over a 3 s recording of both real arms
moving, the measured joint torque follows the model's gravity torque with the
same sign on every joint (J1 0.87-0.92 × the model, r = 0.84). The wrist
joints, whose gravity barely changes, measure what the model says, e.g. left
J6 0.52 Nm against 0.46 Nm.

How hard a fingertip pressed 2 cm into a surface pushes back, from the model's
Jacobian (rigid links, no friction; the fingertip taken 0.12 m out from the IK's
control point along the fingers):

| Pose (right arm) | Direction | Config kp | Impedance kp, wrist 4 (before) | Impedance kp |
|---|---|---|---|---|
| Start pose (elbow 90°) | Forward, along the fingers | 28.2 N | 8.1 N | 8.1 N |
| | Sideways | 5.3 N | 1.3 N | 1.5 N |
| | Down | 4.1 N | 1.1 N | 1.2 N |
| Reaching forward (J1 1.0, J4 0.5 rad) | Forward | 61.6 N | 17.5 N | 17.8 N |
| | Along the fingers | 148.9 N | 42.5 N | 42.5 N |

Joint impedance softens the arm in most directions, but not along the line
from the shoulder to the fingertip with the arm stretched out. A push that
way loads the links, not the motors, so poking a wall head-on with a straight
arm stays stiff. It also does nothing about the impact itself, which depends
on how fast the arm arrives.

| Variable | Default | |
|---|---|---|
| `IMPEDANCE` | `on` | `off` runs plain dora-openarm, with the config's gains and no feedforward |
| `IMPEDANCE_KP` | `20,20,12,12,10,10,10` | kp for joints 1-7. Lower is softer, and follows less closely |
| `GRAVITY_SCALE` | `1.0` | Fraction of the model's gravity sent. Lower it if the arm drifts up |
| `IMPEDANCE_VELOCITY_FF` | `1.0` | Fraction of the commands' velocity sent as dq; `0` is dora-openarm's dq = 0 |
| `IMPEDANCE_FRICTION` | `0.52,0.34,0.38,0.69,0.07,0.13,0.11` | Nm per joint added along the motion while a joint trails; `0,0,0,0,0,0,0` for none |
| `LATENCY_LOG_S` | `10` | Seconds between latency log lines; `0` for none |

The left arm also uses
[`configs/openarm_pedestal_soft_gripper.yaml`](../configs/openarm_pedestal_soft_gripper.yaml):
`openarm_pedestal` with about an eighth of its grip force
(`gripper_posforce_limits` 0.12 instead of 1.0). The driver's conversion of that
value to a torque is marked unverified upstream, so it is tuned on the robot:
the closed fingers pressed on each other with 0.94 Nm at 0.5 and 0.52 Nm at
0.25, both still too strong.

## Timing

```
0.03 s <= POSE_TIMEOUT (0.09) <= clutch stale (0.1) < SMOOTHER_RESET_GAP (0.12)
clutch stale (0.1) < resume gap (0.5)
```

- If webxr kept sending a pose after the clutch has dropped it, a stale hand
  would come back as fresh.
- If webxr's smoother didn't reset after a gap the clutch noticed, the first
  pose back would be smoothed towards the old one, and the arm would slide on
  resume.
- Why 0.09 s and not 0.1 s: webxr times frame arrivals and the clutch times
  grip deliveries, a few ms of dora jitter apart. In openarm-quest-teleop's
  measured headset sessions, with both at 0.1 s, 73 of 793 stalls re-engaged
  on a 66-70 mm slide; at 0.09 s none of 827 did.

## Checks

[`scripts/dora-envsubst`](../scripts/dora-envsubst) runs the same check as
[`scripts/check-dataflow`](../scripts/check-dataflow) before `run`, `start` and
`build`, and refuses to start a dataflow whose teleop nodes disagree on:

- the timing chain above, and `PIVOT_MODE=latched`;
- the tick rates of the leader, the clutch and the IK;
- the IK's input queue size (≥ 2, so a late IK doesn't pair this tick's right
  with last tick's left);
- the gate and the IK's profile, ranges, margin and velocity scale;
- whether the gate has the robot's state.

`SKIP_TELEOP_CHECK=1` skips it.

## Measured

`nodes/dora-openarm-teleop/tests/e2e` (`uv run pytest -m e2e -s`, about 15 s):

- **Setup.** Scripted hands drive the real clutch, IK, gate, HUD and
  dora-openarm-mujoco under dora. The right hand moves 0.10 m and turns 45° in
  0.5 s (0.2 m/s, 90°/s).
- **Measured.** The gate's command, through forward kinematics, against the
  clutch target. The figures below are from one run on a DGX Spark.

| IK arguments | Worst position error | Worst angle error | Settles after the hand stops |
|---|---|---|---|
| this module | 1.9 mm | 0.8° | 6 ms |
| plus the fork's error limits | 1.2 mm | 23.3° | 525 ms |

The test fails if the error limits stop making a difference, and if this
module's tracking gets worse than 5 mm, 3° or 0.1 s.

## Headset setup

```bash
scripts/webxr-setup          # once: tls/ with a local CA, the certificate and the pairing token
```

It prints the URLs to open in the Quest browser, `https://<host>:8443/#k=<token>`,
and the certificates' SHA-256 fingerprints. The Quest browser warns about the
certificate the first time; proceed past the warning. The token is the only thing that keeps others on the
network from driving the robot: don't share the URL.

- `--rotate-token` makes every saved URL stop working.
- `--force` makes new certificates, e.g. after the host's IP changed.

Over USB, `scripts/webxr-usb/start.sh` reverses ports 8443 and 3479 with adb
and runs a TURN server on 3479 (`sudo apt install coturn`). The page opened as
`https://localhost:8443/#k=<token>` relays through it on its own, by patch 18
in the webxr node. **This path isn't tested with a headset yet.**

## What was taken from the noguchi fork

From [guchio123/dora-openarm-data-collection](https://github.com/guchio123/dora-openarm-data-collection):

**Kept:**

- the Japanese README's operating procedure;
- `metadata.yaml` with tasks and `TASK`;
- the UI and recorder changes;
- the pedestal/ZED robot modules and `view-episode.py`;
- the followers' `--align-trigger gripper`. After each Start Arm, pull each
  trigger once: the follower then accepts the gate's commands. The gate starts
  them at the arm, so they align at once.
- `start_epoch`, now set by the gate from the follower's own state;
- the USB relay.

**Not taken:**

- **The IK error limits** (see [IK](#ik-dora-openarm-teleop-ik)).
- **`pose.mode: relative` in `view-webxr.yaml`.** Re-expressing the hands in
  the headset frame every frame moves an engaged arm whenever the head turns.
  The clutch plus a latched pivot gives the same "move from where my hand is"
  without that.
