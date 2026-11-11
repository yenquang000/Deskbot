# Deskbot – learning to drive (Yen's part, plus Nico's sim physics and Ben's Pi runner)

## Run order
```bash
pip install gymnasium stable-baselines3 torch matplotlib numpy
python test_env.py                       # 8 sanity tests, ~2 s
python evaluate.py --episodes 200        # baselines only, no model needed
# Day 2 overnight  (coverage only)
python train.py --stage 1 --steps 1500000 --out models/stage1
# Day 4 overnight  (debris)          Day 7 overnight (pens + all randomisation)
python train.py --stage 2 --steps 2000000 --load models/stage1 --out models/stage2
python train.py --stage 3 --steps 3000000 --load models/stage2 --out models/final
# Day 8
python evaluate.py --episodes 200 --model models/final.zip     # prints PASS / NOT YET
python export_to_pi.py models/final.zip deskbot_policy.npz     # verifies vs SB3, then ships
# Day 9+  (the Pi needs the whole repo; only NumPy from this folder's requirements)
python pi_runner.py --sim --policy deskbot_policy.npz          # loop check without hardware
python test_vision_bridge.py                                   # vision -> policy -> Pico commands, on the sim
cd ../vision && python main.py --drive-check                   # on the robot: test each move
python main.py --drive ../sim/deskbot_policy.npz               # on the robot: run the policy
python make_chart.py                                           # Day 13 chart -> comparison.png
```

## Files
| file | what |
|---|---|
| `obs.py` | the 26-number observation + cleaned-area grid; **shared by sim and Pi** |
| `env.py` | simulation (stages 1-3, per-episode domain randomisation, `Measurements` for Nico's numbers) |
| `baseline.py` | backup lawnmower pattern + `EdgeShield` (blocks any move that would cross an edge) |
| `train.py` / `evaluate.py` | staged PPO / 200-episode comparison on identical seeds |
| `export_to_pi.py` / `pi_inference.py` | SB3 → `.npz` → NumPy-only policy |
| `pi_runner.py` | Pi control loop `run()`; `RobotHardware` is the robot side, `SimHardware` the sim-backed one |
| `vision_hardware.py` | `VisionHardware`: the real `RobotHardware`. Camera pose and objects from `vision/`, moves through the Pico. Also `movement_check()` |
| `test_vision_bridge.py` | end-to-end test of the above with a fake link backed by the simulator |

## Driving the real robot
The loop runs as a thread inside the vision program (`vision/main.py --drive`, see the
main README for the flags and bring-up order). Nothing here is retrained or changed for
it: `env.py`, `train.py` and `deskbot_policy.npz` are as they were.

- **Pose** comes from the camera before every step (`ObsBuilder.set_pose`), not from dead
  reckoning; the cleaned map is marked at the measured pose after every move.
- **Units**: vision gives mm and degrees, `VisionHardware` converts to cm and radians.
  The axes already agree (heading 0 = +x, positive rotation from +x toward +y).
- **Walls**: `obs.travel_room()` (same maths as `DeskbotEnv._clear`) from the camera pose
  and the marker-to-marker desk size.
- **Pens vs debris**: by box shape and fill ratio, constants at the top of
  `vision_hardware.py`. A pen within 15 cm of the pen corner counts as delivered and is
  ignored, as in `env.step`.
- **Cleaning**: assumed to run while driving. `run(clean_on_arrival=True)` also sends a
  `clean` command when a move ends within `robot_radius` of the debris seen before it.
- **`run()` result** now has `reason`: `goal reached`, `max steps`, `bot lost` or `stopped`.
- **Desk size**: a warning is logged if the vision desk is more than 5% off
  `Measurements` (the training randomisation range).

## What changed from the first env.py (and why)
- Randomisation ran once at construction → now every episode (desk size, grip, turn error, sensor noise, camera quality).
- Wall "sensors" measured the wrong thing for angled headings, and could not tell front/back/left/right → exact travel-room along each direction.
- Debris and pens were only ever the list's first item, not the nearest; pens were pushed on turns and reversing → nearest piece, pushed only when driving forward into it.
- Reward ±5000/2000 dominated everything (bad for PPO) → scaled to roughly -30…+150 per episode.
- Reward was only "get closer"; nothing rewarded cleaning → coverage grid (5 cm cells), +0.2 per new cell.
- Observation noise reused one random draw for debris and pen → independent noise, camera misses, phantom detections.
- Shaping is potential-based (can't be farmed by oscillating).

## Assumptions to confirm with Nico / the team
1. `Measurements` defaults are the numbers from the first env.py (desk 120.65 × 59.69 cm, 5 cm step, 10 cm radius, 45° turns). Replace with the real ones.
2. `clean_radius` = robot radius; `coverage_goal` = 0.80. With an 8 cm edge margin the backup pattern reaches ~81%, so 85% is not reachable by it.
3. Robot is placed by hand at a multiple of 45°; 45° turns cannot align to an arbitrary heading.
4. Wall "sensors" report travel room in cm. On the robot there are no range sensors: `VisionHardware.wall_room_cm()` computes it from the camera pose.
5. The bot's ArUco marker is on the robot's centre, and the desk edges are the lines between corner marker centres.
