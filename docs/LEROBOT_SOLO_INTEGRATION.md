# LeRobot / Solo-CLI Integration

This document describes what LeRobot already supports and what Johnny Five adds.

## What LeRobot Already Supports

### Robot Types in LeRobot

| Robot | Motors | Subsystems |
|-------|--------|------------|
| **so_follower** | 6 | Single arm (SO-101) |
| **bi_so_follower** | 12 | Dual arms |
| **lekiwi** | 9 | Arm (6) + Omniwheel base (3) |
| **koch_follower** | 6 | Single arm |
| **openarm_follower** | 6+ | Open-source arm |
| **reachy2** | varies | Pollen Robotics humanoid |
| **earthrover_mini_plus** | varies | Mobile rover |
| **hope_jr** | varies | Humanoid |

### LeKiwi Motor Layout (Reference)

LeKiwi is the closest to Johnny Five - it has arms + mobile base.

```
LeKiwi Motors (9 total on one bus):
├── Arm (motors 1-6)
│   ├── 1: arm_shoulder_pan
│   ├── 2: arm_shoulder_lift
│   ├── 3: arm_elbow_flex
│   ├── 4: arm_wrist_flex
│   ├── 5: arm_wrist_roll
│   └── 6: arm_gripper
│
└── Base (motors 7-9) - Omniwheels, VELOCITY mode
    ├── 7: base_left_wheel
    ├── 8: base_back_wheel
    └── 9: base_right_wheel
```

### Motor Types Supported

- **Feetech STS3215** (7.4V) - Used for arms
- **Feetech STS3250** - Higher torque variant
- **Feetech SM8512BL** - Brushless
- **Feetech SCS0009** - Small servos

### Control Modes

- Position mode (arms, joints)
- Velocity mode (wheels)
- Range normalization (0-100, -100 to 100, degrees)

---

## What Johnny Five Adds

Johnny Five uses **dual buses** (ACM0 + ACM1) and adds subsystems LeRobot doesn't have.

### Johnny Five Motor Layout (19 total)

```
/dev/ttyACM0 (Left Bus) - 11 motors:
├── Left Arm (motors 1-6) - Same as LeKiwi
│   ├── 1: left_shoulder_pan
│   ├── 2: left_shoulder_lift
│   ├── 3: left_elbow_flex
│   ├── 4: left_wrist_flex
│   ├── 5: left_wrist_roll
│   └── 6: left_gripper
│
├── Wheels (motors 7-9) - Mecanum (different from omni)
│   ├── 7: wheel_front_left
│   ├── 8: wheel_front_right
│   └── 9: wheel_back
│
├── Lift (motor 10) - NEW
│   └── 10: lift_vertical
│
└── Hitch (motor 11) - NEW
    └── 11: hitch_grabber

/dev/ttyACM1 (Right Bus) - 8 motors:
├── Right Arm (motors 1-6) - Same as LeKiwi
│   ├── 1: right_shoulder_pan
│   ├── 2: right_shoulder_lift
│   ├── 3: right_elbow_flex
│   ├── 4: right_wrist_flex
│   ├── 5: right_wrist_roll
│   └── 6: right_gripper
│
└── Gantry (motors 7-8) - NEW (camera pan/tilt)
    ├── 7: gantry_pan
    └── 8: gantry_tilt
```

### New Subsystems for Upstream

These should be contributed back to LeRobot:

#### 1. Gantry (Camera Pan/Tilt)

```python
# Proposed addition to LeRobot
GANTRY_MOTORS = {
    "gantry_pan": {"id": 7, "mode": "position", "range": [-90, 90]},
    "gantry_tilt": {"id": 8, "mode": "position", "range": [-45, 45]},
}
```

Use case: Head tracking, looking at speakers, visual attention.

#### 2. Lift (Vertical Height Adjustment)

```python
LIFT_MOTORS = {
    "lift_vertical": {"id": 10, "mode": "position", "range": [0, 300]},  # mm
}
```

Use case: Adjusting height to match human eye level, reaching high/low shelves.

#### 3. Hitch (Rear Grabber)

```python
HITCH_MOTORS = {
    "hitch_grabber": {"id": 11, "mode": "position", "range": [0, 100]},  # 0=open, 100=closed
}
```

Use case: Towing carts, grabbing charger cables, docking.

#### 4. Mecanum Wheels (Holonomic Drive)

LeKiwi uses omniwheels with 3-wheel configuration. Johnny Five uses 3 mecanum wheels.

```python
# Different kinematics from omniwheel
MECANUM_BASE = {
    "wheel_front_left": {"id": 7, "mode": "velocity"},
    "wheel_front_right": {"id": 8, "mode": "velocity"},
    "wheel_back": {"id": 9, "mode": "velocity"},
}
```

#### 5. Dual Bus Support

LeRobot assumes single bus. Johnny Five needs:

```python
BUSES = {
    "left": "/dev/ttyACM0",   # Left arm + wheels + lift + hitch
    "right": "/dev/ttyACM1",  # Right arm + gantry
}
```

---

## Proposed LeRobot Config: johnny5_follower

```python
# src/lerobot/robots/johnny5_follower/config_johnny5_follower.py

from lerobot.robots.config import RobotConfig

class Johnny5FollowerConfig(RobotConfig):
    robot_type = "johnny5_follower"

    buses = {
        "left": {"port": "/dev/ttyACM0", "motors": list(range(1, 12))},
        "right": {"port": "/dev/ttyACM1", "motors": list(range(1, 9))},
    }

    # Arms (same as bi_so_follower)
    left_arm = ["left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
                "left_wrist_flex", "left_wrist_roll", "left_gripper"]
    right_arm = ["right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
                 "right_wrist_flex", "right_wrist_roll", "right_gripper"]

    # Base (mecanum)
    base = ["wheel_front_left", "wheel_front_right", "wheel_back"]

    # New subsystems
    gantry = ["gantry_pan", "gantry_tilt"]
    lift = ["lift_vertical"]
    hitch = ["hitch_grabber"]
```

---

## CLI Commands Needed

### Existing (works)

```bash
# Calibrate arm
lerobot-calibrate --robot.type=johnny5_follower --robot.port=/dev/ttyACM0

# Find ports
lerobot-find-port

# Setup motors
lerobot-setup-motors --robot.type=johnny5_follower
```

### New Commands (to add)

```bash
# Gantry control
solo robo --port /dev/ttyACM1 --ids 7,8 --positions 0,0  # center
solo robo --port /dev/ttyACM1 --ids 7,8 --positions 45,10  # look right/up

# Lift control
solo robo --port /dev/ttyACM0 --id 10 --position 150  # mid height

# Hitch control
solo robo --port /dev/ttyACM0 --id 11 --position 100  # close
solo robo --port /dev/ttyACM0 --id 11 --position 0    # open

# Full status
solo status --ports /dev/ttyACM0,/dev/ttyACM1
```

---

## Upstream Contribution Plan

1. **Fork lerobot** → johnny5-lerobot
2. **Add johnny5_follower** robot type with:
   - Dual bus support
   - Gantry subsystem
   - Lift subsystem
   - Hitch subsystem
   - Mecanum kinematics
3. **Test on hardware**
4. **Submit PR** to huggingface/lerobot

The goal is to make these additions generic enough that other robots can use them:
- Gantry → Any robot with a camera on pan/tilt
- Lift → Any robot with vertical adjustment
- Hitch → Any robot that needs to grab/tow things
- Dual bus → Any robot with motors split across buses

---

## References

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- [LeKiwi Docs](https://huggingface.co/docs/lerobot/main/en/lekiwi)
- [Feetech Motor SDK](https://github.com/huggingface/lerobot/blob/main/lerobot/common/robot_devices/motors/feetech.py)
