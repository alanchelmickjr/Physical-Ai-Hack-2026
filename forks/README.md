# Forked Repositories

This directory contains forked repositories with Johnny Five-specific additions.
Both repos have custom code not in upstream.

## Contents

### johnny5-lerobot
**Forked from:** https://github.com/liyiteng/lerobot_alohamini
**Our fork:** https://github.com/alanchelmickjr/johnny5-lerobot

LeRobot variant adapted for Johnny Five. Custom additions:
- `src/lerobot/robots/johnny5/` - Johnny Five robot configuration
- Custom motor IDs matching our 19-servo layout
- Dual-bus support (ACM0 + ACM1)
- Gantry control integration

### johnny5-solo
**Forked from:** https://github.com/TheRobotStudio/SO-ARM100
**Our fork:** https://github.com/alanchelmickjr/johnny5-solo

Solo-CLI fork with Johnny Five extensions:
- Multi-bus support for split arm configuration
- Gantry pan/tilt commands
- Lift and hitch motor control
- Wheel velocity mode for mecanum drive
- Calibration profiles for Johnny Five hardware

## Setup

Clone our forks (not upstream):

```bash
cd forks

# Johnny Five LeRobot fork
git clone https://github.com/alanchelmickjr/johnny5-lerobot.git

# Johnny Five Solo-CLI fork
git clone https://github.com/alanchelmickjr/johnny5-solo.git
```

## Johnny Five Hardware Layout

```
/dev/ttyACM0 (Left Bus):
  - Motors 1-6: Left arm (SO-ARM100)
  - Motors 7-9: Mecanum wheels
  - Motor 10: Lift
  - Motor 11: Hitch

/dev/ttyACM1 (Right Bus):
  - Motors 1-6: Right arm (SO-ARM100)
  - Motors 7-8: Gantry (pan/tilt)
```

## Installing Solo-CLI

```bash
# From our fork (recommended):
cd johnny5-solo
pip install -e . --break-system-packages

# Or upstream (missing Johnny Five features):
pip install solo-cli --break-system-packages
```

## Installing LeRobot

```bash
cd johnny5-lerobot
pip install -e ".[dev]" --break-system-packages
```

## Keeping Forks Updated

```bash
# Add upstream remotes
cd johnny5-lerobot
git remote add upstream https://github.com/liyiteng/lerobot_alohamini.git

cd ../johnny5-solo
git remote add upstream https://github.com/TheRobotStudio/SO-ARM100.git

# Pull upstream changes
git fetch upstream
git merge upstream/main
```
