# Forked Repositories

This directory contains forked/cloned repositories for Chloe's motor control.

## Contents

### chloe-lerobot
Forked from: https://github.com/liyiteng/lerobot_alohamini

LeRobot variant adapted for AlohaMini-style robots. We've added:
- `src/lerobot/robots/chloe/` - Chloe-specific robot configuration
- Custom motor IDs matching our hardware layout

### chloe-solo
Cloned from: https://github.com/TheRobotStudio/SO-ARM100

SO-ARM100 / Solo-CLI for controlling Dynamixel-based arms.

## Setup

These repos are not committed to the main repo (they're in .gitignore).
Clone them manually:

```bash
cd forks

# LeRobot with AlohaMini support
git clone https://github.com/liyiteng/lerobot_alohamini.git chloe-lerobot

# SO-ARM100 / Solo-CLI
git clone https://github.com/TheRobotStudio/SO-ARM100.git chloe-solo
```

## Chloe Hardware Layout

```
/dev/ttyACM0 (Left Bus):
  - Motors 1-6: Left arm
  - Motors 7-9: Mecanum wheels
  - Motor 10: Lift

/dev/ttyACM1 (Right Bus):
  - Motors 1-6: Right arm
  - Motors 7-8: Gantry (pan/tilt)
```

## Installing Solo-CLI

```bash
pip install solo-cli --break-system-packages

# Or from local clone:
cd chloe-solo
pip install -e . --break-system-packages
```

## Installing LeRobot

```bash
cd chloe-lerobot
pip install -e ".[dev]" --break-system-packages
```
