# Forked Repositories

This directory contains forked/cloned repositories for Johnny Five motor control.

## Contents

### johnny5-lerobot
Forked from: https://github.com/liyiteng/lerobot_alohamini

LeRobot variant adapted for AlohaMini-style robots. We've added:
- `src/lerobot/robots/johnny5/` - Johnny Five robot configuration
- Custom motor IDs matching our hardware layout

### johnny5-solo
Cloned from: https://github.com/TheRobotStudio/SO-ARM100

SO-ARM100 / Solo-CLI for controlling Dynamixel-based arms.

## Setup

These repos are not committed to the main repo (they're in .gitignore).
Clone them manually:

```bash
cd forks

# LeRobot with AlohaMini support
git clone https://github.com/liyiteng/lerobot_alohamini.git johnny5-lerobot

# SO-ARM100 / Solo-CLI
git clone https://github.com/TheRobotStudio/SO-ARM100.git johnny5-solo
```

## Johnny Five Hardware Layout

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
cd johnny5-solo
pip install -e . --break-system-packages
```

## Installing LeRobot

```bash
cd johnny5-lerobot
pip install -e ".[dev]" --break-system-packages
```
