# Johnny Five Hardware Specification

## Motor Count: 19 Servos Total

| Bus | Subsystem | Motor IDs | Count | Servo Type |
|-----|-----------|-----------|-------|------------|
| ACM0 | Left Arm | 1-6 | 6 | XL330-M288-T |
| ACM0 | Omni Base | 7, 8, 9 | 3 | XL330-M288-T |
| ACM0 | Lift | 10 | 1 | XL330-M288-T |
| ACM0 | Hitch | 11 | 1 | XL330-M288-T |
| ACM1 | Right Arm | 1-6 | 6 | XL330-M288-T |
| ACM1 | Gantry | 7, 8 | 2 | XL330-M288-T |
| **Total** | | | **19** | |

## Bus Layout

```
/dev/ttyACM0 (Left Bus - Waveshare on back)
├── Left SO101 Arm
│   ├── Motor 1: Base rotation
│   ├── Motor 2: Shoulder pitch
│   ├── Motor 3: Elbow pitch
│   ├── Motor 4: Wrist pitch
│   ├── Motor 5: Wrist roll
│   └── Motor 6: Gripper
├── Omni Base
│   ├── Motor 7: Front wheel
│   ├── Motor 8: Back-left wheel
│   └── Motor 9: Back-right wheel
├── Lift
│   └── Motor 10: Vertical lift (300mm travel)
└── Hitch (Rear Grabber)
    └── Motor 11: IKEA cart tow / charger dock

/dev/ttyACM1 (Right Bus - Waveshare on back)
├── Right SO101 Arm
│   ├── Motor 1: Base rotation
│   ├── Motor 2: Shoulder pitch
│   ├── Motor 3: Elbow pitch
│   ├── Motor 4: Wrist pitch
│   ├── Motor 5: Wrist roll
│   └── Motor 6: Gripper
└── Camera Gantry
    ├── Motor 7: Pan (horizontal)
    └── Motor 8: Tilt (vertical)
```

## SO101 Arm Joint Specifications

| Joint | Motor | Range | Home | Description |
|-------|-------|-------|------|-------------|
| Base | 1 | -150° to +150° | 0° | Rotation at shoulder |
| Shoulder | 2 | -90° to +90° | -45° | Pitch up/down |
| Elbow | 3 | 0° to 180° | 90° | Elbow bend |
| Wrist Pitch | 4 | -90° to +90° | 45° | Wrist up/down |
| Wrist Roll | 5 | -180° to +180° | 0° | Wrist rotation |
| Gripper | 6 | 0° to 80° | 0° | Open/close |

## Gantry Specifications

| Joint | Motor | Range | Home | Description |
|-------|-------|-------|------|-------------|
| Pan | 7 | -90° to +90° | 0° | Horizontal rotation |
| Tilt | 8 | -45° to +45° | 0° | Vertical angle |

## Lift Specifications

| Parameter | Value |
|-----------|-------|
| Motor ID | 10 |
| Travel | 0-300mm |
| Home Position | 150mm (center) |
| Speed | 50mm/s max |

## Hitch Specifications (Rear Grabber)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Motor ID | 11 | On ACM0 bus |
| Range | 0° to 80° | Open to closed |
| Home (Open) | 0° | Released |
| Closed | 80° | Grabbed/docked |
| Grip Force | ~500g | Similar to arm grippers |

**Embedded Charging Contacts:**
The hitch gripper has embedded charging contacts on the inner grip surfaces:
- **Docking Mode**: Contacts engage with charger dock connector → robot charges
- **Tow Mode**: IKEA cart handle never touches contacts → mechanical grip only, no electrical connection

**Use Cases:**
- **IKEA Cart Tow**: Mechanical grab of cart handle, no electrical contact
- **Charger Dock**: Grip + electrical contact for autonomous charging
- **Rear Object Grab**: Pick up items behind the robot

**Why this matters:** A small XL330 servo lets a compact robot tow a full-size IKEA utility cart, demonstrating that you don't need a massive expensive robot to move heavy loads.

## Base Specifications (Omni)

| Wheel | Motor | Direction |
|-------|-------|-----------|
| Front | 7 | Forward = positive |
| Back-Left | 8 | Forward = positive |
| Back-Right | 9 | Forward = positive |

Movement vectors:
- Forward: [+, +, +]
- Backward: [-, -, -]
- Strafe Left: [-, +, -]
- Strafe Right: [+, -, +]
- Rotate CW: [+, -, +]
- Rotate CCW: [-, +, -]

## Calibration Sequence

### 1. Pre-Flight Check
```
solo scan --port /dev/ttyACM0  # Should find IDs 1-11
solo scan --port /dev/ttyACM1  # Should find IDs 1-8
```

### 2. Arm Calibration (per arm)
```bash
# 1. Disable torque (limp mode)
solo robo --port $PORT --ids 1,2,3,4,5,6 --torque off

# 2. Manually move to home position (human does this)

# 3. Set current position as home offset
solo robo --port $PORT --ids 1,2,3,4,5,6 --set-home

# 4. Enable torque
solo robo --port $PORT --ids 1,2,3,4,5,6 --torque on

# 5. Test: move to home
solo robo --port $PORT --ids 1,2,3,4,5,6 --positions 0,-45,90,45,0,0
```

### 3. Gantry Calibration
```bash
# 1. Disable torque
solo robo --port /dev/ttyACM1 --ids 7,8 --torque off

# 2. Center manually

# 3. Set home
solo robo --port /dev/ttyACM1 --ids 7,8 --set-home

# 4. Test range
solo robo --port /dev/ttyACM1 --ids 7,8 --positions -90,0 --speed 0.3
solo robo --port /dev/ttyACM1 --ids 7,8 --positions 90,0 --speed 0.3
solo robo --port /dev/ttyACM1 --ids 7,8 --positions 0,0 --speed 0.3
```

### 4. Lift Calibration
```bash
# 1. Find bottom limit
solo robo --port /dev/ttyACM0 --ids 10 --find-limit down

# 2. Find top limit
solo robo --port /dev/ttyACM0 --ids 10 --find-limit up

# 3. Go to center
solo robo --port /dev/ttyACM0 --ids 10 --positions 150
```

### 5. Base Calibration
```bash
# Test each wheel individually
solo robo --port /dev/ttyACM0 --ids 7 --velocity 50 --duration 1
solo robo --port /dev/ttyACM0 --ids 8 --velocity 50 --duration 1
solo robo --port /dev/ttyACM0 --ids 9 --velocity 50 --duration 1

# Test forward
solo robo --port /dev/ttyACM0 --ids 7,8,9 --velocities 50,50,50 --duration 1

# Test rotation
solo robo --port /dev/ttyACM0 --ids 7,8,9 --velocities 50,-50,50 --duration 1
```

### 6. Hitch Calibration (Rear Grabber)
```bash
# 1. Disable torque
solo robo --port /dev/ttyACM0 --ids 11 --torque off

# 2. Manually open fully (0° position)

# 3. Set home
solo robo --port /dev/ttyACM0 --ids 11 --set-home

# 4. Enable torque
solo robo --port /dev/ttyACM0 --ids 11 --torque on

# 5. Test open/close
solo robo --port /dev/ttyACM0 --ids 11 --positions 0 --speed 0.3   # Open
solo robo --port /dev/ttyACM0 --ids 11 --positions 80 --speed 0.3  # Closed
solo robo --port /dev/ttyACM0 --ids 11 --positions 0 --speed 0.3   # Open
```

## Diagnostic Commands

### Scan for Motors
```bash
solo scan --port /dev/ttyACM0 --baud 1000000
solo scan --port /dev/ttyACM1 --baud 1000000
```

### Read Motor Status
```bash
solo robo --port $PORT --ids $ID --status
# Returns: position, velocity, temperature, load, voltage
```

### Ping Motor
```bash
solo robo --port $PORT --ids $ID --ping
```

### Set Motor ID
```bash
# WARNING: Only one motor on bus when changing ID
solo robo --port $PORT --ids $OLD_ID --set-id $NEW_ID
```

### Reboot Motor
```bash
solo robo --port $PORT --ids $ID --reboot
```

## Motor ID Layouts Comparison

Different builds use different motor IDs. Johnny5 uses clean sequential numbering:

| Subsystem | Johnny5 | LeKiwi | Why Johnny5 is cleaner |
|-----------|---------|--------|------------------------|
| Left Arm | 1-6 | 1-6 | Same |
| Right Arm | 1-6 | 1-6 | Same |
| Base Wheels | 7, 8, 9 | 8, 9, 10 | Sequential after arm |
| Lift | 10 | 11 | Sequential after base |
| Hitch | 11 | N/A | Johnny5 only |
| Gantry | 7, 8 (ACM1) | N/A | Johnny5 only |

**Johnny5 ID Logic:**
```
ACM0: [Arm 1-6] [Base 7-9] [Lift 10] [Hitch 11]
ACM1: [Arm 1-6] [Gantry 7-8]
```

This makes it easy to remember:
- Arms always 1-6
- Everything else starts at 7 and goes up
- Each bus handles one arm plus accessories

The verbal calibration system can detect and adapt to any layout variation.

## Safety Limits

| Parameter | Value |
|-----------|-------|
| Max arm joint velocity | 100°/s |
| Max gripper force | 500g |
| Max base velocity | 0.5 m/s |
| Max lift velocity | 50mm/s |
| Emergency stop latency | <50ms |
| Torque limit (collision) | 80% max |
