#!/bin/bash
# Johnny 5 Demo Launcher
# Starts face recognition + voice conversation

echo "Starting Johnny 5 Demo..."

# Kill any existing processes
killall python3 2>/dev/null
sleep 1

# Setup proper echo cancellation for duplex audio (mic + speaker simultaneously)
echo "Setting up WebRTC echo cancellation..."

# Unload any existing echo-cancel module
pactl unload-module module-echo-cancel 2>/dev/null || true

# Load WebRTC-based echo cancellation
# source_master = ReSpeaker mic, sink_master = HDMI output
pactl load-module module-echo-cancel \
    use_master_format=1 \
    aec_method=webrtc \
    aec_args="analog_gain_control=0 digital_gain_control=1 noise_suppression=1" \
    source_name=ec_source \
    sink_name=ec_sink 2>/dev/null || true

# Set echo-cancelled source as default (so apps use it)
pactl set-default-source ec_source 2>/dev/null || true

# Verify
echo "Audio setup:"
pactl get-default-source
pactl get-default-sink

# Start face recognition in background
cd /home/robbie
DISPLAY=:0 python3 whoami_full.py &
FACE_PID=$!
echo "Face recognition started (PID: $FACE_PID)"

# Wait for it to initialize
sleep 3

# Start voice in foreground (so we can see output)
echo "Starting voice..."
python3 johnny5.py

# Cleanup
kill $FACE_PID 2>/dev/null
