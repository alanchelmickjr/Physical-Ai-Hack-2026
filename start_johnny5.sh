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

# Load WebRTC-based echo cancellation with tuned settings
# source_master = ReSpeaker mic, sink_master = HDMI output
# Extended filter and voice detection help with longer echo tails
pactl load-module module-echo-cancel \
    use_master_format=1 \
    aec_method=webrtc \
    aec_args="analog_gain_control=0 digital_gain_control=1 noise_suppression=1 extended_filter=1 voice_detection=1 high_pass_filter=1" \
    source_name=ec_source \
    sink_name=ec_sink 2>/dev/null || true

# Set echo-cancelled source as default (so apps use it)
pactl set-default-source ec_source 2>/dev/null || true

# =============================================================================
# ReSpeaker 3.5mm Output (Primary) with HDMI Fallback
# =============================================================================
# ReSpeaker's XMOS DSP provides hardware AEC when speaker is on 3.5mm jack.
# Falls back to HDMI if ReSpeaker output not detected.

RESPEAKER_SINK=$(pactl list sinks short | grep -i respeaker | awk '{print $2}')
if [ -n "$RESPEAKER_SINK" ]; then
    echo "Using ReSpeaker 3.5mm output (hardware AEC)"
    pactl set-default-sink "$RESPEAKER_SINK"
else
    echo "ReSpeaker output not found, using HDMI fallback"
    # HDMI will be default, software AEC handles echo
fi
# =============================================================================

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
