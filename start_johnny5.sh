#!/bin/bash
# Johnny 5 Demo Launcher
# Starts face recognition (via systemd) + voice conversation

echo "Starting Johnny 5 Demo..."

# Kill any existing johnny5 process
pkill -f johnny5.py 2>/dev/null
sleep 1

# Setup proper echo cancellation for duplex audio (mic + speaker simultaneously)
echo "Setting up WebRTC echo cancellation..."

# Unload any existing echo-cancel module
pactl unload-module module-echo-cancel 2>/dev/null || true

# Load WebRTC-based echo cancellation with tuned settings
pactl load-module module-echo-cancel \
    use_master_format=1 \
    aec_method=webrtc \
    aec_args="analog_gain_control=0 digital_gain_control=1 noise_suppression=1 extended_filter=1 voice_detection=1 high_pass_filter=1" \
    source_name=ec_source \
    sink_name=ec_sink 2>/dev/null || true

# Set echo-cancelled source as default
pactl set-default-source ec_source 2>/dev/null || true

# ReSpeaker 3.5mm Output with HDMI Fallback
RESPEAKER_SINK=$(pactl list sinks short | grep -i respeaker | awk '{print $2}')
if [ -n "$RESPEAKER_SINK" ]; then
    echo "Using ReSpeaker 3.5mm output (hardware AEC)"
    pactl set-default-sink "$RESPEAKER_SINK"
else
    echo "ReSpeaker output not found, using HDMI fallback"
fi

# Verify
echo "Audio setup:"
pactl get-default-source
pactl get-default-sink

# Restart face recognition via systemd
echo "Restarting face recognition service..."
sudo systemctl restart whoami.service
sleep 2

# Check if it started
if systemctl is-active --quiet whoami.service; then
    echo "Face recognition running"
else
    echo "WARNING: Face recognition failed to start"
    journalctl -u whoami.service -n 5 --no-pager
fi

# Start voice in foreground
cd /home/robbie
echo "Starting voice..."
python3 johnny5.py
