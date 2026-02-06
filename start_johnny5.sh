#!/bin/bash
# Johnny 5 Startup Script
# Uses PulseAudio WebRTC software AEC with ReSpeaker

set -e
cd "$(dirname "$0")"

echo "=== Johnny 5 Starting ==="
echo "--- Audio Setup ---"

# Clean up old modules
pactl unload-module module-echo-cancel 2>/dev/null || true
pactl unload-module module-remap-source 2>/dev/null || true

# Find ReSpeaker devices
RESPEAKER_INPUT=$(pactl list sources short | grep -i respeaker | grep input | awk '{print $2}')
RESPEAKER_SINK=$(pactl list sinks short | grep -i respeaker | awk '{print $2}')

if [ -n "$RESPEAKER_INPUT" ] && [ -n "$RESPEAKER_SINK" ]; then
    echo "ReSpeaker input: $RESPEAKER_INPUT"
    echo "ReSpeaker output: $RESPEAKER_SINK"

    # Set ReSpeaker as defaults FIRST
    pactl set-default-source "$RESPEAKER_INPUT"
    pactl set-default-sink "$RESPEAKER_SINK"

    # WebRTC software AEC - uses defaults (like Saturday)
    if pactl load-module module-echo-cancel \
        use_master_format=1 \
        aec_method=webrtc \
        aec_args="analog_gain_control=0 digital_gain_control=1 noise_suppression=1" \
        source_name=ec_source \
        sink_name=ec_sink; then
        # Now set AEC sources as defaults
        pactl set-default-source ec_source
        pactl set-default-sink ec_sink
        echo "Source: ec_source (WebRTC AEC)"
        echo "Sink: ec_sink (WebRTC AEC)"
    else
        echo "WARNING: AEC module failed to load, using raw ReSpeaker input"
    fi
else
    echo "WARNING: ReSpeaker not found"
    if [ -z "$RESPEAKER_INPUT" ]; then echo "  - No ReSpeaker input"; fi
    if [ -z "$RESPEAKER_SINK" ]; then echo "  - No ReSpeaker sink"; fi
fi

# Verify what's loaded
echo "Audio setup:"
pactl get-default-source
pactl get-default-sink
echo "Loaded modules:"
pactl list modules short | grep -E "echo-cancel|remap"

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
