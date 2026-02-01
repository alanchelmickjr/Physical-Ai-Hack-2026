#!/bin/bash
export DISPLAY=:0

# Set HDMI as default audio output
pactl set-default-sink alsa_output.platform-3510000.hda.hdmi-stereo 2>/dev/null || true

pkill -f whoami_full || true
sleep 1
cd /home/robbie
nohup python3 whoami_full.py > /tmp/whoami.log 2>&1 &
