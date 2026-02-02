#!/bin/bash
# Enable ReSpeaker 3.5mm Output for Hardware AEC
#
# Run this AFTER connecting a powered speaker to the ReSpeaker's 3.5mm jack.
# This enables the XMOS DSP's built-in echo cancellation, which is much better
# than software AEC because it has direct access to the playback reference signal.
#
# Usage:
#   ./enable_respeaker_output.sh         # Enable ReSpeaker output
#   ./enable_respeaker_output.sh --hdmi  # Switch back to HDMI
#

echo "ReSpeaker Audio Output Configuration"
echo "====================================="

# Find ReSpeaker sink
RESPEAKER_SINK=$(pactl list sinks short | grep -i respeaker | head -1 | cut -f2)
HDMI_SINK=$(pactl list sinks short | grep -i hdmi | head -1 | cut -f2)

echo "Available sinks:"
echo "  ReSpeaker: ${RESPEAKER_SINK:-NOT FOUND}"
echo "  HDMI:      ${HDMI_SINK:-NOT FOUND}"
echo ""

if [ "$1" == "--hdmi" ]; then
    # Switch back to HDMI
    if [ -n "$HDMI_SINK" ]; then
        pactl set-default-sink "$HDMI_SINK"
        echo "✓ Switched to HDMI output"
        echo ""
        echo "NOTE: Software AEC is active. Johnny may still hear herself."
    else
        echo "✗ HDMI sink not found!"
        exit 1
    fi
else
    # Enable ReSpeaker 3.5mm output
    if [ -z "$RESPEAKER_SINK" ]; then
        echo "✗ ReSpeaker not found!"
        echo ""
        echo "Make sure the ReSpeaker USB Mic Array is connected."
        exit 1
    fi

    # Set ReSpeaker as default output
    pactl set-default-sink "$RESPEAKER_SINK"
    echo "✓ Set ReSpeaker 3.5mm as default audio output"

    # The echo-cancel module is no longer needed for playback
    # since the XMOS DSP handles AEC internally
    echo ""
    echo "Hardware AEC is now active!"
    echo ""
    echo "The XMOS XVF-3000 DSP will automatically cancel Johnny's voice"
    echo "from the microphone input because it has access to the playback"
    echo "reference signal through the 3.5mm output path."
    echo ""
    echo "Benefits:"
    echo "  - Much better echo cancellation"
    echo "  - Can enable user interrupts (allow_user_interrupt=True)"
    echo "  - No 0.3s delay after speaking"
    echo "  - More natural conversation"
fi

echo ""
echo "Current default sink:"
pactl get-default-sink
echo ""
echo "Test audio:"
echo "  speaker-test -t wav -c 2 -l 1"
