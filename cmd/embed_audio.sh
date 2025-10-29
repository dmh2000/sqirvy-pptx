#!/usr/bin/env bash

# Script to embed audio files into PowerPoint slides
# Usage: ./embed_audio.sh <powerpoint_file>

set -e  # Exit on error

# Check if PowerPoint file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <powerpoint_file>"
    exit 1
fi

PPTX_FILE="$1"

# Check if PowerPoint file exists
if [ ! -f "$PPTX_FILE" ]; then
    echo "Error: PowerPoint file '$PPTX_FILE' not found"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMBED_SCRIPT="$SCRIPT_DIR/embed_audio.py"

# Check if embed_audio.py exists
if [ ! -f "$EMBED_SCRIPT" ]; then
    echo "Error: embed_audio.py not found at $EMBED_SCRIPT"
    exit 1
fi

# Find all audio-N.mp3 files in current directory and sort numerically by N
# Using process substitution to avoid subshell issues
while IFS= read -r audio_file; do
    # Extract the slide number from filename (audio-N.mp3 -> N)
    slide_number=$(echo "$audio_file" | sed -n 's/^audio-\([0-9]\+\)\.mp3$/\1/p')

    if [ -n "$slide_number" ]; then
        echo "Embedding $audio_file into slide $slide_number..."
        python3 "$EMBED_SCRIPT" -s "$slide_number" "$PPTX_FILE" "$audio_file"
    fi
done < <(find . -maxdepth 1 -name "audio-[0-9]*.mp3" -printf "%f\n" | sort -t'-' -k2 -n)

echo "All audio files embedded successfully!"
