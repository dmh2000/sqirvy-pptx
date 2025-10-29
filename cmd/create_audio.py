#!/usr/bin/env python3
"""
Convert a text file to MP3 audio.

This script reads a text file and converts it to a single MP3 audio file
using Google Gemini TTS.
"""

import argparse
import os
import sys
from audio_utils import speak, speak11

def make_audio(output_file,text, voice="Sadaltager", output_dir=".", tts_function="speak"):
    """
    Convert a text file to a single MP3 audio file.

    Args:
        text_file: Path to the text file to convert
        voice: Voice name to use for TTS (default: "Sadaltager")
        output_dir: Directory to save the output MP3 file (default: current directory)
        tts_function: TTS function to use - "speak" (Google Gemini) or "speak11" (ElevenLabs) (default: "speak")
    """

    print(f"Text length: {len(text)} characters")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename based on input filename (without extension)
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}")

    # Select the TTS function
    if tts_function == "speak":
        tts_func = speak
        print(f"Converting to audio using Google Gemini with voice '{voice}'...")
    elif tts_function == "speak11":
        tts_func = speak11
        print(f"Converting to audio using ElevenLabs with voice '{voice}'...")
    else:
        print(f"Error: Invalid TTS function '{tts_function}'. Must be 'speak' or 'speak11'", file=sys.stderr)
        sys.exit(1)

    try:
        # Convert text to speech
        result = tts_func(voice, text, output_file)
        print(f"Successfully created: {result}")
    except Exception as e:
        print(f"Error converting text to audio: {e}", file=sys.stderr)
        sys.exit(1)

    return output_file