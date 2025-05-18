"""Provides functionality to convert audio and video files to standard audio formats."""
import os
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from fastmcp import Context
from typing import List, Optional

SUPPORTED_FORMATS: List[str] = ["wav", "flac", "mp3", "ogg", "aac"]
"""List of audio formats supported for output by the conversion process."""

TARGET_SAMPLE_RATE: int = 16000
"""Target audio sample rate (in Hz) for the converted audio files (16kHz)."""

async def convert_to_audio(ctx: Context, audio_path: str, output_format: str = "wav", output_dir: str = "temp_audio") -> Optional[str]:
    """
    Converts an audio or video file to a specified audio format (WAV or FLAC default),
    resampling to 16kHz mono.

    This function takes an input file path, attempts to load it as an audio/video file,
    converts it to mono, resamples it to `TARGET_SAMPLE_RATE`, and then exports it
    to the specified `output_format` in the `output_dir`.
    FFmpeg must be installed and in the system's PATH for `pydub` to handle
    various formats.

    Args:
        ctx: The FastMCP Context object for logging and progress reporting.
        audio_path: Absolute path to the input audio or video file.
        output_format: Desired output audio format (e.g., "wav", "flac").
                       Defaults to "wav". Must be one of `SUPPORTED_FORMATS`.
        output_dir: Directory where the converted audio file will be saved.
                    If not an absolute path, it's treated as relative to the
                    current working directory where the server is run. It is
                    recommended to use absolute paths for `output_dir` in production.

    Returns:
        The absolute path to the converted audio file if successful, otherwise None.
    """
    if not os.path.isabs(audio_path):
        await ctx.warning(f"Received relative audio_path: '{audio_path}'. It is strongly recommended to use absolute paths for inputs.")

    if not os.path.exists(audio_path):
        await ctx.error(f"Audio/video file not found at {audio_path}")
        return None

    if output_format.lower() not in SUPPORTED_FORMATS:
        await ctx.error(f"Unsupported output format '{output_format}'. Supported formats: {SUPPORTED_FORMATS}")
        return None

    if not os.path.isabs(output_dir):
        await ctx.warning(f"output_dir '{output_dir}' is not absolute. Converted file will be saved relative to current working directory: {os.path.abspath(output_dir)}")

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            await ctx.info(f"Created output directory: {output_dir}")
        except OSError as e:
            await ctx.error(f"Could not create output directory '{output_dir}': {e}")
            return None

    try:
        await ctx.info(f"Loading audio/video file: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        await ctx.info(f"Successfully loaded audio/video. Duration: {len(audio) / 1000.0:.2f}s, Channels: {audio.channels}, Frame Rate: {audio.frame_rate}Hz")

        audio = audio.set_channels(1)
        await ctx.info("Converted to mono audio.")

        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
        await ctx.info(f"Resampled to {TARGET_SAMPLE_RATE}Hz.")

        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        output_filename = f"{base_filename}.{output_format.lower()}"
        output_path = os.path.join(os.path.abspath(output_dir), output_filename)

        await ctx.info(f"Exporting to {output_format.lower()} at: {output_path}")
        audio.export(output_path, format=output_format.lower())
        await ctx.info(f"Successfully converted to {output_path}")
        return output_path

    except CouldntDecodeError as e:
        await ctx.error(f"Error decoding audio/video file '{audio_path}': {e}. Ensure FFmpeg is installed and in your PATH.")
        return None
    except Exception as e:
        await ctx.error(f"An unexpected error occurred during conversion of '{audio_path}': {e} (Exception type: {type(e).__name__})")
        return None