"""Utility functions and constants for file and directory operations, particularly for temporary file management."""
import os
import shutil
from fastmcp import Context

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_FILE_DIR)
WORKSPACE_ROOT = os.path.dirname(SRC_DIR)

TEMP_BASE_DIR = os.path.join(WORKSPACE_ROOT, ".temp_transcription_server")
"""Base directory for all temporary files created by the server."""

TEMP_UPLOAD_DIR = os.path.join(TEMP_BASE_DIR, "temp_uploads")
"""Directory for temporary storage of uploaded files before processing."""

TEMP_AUDIO_DIR = os.path.join(TEMP_BASE_DIR, "temp_audio")
"""Directory for temporary storage of converted audio files."""

async def cleanup_temp_files(*paths: str, ctx: Context) -> None:
    """
    Removes specified files or directories.

    This function attempts to delete each path provided. If a path is a file,
    it's removed. If it's a directory, the entire directory tree is removed.
    Logs actions and errors using the provided `fastmcp.Context` object.

    Args:
        *paths: A variable number of string arguments, each representing a path
                to a file or directory to be cleaned up.
        ctx: The `fastmcp.Context` instance to use for logging. This is required.
    """
    for path in paths:
        if not path:
            continue
        try:
            if os.path.isfile(path):
                os.remove(path)
                log_msg = f"Cleaned up temporary file: {path}"
                await ctx.debug(log_msg)
            elif os.path.isdir(path):
                shutil.rmtree(path)
                log_msg = f"Cleaned up temporary directory: {path}"
                await ctx.debug(log_msg)
        except Exception as e:
            log_msg = f"Error cleaning up temporary path {path}: {e}"
            await ctx.warning(log_msg)
            pass