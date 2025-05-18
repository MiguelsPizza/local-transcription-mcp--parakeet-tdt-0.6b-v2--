"""Utility functions for formatting transcription outputs."""
from typing import Union, List, Dict, Any # Added Dict, Any for more precise dict typing
from src.models.transcription_models import TranscriptionResult, WordTimestamp # Import necessary models

def _format_transcription_output(transcription_data: Union[TranscriptionResult, dict], line_char_limit: int = 80) -> str:
    """
    Formats the transcription text with timestamps prepended to lines
    when they wrap due to character limit.
    Accepts either a TranscriptionResult Pydantic model or a dictionary (for fallback).
    """
    words: list
    full_text: str

    if isinstance(transcription_data, TranscriptionResult):
        words = transcription_data.word_timestamps
        full_text = transcription_data.text
    elif isinstance(transcription_data, dict):
        words = transcription_data.get("word_timestamps", [])
        full_text = transcription_data.get("text", "")
    else:
        return ""

    if not words:
        return full_text

    formatted_lines = []
    current_line_content = ""
    current_line_prefix = ""

    for i, word_obj in enumerate(words):
        word_text: str
        word_start_time: Union[float, None]

        if isinstance(word_obj, WordTimestamp):
            word_text = word_obj.word
            word_start_time = word_obj.start_time
        elif isinstance(word_obj, dict):
            word_text = word_obj.get("word", "")
            word_start_time = word_obj.get("start_time", word_obj.get("start"))
        else:
            continue

        if not word_text.strip():
            continue

        current_word_segment = f" {word_text}" if current_line_content else word_text

        if word_start_time is None:
            if not current_line_content:
                current_line_content = word_text
            elif len(current_line_prefix + current_line_content + current_word_segment) > line_char_limit:
                formatted_lines.append(current_line_prefix + current_line_content)
                current_line_prefix = ""
                current_line_content = word_text
            else:
                current_line_content += current_word_segment
            continue

        new_prefix = f"[{word_start_time:.2f}s] "
        if not current_line_content:
            current_line_prefix = new_prefix
            current_line_content = word_text
        else:
            if len(current_line_prefix + current_line_content + current_word_segment) > line_char_limit:
                formatted_lines.append(current_line_prefix + current_line_content)
                current_line_prefix = new_prefix
                current_line_content = word_text
            else:
                current_line_content += current_word_segment

    if current_line_content:
        formatted_lines.append(current_line_prefix + current_line_content)

    return "\n".join(formatted_lines)