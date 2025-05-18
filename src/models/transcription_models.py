"""Pydantic models for transcription input, output, and timestamp structures."""
from pydantic import BaseModel, FilePath, validator, Field
from typing import List, Optional
import os

class TranscriptionInput(BaseModel):
    """Input data model for transcription requests."""
    audio_path: FilePath = Field(..., description="Absolute path to the audio file for transcription.")
    include_timestamps: bool = Field(True, description="Whether to include timestamps in the transcription result.")

    @validator('audio_path')
    def audio_path_must_exist_and_be_absolute(cls, v: FilePath) -> FilePath:
        """Validates that the audio path exists and converts it to an absolute path."""
        path_str = str(v)
        abs_path = os.path.abspath(path_str) if not os.path.isabs(path_str) else path_str
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Audio file not found at {abs_path}")
        return FilePath(abs_path)

class WordTimestamp(BaseModel):
    """Data model for word-level timestamps."""
    word: str = Field(..., description="The transcribed word.")
    start_time: float = Field(..., description="Start time of the word in seconds.")
    end_time: float = Field(..., description="End time of the word in seconds.")

class SegmentTimestamp(BaseModel):
    """Data model for segment-level timestamps."""
    text: str = Field(..., description="The transcribed text segment.")
    start_time: float = Field(..., description="Start time of the segment in seconds.")
    end_time: float = Field(..., description="End time of the segment in seconds.")

class CharTimestamp(BaseModel):
    """Data model for character-level timestamps."""
    char: str = Field(..., description="The transcribed character.")
    start_time: float = Field(..., description="Start time of the character in seconds.")
    end_time: float = Field(..., description="End time of the character in seconds.")

class TranscriptionResult(BaseModel):
    """Data model for the complete transcription result."""
    text: str = Field(..., description="The full transcribed text.")
    word_timestamps: List[WordTimestamp] = Field([], description="List of word-level timestamps.")
    segment_timestamps: List[SegmentTimestamp] = Field([], description="List of segment-level timestamps.")
    char_timestamps: List[CharTimestamp] = Field([], description="List of character-level timestamps.") 