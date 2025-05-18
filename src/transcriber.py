"""Handles audio transcription using a NeMo ASR model."""
import nemo.collections.asr as nemo_asr
import torch
import os
from pydantic import ValidationError
from typing import Optional, Union, List, Any
from fastmcp import Context
from src.models.transcription_models import TranscriptionInput, WordTimestamp, SegmentTimestamp, CharTimestamp, TranscriptionResult

MODEL_NAME: str = "nvidia/parakeet-tdt-0.6b-v2"
"""The specific NeMo ASR model to be used for transcription."""

ASR_MODEL: Optional[nemo_asr.models.ASRModel] = None
"""Global variable to hold the loaded ASR model instance. Lazily loaded."""

async def load_model(ctx: Context) -> nemo_asr.models.ASRModel:
    """
    Loads the Parakeet ASR model specified by `MODEL_NAME`.

    This function checks if the model is already loaded. If not, it attempts to
    download and instantiate it from NeMo's pretrained models. It logs progress
    and success/failure using the provided context.
    Detects GPU availability and logs information accordingly.

    Args:
        ctx: The FastMCP Context object for logging and progress reporting.

    Returns:
        The loaded `nemo_asr.models.ASRModel` instance.

    Raises:
        Exception: Propagates any exception raised during model loading from NeMo.
    """
    global ASR_MODEL
    if ASR_MODEL is not None:
        await ctx.debug(f"ASR model {MODEL_NAME} already loaded.")
        return ASR_MODEL

    await ctx.info(f"Loading ASR model: {MODEL_NAME}...")
    await ctx.report_progress(progress=0, total=100)

    if torch.cuda.is_available():
        await ctx.info("NVIDIA GPU detected. Model will run on GPU.")
    else:
        await ctx.warning("No NVIDIA GPU detected. Model will run on CPU (this may be slow).")
    
    try:
        await ctx.report_progress(progress=10, total=100)
        
        ASR_MODEL = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        
        await ctx.info(f"Model {MODEL_NAME} loaded successfully.")
        await ctx.report_progress(progress=100, total=100)
            
    except Exception as e:
        await ctx.error(f"Failed to load ASR model {MODEL_NAME}: {e}")
        await ctx.report_progress(progress=100, total=100)
        raise 
        
    return ASR_MODEL

async def _process_transcription_output(
    ctx: Context,
    nemo_output: Any,
    audio_path: str,
    include_timestamps: bool
) -> Union[TranscriptionResult, str]:
    """
    Processes the raw output from NeMo's ASR model transcription.

    This helper function parses the output from `model.transcribe()`. If timestamps
    are requested and available, it extracts word, segment, and character timestamps
    and populates a `TranscriptionResult` object. Otherwise, it returns the plain
    transcribed text.
    Includes extensive error checking and warnings for malformed or missing data
    in the NeMo output, especially concerning timestamp structures.

    Args:
        ctx: The FastMCP Context object for logging warnings or errors.
        nemo_output: The direct output from the NeMo ASR model's `transcribe` method.
        audio_path: The path to the audio file, used for context in logging.
        include_timestamps: Boolean indicating whether timestamps were requested and
                            should be processed.

    Returns:
        A `TranscriptionResult` object if timestamps are included and processed,
        otherwise a string containing the transcribed text.

    Raises:
        ValueError: If the transcription output is empty or fundamentally unprocessable.
    """
    if not nemo_output or not nemo_output[0]:
        await ctx.error(f"Transcription output is empty for file: {audio_path}.")
        raise ValueError(f"Transcription output empty for {audio_path}")

    result_obj = nemo_output[0]

    if isinstance(result_obj, str):
        return result_obj

    transcribed_text = getattr(result_obj, 'text', "")

    if include_timestamps:
        timestamp_data_dict = getattr(result_obj, 'timestamp', None)
        
        processed_word_timestamps = []
        processed_segment_timestamps = []
        processed_char_timestamps = []

        if isinstance(timestamp_data_dict, dict):
            word_list = timestamp_data_dict.get('word')
            if isinstance(word_list, list):
                for item in word_list:
                    if isinstance(item, dict):
                        try:
                            word = item.get('word')
                            start_time = item.get('start')
                            end_time = item.get('end')
                            if word is not None and start_time is not None and end_time is not None:
                                processed_word_timestamps.append(WordTimestamp(word=word, start_time=start_time, end_time=end_time))
                            else:
                                await ctx.warning(f"Skipping word timestamp item with missing fields: {item} for {audio_path}")
                        except ValidationError as e:
                            await ctx.warning(f"Validation error for word timestamp item {item} for {audio_path}: {e}")
                    else:
                        await ctx.warning(f"Skipping non-dict word timestamp item: {item} for {audio_path}")

            char_list = timestamp_data_dict.get('char')
            if isinstance(char_list, list):
                for item in char_list:
                    if isinstance(item, dict):
                        try:
                            char_val_list = item.get('char')
                            start_time = item.get('start')
                            end_time = item.get('end')
                            if char_val_list and isinstance(char_val_list, list) and \
                               len(char_val_list) > 0 and char_val_list[0] is not None and \
                               start_time is not None and end_time is not None:
                                processed_char_timestamps.append(CharTimestamp(char=char_val_list[0], start_time=start_time, end_time=end_time))
                            else:
                                await ctx.warning(f"Skipping char timestamp item with missing/invalid fields: {item} for {audio_path}")
                        except (ValidationError, IndexError, TypeError) as e:
                            await ctx.warning(f"Error processing char timestamp item {item} for {audio_path}: {e}")
                    else:
                        await ctx.warning(f"Skipping non-dict char timestamp item: {item} for {audio_path}")
            
            segment_list = timestamp_data_dict.get('segment')
            if isinstance(segment_list, list):
                for item in segment_list:
                    if isinstance(item, dict):
                        try:
                            text = item.get('segment')
                            start_time = item.get('start')
                            end_time = item.get('end')
                            if text is not None and start_time is not None and end_time is not None:
                                processed_segment_timestamps.append(SegmentTimestamp(text=text, start_time=start_time, end_time=end_time))
                            else:
                                await ctx.warning(f"Skipping segment timestamp item with missing fields: {item} for {audio_path}")
                        except ValidationError as e:
                            await ctx.warning(f"Validation error for segment timestamp item {item} for {audio_path}: {e}")
                    else:
                        await ctx.warning(f"Skipping non-dict segment timestamp item: {item} for {audio_path}")
        
        elif timestamp_data_dict is not None:
            await ctx.warning(f"Timestamp data for {audio_path} is not in the expected dictionary format. Type: {type(timestamp_data_dict)}")

        if include_timestamps:
            if not timestamp_data_dict:
                 await ctx.warning(f"Timestamps requested but no timestamp data found in model output for {audio_path}.")
            elif not (processed_word_timestamps or processed_char_timestamps or processed_segment_timestamps):
                 await ctx.warning(f"Timestamps requested and raw timestamp data found, but no valid timestamps were processed for {audio_path}. Check model output structure and parsing logic.")
        
        return TranscriptionResult(
            text=transcribed_text,
            word_timestamps=processed_word_timestamps,
            segment_timestamps=processed_segment_timestamps, 
            char_timestamps=processed_char_timestamps      
        )
    else:
        if hasattr(result_obj, 'text'):
            return getattr(result_obj, 'text', "")
        elif isinstance(result_obj, str):
            return result_obj
        else:
            await ctx.warning(f"Unexpected output format when timestamps are false for {audio_path}. Type: {type(result_obj)}")
            return ""


async def transcribe_audio_file(ctx: Context, input_data: TranscriptionInput) -> Union[TranscriptionResult, str]:
    """
    Transcribes an audio file using the loaded Parakeet ASR model.

    This function first ensures the ASR model is loaded (calling `load_model`).
    Then, it invokes the model's `transcribe` method with the provided audio file path
    and timestamp preference. The raw output is then processed by
    `_process_transcription_output`.

    Args:
        ctx: The FastMCP Context object for logging and progress reporting.
        input_data: A `TranscriptionInput` Pydantic model containing the
                    absolute path to the audio file and a flag for including timestamps.

    Returns:
        A `TranscriptionResult` object containing the transcription text and detailed
        timestamps (if requested and available), or a plain string with the transcribed text.

    Raises:
        FileNotFoundError: If the audio file specified in `input_data` does not exist
                           (primarily handled by `TranscriptionInput` validator).
        ValueError: If transcription output is empty (propagated from `_process_transcription_output`).
        Various exceptions from NeMo if model loading or the transcription process itself fails.
    """
    model = await load_model(ctx)

    await ctx.info(f"Transcribing audio file: {input_data.audio_path} (Timestamps: {input_data.include_timestamps})")
    
    nemo_output = model.transcribe(
        [str(input_data.audio_path)],
        timestamps=input_data.include_timestamps
    )
    await ctx.info(f"Nemo output: {nemo_output}")
    
    return await _process_transcription_output(ctx, nemo_output, str(input_data.audio_path), input_data.include_timestamps)