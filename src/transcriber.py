"""Handles audio transcription using a NeMo ASR model."""
import nemo.collections.asr as nemo_asr
import torch
import os
import math
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pydantic import ValidationError
from typing import Optional, Union, List, Any, Tuple
from fastmcp import Context
from src.models.transcription_models import TranscriptionInput, WordTimestamp, SegmentTimestamp, CharTimestamp, TranscriptionResult
from src.utils.file_utils import TEMP_AUDIO_DIR

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
    include_timestamps: bool,
    offset_seconds: float = 0.0
) -> Union[TranscriptionResult, str]:
    """
    Processes the raw output from NeMo's ASR model transcription.
    Adjusts timestamps by `offset_seconds` if provided.

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
        offset_seconds: Float indicating the offset to apply to timestamps.

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
        # If it's just a string, and we have an offset, it implies segmented transcription
        # without timestamps. The calling function will handle concatenation.
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
                                processed_word_timestamps.append(WordTimestamp(
                                    word=word, 
                                    start_time=start_time + offset_seconds, 
                                    end_time=end_time + offset_seconds
                                ))
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
                                processed_char_timestamps.append(CharTimestamp(
                                    char=char_val_list[0], 
                                    start_time=start_time + offset_seconds, 
                                    end_time=end_time + offset_seconds
                                ))
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
                                processed_segment_timestamps.append(SegmentTimestamp(
                                    text=text, 
                                    start_time=start_time + offset_seconds, 
                                    end_time=end_time + offset_seconds
                                ))
                            else:
                                await ctx.warning(f"Skipping segment timestamp item with missing fields: {item} for {audio_path}")
                        except ValidationError as e:
                            await ctx.warning(f"Validation error for segment timestamp item {item} for {audio_path}: {e}")
                    else:
                        await ctx.warning(f"Skipping non-dict segment timestamp item: {item} for {audio_path}")
        
        elif timestamp_data_dict is not None:
            await ctx.warning(f"Timestamp data for {audio_path} is not in the expected dictionary format. Type: {type(timestamp_data_dict)}")

        # Update the main transcribed_text if segments are present, to ensure it's consistent
        # This is particularly important if the original 'text' field at the top level of result_obj
        # doesn't include all segment text or if we want to reconstruct it from segments.
        if processed_segment_timestamps:
            transcribed_text = " ".join([s.text for s in processed_segment_timestamps])
        elif processed_word_timestamps: # Fallback if no segments, but words are there
            transcribed_text = " ".join([w.word for w in processed_word_timestamps])

        if include_timestamps:
            if not timestamp_data_dict:
                 await ctx.warning(f"Timestamps requested but no timestamp data found in model output for {audio_path}.")
            elif not (processed_word_timestamps or processed_char_timestamps or processed_segment_timestamps):
                 await ctx.warning(f"Timestamps requested and raw timestamp data found, but no valid timestamps were processed for {audio_path}. Check model output structure and parsing logic.")
        
        return TranscriptionResult(
            text=transcribed_text, # Use potentially reconstructed text
            word_timestamps=processed_word_timestamps,
            segment_timestamps=processed_segment_timestamps, 
            char_timestamps=processed_char_timestamps      
        )
    else: # Not include_timestamps
        if hasattr(result_obj, 'text'):
            return getattr(result_obj, 'text', "")
        elif isinstance(result_obj, str):
            return result_obj
        else:
            await ctx.warning(f"Unexpected output format when timestamps are false for {audio_path}. Type: {type(result_obj)}")
            return ""


async def _transcribe_single_file_segment(
    ctx: Context, 
    model: nemo_asr.models.ASRModel, 
    file_path: str, 
    include_timestamps: bool, 
    offset_seconds: float = 0.0
) -> Union[TranscriptionResult, str]:
    """Helper to transcribe a single audio file (or segment) and process its output."""
    await ctx.debug(f"Transcribing segment: {file_path} with offset: {offset_seconds}s")
    nemo_output = model.transcribe(
        [file_path],
        timestamps=include_timestamps
    )
    return await _process_transcription_output(ctx, nemo_output, file_path, include_timestamps, offset_seconds)


async def transcribe_audio_file(ctx: Context, input_data: TranscriptionInput) -> Union[TranscriptionResult, str]:
    """
    Transcribes an audio file using the loaded Parakeet ASR model.
    If the audio duration exceeds `input_data.segment_length_minutes`,
    it splits the audio into segments, transcribes them individually,
    and concatenates the results, adjusting timestamps accordingly.

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
    
    audio_path_str = str(input_data.audio_path)
    segment_length_minutes = input_data.segment_length_minutes
    segment_length_ms = segment_length_minutes * 60 * 1000

    await ctx.info(f"Beginning transcription for: {audio_path_str}, SegL: {segment_length_minutes}min, Timestamps: {input_data.include_timestamps}")

    try:
        audio = AudioSegment.from_file(audio_path_str)
    except CouldntDecodeError as e:
        await ctx.error(f"Pydub could not decode audio file {audio_path_str}: {e}")
        raise ValueError(f"Could not decode audio file: {audio_path_str}") from e
    except Exception as e:
        await ctx.error(f"Error loading audio file {audio_path_str} with pydub: {e}")
        raise ValueError(f"Error loading audio file with pydub: {audio_path_str}") from e

    duration_ms = len(audio)
    
    if duration_ms <= segment_length_ms:
        await ctx.info(f"Audio duration ({duration_ms/1000.0:.2f}s) is within segment length ({segment_length_minutes} min). Transcribing directly.")
        return await _transcribe_single_file_segment(ctx, model, audio_path_str, input_data.include_timestamps)
    else:
        await ctx.info(f"Audio duration ({duration_ms/1000.0:.2f}s) exceeds segment length ({segment_length_minutes} min). Splitting into segments.")
        
        num_segments = math.ceil(duration_ms / segment_length_ms)
        await ctx.info(f"Splitting into {num_segments} segments.")

        all_texts: List[str] = []
        all_word_timestamps: List[WordTimestamp] = []
        all_segment_timestamps: List[SegmentTimestamp] = []
        all_char_timestamps: List[CharTimestamp] = []
        
        temp_segment_paths: List[str] = []
        current_offset_seconds: float = 0.0

        original_basename = os.path.splitext(os.path.basename(audio_path_str))[0]
        output_format = os.path.splitext(audio_path_str)[1].lstrip('.')


        # Ensure TEMP_AUDIO_DIR exists (it should be created by audio_converter or server, but double check)
        if not os.path.exists(TEMP_AUDIO_DIR):
            try:
                os.makedirs(TEMP_AUDIO_DIR)
                await ctx.info(f"Created temporary directory for segments: {TEMP_AUDIO_DIR}")
            except OSError as e:
                await ctx.error(f"Could not create temporary segment directory '{TEMP_AUDIO_DIR}': {e}")
                raise # Re-raise to stop processing

        for i in range(num_segments):
            start_ms = i * segment_length_ms
            end_ms = min((i + 1) * segment_length_ms, duration_ms)
            segment = audio[start_ms:end_ms]
            
            segment_filename = f"{original_basename}_segment_{i+1}_of_{num_segments}.{output_format}"
            segment_path = os.path.join(TEMP_AUDIO_DIR, segment_filename)
            temp_segment_paths.append(segment_path)

            try:
                await ctx.info(f"Exporting segment {i+1}/{num_segments}: {segment_path} ({start_ms/1000.0:.2f}s to {end_ms/1000.0:.2f}s)")
                segment.export(segment_path, format=output_format)
            except Exception as e:
                await ctx.error(f"Failed to export audio segment {segment_path}: {e}")
                # Cleanup already created segments before raising
                for p in temp_segment_paths:
                    if os.path.exists(p): os.remove(p)
                raise ValueError(f"Failed to export audio segment {segment_path}") from e

            await ctx.report_progress(progress=i, total=num_segments) # Report progress per segment

            segment_result = await _transcribe_single_file_segment(
                ctx, model, segment_path, input_data.include_timestamps, current_offset_seconds
            )

            if isinstance(segment_result, str):
                all_texts.append(segment_result)
            elif isinstance(segment_result, TranscriptionResult):
                all_texts.append(segment_result.text)
                all_word_timestamps.extend(segment_result.word_timestamps)
                all_segment_timestamps.extend(segment_result.segment_timestamps)
                all_char_timestamps.extend(segment_result.char_timestamps)
            
            current_offset_seconds += (end_ms - start_ms) / 1000.0 # Increment offset by actual segment duration

        await ctx.report_progress(progress=num_segments, total=num_segments) # Final progress update

        # Cleanup temporary segment files
        for seg_path in temp_segment_paths:
            try:
                if os.path.exists(seg_path):
                    os.remove(seg_path)
                    await ctx.debug(f"Removed temporary segment: {seg_path}")
            except Exception as e:
                await ctx.warning(f"Could not remove temporary segment {seg_path}: {e}")
        
        final_text = " ".join(all_texts).strip()

        if not input_data.include_timestamps:
            return final_text
        else:
            return TranscriptionResult(
                text=final_text,
                word_timestamps=all_word_timestamps,
                segment_timestamps=all_segment_timestamps,
                char_timestamps=all_char_timestamps
            )