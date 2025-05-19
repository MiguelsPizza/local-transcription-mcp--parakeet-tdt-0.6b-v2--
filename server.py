import os
from typing import Annotated, Literal
import platform
import psutil
import torch

from pydantic import Field
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError

from src.audio_converter import convert_to_audio
from src.transcriber import (
    transcribe_audio_file,
    load_model as load_transcription_model
)
from src.models.transcription_models import TranscriptionInput, TranscriptionResult
from src.utils.file_utils import TEMP_AUDIO_DIR, cleanup_temp_files
from src.utils.formatting_utils import _format_transcription_output

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SERVER_DIR, ".."))

mcp = FastMCP(
    name="parakeet-transcription-server",
    version="0.1.0",
    dependencies=[
        "nemo_toolkit[asr]",
        "pydantic",
        "pydub",
        "torch",
        "torchaudio",
        "psutil"
    ],
    instructions="""This server provides tools for audio/video transcription using NVIDIA Parakeet TDT 0.6B V2 
and resources to get ASR model information and system hardware specifications.
FFmpeg must be installed and in your PATH. All file paths provided to tools must be absolute.

Recommended Workflow:
1. Read resource `info://system_hardware_specs` to understand the server's hardware capabilities (CPU, RAM, GPU).
2. Based on the hardware specs, determine an appropriate `segment_length_minutes` for the `transcribe_audio` tool.
   - For systems with capable NVIDIA GPUs, longer segments (up to 24 minutes) can be used.
   - For systems without NVIDIA GPUs or with limited RAM, it is highly recommended to use shorter segments (e.g., 1-10 minutes) to conserve resources. Choose an appropriate length.
3. Call `transcribe_audio` tool with the audio file and the chosen `segment_length_minutes`.
4. Resource `info://asr_model` can be read anytime for model-specific details."""
)

@mcp.tool(
    description="Transcribes an audio/video file to text using Parakeet TDT 0.6B V2.",
    annotations={"readOnlyHint": True},
    tags={"transcription", "asr", "audio", "video"},
)
async def transcribe_audio(
    ctx: Context,
    audio_file_path: Annotated[str, Field(description="Absolute path to the audio/video file to transcribe. The server will process this file.")],
    output_format: Annotated[Literal["wav", "flac"], Field(description="The intermediate audio format ('wav' or 'flac'). Parakeet supports both. This determines the format of the temporary audio file processed by the ASR model.")] = "wav",
    include_timestamps: Annotated[bool, Field(description="Whether to include word and segment level timestamps in the output. If true, the output may be formatted with timestamps.")] = True,
    line_character_limit: Annotated[int, Field(description="Character limit per line for formatted transcription output with timestamps. Default is 80.", ge=40, le=200)] = 80,
    segment_length_minutes: Annotated[int, Field(description="Maximum length of audio segments in minutes for transcription. Audio longer than this will be split. Default is 5 minutes, max 24.", ge=1, le=24)] = 5
) -> dict:
    """
    Takes the absolute path to an audio/video file, converts it to WAV or FLAC,
    and returns the transcription. If an error occurs, it raises a ToolError.
    Audio files longer than `segment_length_minutes` will be split into segments
    of that approximate duration and transcribed sequentially.

    Args:
        ctx: The MCP Context object.
        audio_file_path: Absolute path to the audio/video file to transcribe.
        output_format: The intermediate audio format ('wav' or 'flac'). Default is 'wav'.
        include_timestamps: Whether to include word and segment level timestamps in the output.
                            Default is True.
        line_character_limit: Character limit for formatting output with timestamps.
                            Default is 80.
        segment_length_minutes: Maximum length of audio segments in minutes.
                                Audio longer than this will be split. Default is 5.

    Returns:
        A dictionary containing the transcription, potentially with timestamps.

    Raises:
        ToolError: If any error occurs during the process (e.g., file not found,
                   conversion failure, transcription failure, or invalid input).
    """
    await ctx.info(f"Received audio_file_path: '{audio_file_path}', output_format: '{output_format}', include_timestamps: {include_timestamps}, line_char_limit: {line_character_limit}, segment_length_minutes: {segment_length_minutes}")

    if not isinstance(audio_file_path, str) or not audio_file_path.strip():
        error_msg = "Error: audio_file_path must be a non-empty string."
        await ctx.error(error_msg)
        raise ToolError(error_msg)

    if not os.path.isabs(audio_file_path):
        error_msg = f"Error: Please provide an absolute path for the audio/video file. Received: '{audio_file_path}'"
        await ctx.error(error_msg)
        raise ToolError(error_msg)

    if not os.path.exists(audio_file_path):
        error_msg = f"Error: File not found at '{audio_file_path}'"
        await ctx.error(error_msg)
        raise ToolError(error_msg)

    converted_audio_path = None
    try:
        await ctx.info(f"Starting transcription for: {audio_file_path}")
        await ctx.report_progress(0, 3)

        await ctx.info(f"Converting '{audio_file_path}' to '{output_format}'...")
        converted_audio_path = await convert_to_audio(ctx, audio_file_path, output_format, TEMP_AUDIO_DIR)

        if not converted_audio_path:
            error_msg = f"Error: Failed to convert file '{audio_file_path}'."
            await ctx.error(error_msg)
            raise ToolError(error_msg)

        await ctx.info(f"Converted to: {converted_audio_path}")
        await ctx.report_progress(1, 3)

        await ctx.info(f"Transcribing '{converted_audio_path}'...")

        transcription_input = TranscriptionInput(
            audio_path=converted_audio_path,
            include_timestamps=include_timestamps,
            segment_length_minutes=segment_length_minutes
        )
        transcription_result = await transcribe_audio_file(ctx, transcription_input)
        await ctx.report_progress(2, 3)

        if transcription_result is None:
            error_msg = f"Error: Transcription failed for '{converted_audio_path}' (returned None)."
            await ctx.error(error_msg)
            raise ToolError(error_msg)

        await ctx.info("Transcription successful.")
        await ctx.report_progress(3, 3)

        final_transcription_output: str
        response_message: str

        if isinstance(transcription_result, str):
            final_transcription_output = transcription_result
            response_message = "Transcription successful (text only)."
        elif isinstance(transcription_result, TranscriptionResult):
            if include_timestamps and transcription_result.word_timestamps:
                final_transcription_output = _format_transcription_output(transcription_result, line_char_limit=line_character_limit)
                response_message = "Transcription successful with formatted timestamps."
            else:
                final_transcription_output = transcription_result.text
                if include_timestamps:
                    response_message = "Transcription successful (text only; detailed timestamps not available or empty for formatting)."
                else:
                    response_message = "Transcription successful (text only)."
        else:
            await ctx.error(f"Internal error: transcription_result was neither str nor TranscriptionResult. Type: {type(transcription_result)}. Result: {transcription_result}")
            final_transcription_output = "Error: Transcription produced an unexpected data structure."
            response_message = "Transcription error: unexpected result data structure."

        return {
            "message": response_message,
            "file_processed": audio_file_path,
            "transcription": final_transcription_output
        }

    except Exception as e:
        await ctx.error(f"Server error during transcription: {str(e)}")
        raise ToolError(f"Error: An unexpected server error occurred: {str(e)}")
    finally:
        if converted_audio_path:
            await cleanup_temp_files(converted_audio_path, ctx=ctx)


@mcp.resource(
    uri="info://asr_model",
    name="asr_model_information",
    description="Provides information about the ASR model being used (Nvidia Parakeet TDT 0.6B V2).",
    tags={"asr", "model-info"}
)
async def get_asr_model_info(ctx: Context) -> dict:
    """Returns details about the loaded ASR model."""
    from src.transcriber import MODEL_NAME as parakeet_model_name
    model = await load_transcription_model(ctx)
    if model:
        return {
            "model_name": parakeet_model_name,
            "status": "Loaded",
            "input_requirements": "16kHz Audio (.wav or .flac), Monochannel",
            "output_type": "Text with optional Punctuation, Capitalization, and Timestamps.",
            "license": "CC-BY-4.0",
            "note": "This model is optimized for NVIDIA GPU-accelerated systems."
        }
    else:
        return {
            "model_name": parakeet_model_name,
            "status": "Error loading model or model not loaded yet.",
            "note": "Transcription services will not be available until the model is loaded."
        }

@mcp.resource(
    uri="info://system_hardware_specs",
    name="system_hardware_specifications",
    description="Retrieves system hardware specifications relevant for performance estimation.",
    tags={"system", "hardware", "info"}
)
async def get_system_hardware_specs(ctx: Context) -> dict:
    """
    Gathers and returns system hardware information including OS, CPU, RAM, and GPU.
    This information can be used to estimate performance and recommend appropriate
    settings for tasks like audio segmentation.
    """
    await ctx.info("Gathering system hardware specifications...")
    specs = {}

    try:
        specs['os_platform'] = platform.system()
        specs['os_version'] = platform.version()
        specs['os_release'] = platform.release()
        specs['architecture'] = platform.machine()

        specs['cpu_model'] = platform.processor() if platform.processor() else "N/A"
        specs['cpu_physical_cores'] = psutil.cpu_count(logical=False)
        specs['cpu_logical_cores'] = psutil.cpu_count(logical=True)
        cpu_freq_info = psutil.cpu_freq()
        specs['cpu_frequency_max_ghz'] = round(cpu_freq_info.max / 1000, 2) if cpu_freq_info else "N/A"
        
        svmem = psutil.virtual_memory()
        specs['ram_total_gb'] = round(svmem.total / (1024**3), 2)
        specs['ram_available_gb'] = round(svmem.available / (1024**3), 2)

        gpu_info = []
        if torch.cuda.is_available():
            specs['cuda_available'] = True
            specs['cuda_version'] = torch.version.cuda
            num_gpus = torch.cuda.device_count()
            specs['gpu_count'] = num_gpus
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory_total_gb = round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                gpu_info.append({
                    "name": gpu_name,
                    "memory_total_gb": gpu_memory_total_gb,
                    "cuda_capability": torch.cuda.get_device_capability(i)
                })
        else:
            specs['cuda_available'] = False
            specs['gpu_count'] = 0
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                 gpu_info.append({
                    "name": "Apple Metal Performance Shaders (MPS)",
                    "memory_total_gb": "N/A (Shared with System Memory)",
                    "notes": "MPS is available for PyTorch on this Mac."
                })
            elif platform.system() == "Darwin":
                gpu_info.append({
                    "name": "Apple GPU (Integrated or Discrete)",
                    "memory_total_gb": "N/A (Likely Shared with System Memory)",
                    "notes": "CUDA not available. Specific GPU details may require OS-specific tools."
                })
        specs['gpus'] = gpu_info
        
        await ctx.info(f"Successfully gathered hardware specs: {specs}")
        return specs

    except Exception as e:
        await ctx.error(f"Error gathering system hardware specifications: {e}")
        error_response = {"error": f"Failed to retrieve full hardware specs: {str(e)}"}
        if specs: 
            specs["error_partial_results"] = str(e)
            return specs
        return error_response