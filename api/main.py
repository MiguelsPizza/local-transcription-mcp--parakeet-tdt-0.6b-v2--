import asyncio
import logging
import os
import platform
import psutil
import shutil
import torch
import uuid
from pathlib import Path
from typing import Literal, Annotated

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
from pydantic import BaseModel, Field

# Configure basic logging for the API
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Attempt to import necessary modules from the src directory
# Assuming the script is run from the project root or PYTHONPATH is set up
try:
    from src.audio_converter import convert_to_audio
    from src.transcriber import (
        transcribe_audio_file,
        load_model as load_transcription_model,
        MODEL_NAME as PARAKKEET_MODEL_NAME
    )
    from src.models.transcription_models import TranscriptionInput, TranscriptionResult
    from src.utils.file_utils import TEMP_AUDIO_DIR as SHARED_TEMP_AUDIO_DIR, cleanup_temp_files
    from src.utils.formatting_utils import _format_transcription_output
except ImportError as e:
    logger.error(f"Failed to import src modules: {e}. Ensure PYTHONPATH is set correctly or run from project root.")
    # Define dummy functions or raise an error to prevent app startup if critical modules are missing
    # For now, we'll let it proceed and FastAPI will fail on endpoint calls if these are not loaded.
    # In a production scenario, you might want to handle this more gracefully or prevent startup.
    PARAKKEET_MODEL_NAME = "N/A due to import error"
    SHARED_TEMP_AUDIO_DIR = Path("temp_audio_files_api_fallback") # Fallback temp dir


# Ensure the base temporary directory for audio processing exists
TEMP_AUDIO_DIR = Path(SHARED_TEMP_AUDIO_DIR)
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TEMP_UPLOADS_DIR = TEMP_AUDIO_DIR / "uploads"
TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class SimpleContext:
    """
    A simple context-like class for providing logging and basic async methods
    that src functions might expect if they were designed for FastMCP's context.
    """
    async def info(self, message: str):
        logger.info(message)

    async def error(self, message:str):
        logger.error(message)

    async def debug(self, message: str):
        logger.debug(message)

    async def warning(self, message: str):
        logger.warning(message)

    async def report_progress(self, progress: int, total: int):
        logger.info(f"Progress: {progress}/{total}")

simple_ctx = SimpleContext()

app = FastAPI(
    title="Parakeet Transcription API",
    version="0.1.0",
    description="API for transcribing audio/video files and getting model/system info.",
)

# Add CORS middleware for permissive testing
# WARNING: This is highly permissive and should NOT be used in production.
# For production, restrict origins, methods, and headers as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Pydantic models for API responses (optional, but good practice)
class TranscriptionResponse(BaseModel):
    message: str
    file_processed: str
    transcription: str

class ModelInfoResponse(BaseModel):
    model_name: str
    status: str
    input_requirements: str | None = None
    output_type: str | None = None
    license: str | None = None
    note: str | None = None

class GPUInfo(BaseModel):
    name: str
    memory_total_gb: float | str
    cuda_capability: tuple[int, int] | None = None
    notes: str | None = None

class SystemHardwareResponse(BaseModel):
    os_platform: str | None = None
    os_version: str | None = None
    os_release: str | None = None
    architecture: str | None = None
    cpu_model: str | None = None
    cpu_physical_cores: int | None = None
    cpu_logical_cores: int | None = None
    cpu_frequency_max_ghz: float | str | None = None
    ram_total_gb: float | None = None
    ram_available_gb: float | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None
    gpu_count: int | None = None
    gpus: list[GPUInfo] = []
    error: str | None = None
    error_partial_results: str | None = None


@app.post("/transcribe/", response_model=TranscriptionResponse)
async def api_transcribe_audio(
    file: UploadFile = File(..., description="Audio or video file to transcribe."),
    output_format: Annotated[Literal["wav", "flac"], Form(description="Intermediate audio format for ASR.")] = "wav",
    include_timestamps: Annotated[bool, Form(description="Include word/segment timestamps in output.")] = True,
    line_character_limit: Annotated[int, Form(description="Char limit per line for timestamped output.", ge=40, le=200)] = 80,
    segment_length_minutes: Annotated[int, Form(description="Max audio segment length in minutes (1-24).", ge=1, le=24)] = 5
):
    """
    Transcribes an uploaded audio/video file.
    """
    # Save uploaded file temporarily
    # Use a unique filename to avoid collisions
    temp_file_suffix = Path(file.filename or "unknown_file").suffix
    temp_file_path = TEMP_UPLOADS_DIR / f"{uuid.uuid4()}{temp_file_suffix}"
    absolute_temp_file_path = str(temp_file_path.resolve())

    try:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        await simple_ctx.info(f"Uploaded file '{file.filename}' saved to: {absolute_temp_file_path}")

        converted_audio_path = None
        try:
            await simple_ctx.info(f"Converting '{file.filename}' to '{output_format}'...")
            # convert_to_audio expects output_dir to be TEMP_AUDIO_DIR
            converted_audio_path = await convert_to_audio(simple_ctx, absolute_temp_file_path, output_format, TEMP_AUDIO_DIR)

            if not converted_audio_path:
                raise HTTPException(status_code=500, detail=f"Failed to convert file '{file.filename}'.")
            await simple_ctx.info(f"Converted to: {converted_audio_path}")

            await simple_ctx.info(f"Transcribing '{converted_audio_path}'...")
            transcription_input_model = TranscriptionInput(
                audio_path=converted_audio_path, # Should be absolute path
                include_timestamps=include_timestamps,
                segment_length_minutes=segment_length_minutes
            )
            # Ensure transcribe_audio_file can be called with SimpleContext
            transcription_result_model = await transcribe_audio_file(simple_ctx, transcription_input_model)

            if transcription_result_model is None:
                raise HTTPException(status_code=500, detail=f"Transcription failed for '{converted_audio_path}' (returned None).")
            await simple_ctx.info("Transcription successful.")

            final_transcription_output: str
            response_message: str

            if isinstance(transcription_result_model, str):
                final_transcription_output = transcription_result_model
                response_message = "Transcription successful (text only)."
            elif isinstance(transcription_result_model, TranscriptionResult):
                if include_timestamps and transcription_result_model.word_timestamps:
                    final_transcription_output = _format_transcription_output(transcription_result_model, line_char_limit=line_character_limit)
                    response_message = "Transcription successful with formatted timestamps."
                else:
                    final_transcription_output = transcription_result_model.text
                    response_message = "Transcription successful (text only; timestamps not available or not requested for formatting)."
            else:
                await simple_ctx.error(f"Internal error: transcription_result was neither str nor TranscriptionResult. Type: {type(transcription_result_model)}. Result: {transcription_result_model}")
                # This case should ideally result in a 500 error
                raise HTTPException(status_code=500, detail="Transcription produced an unexpected data structure.")

            return TranscriptionResponse(
                message=response_message,
                file_processed=file.filename or "N/A",
                transcription=final_transcription_output
            )
        
        except HTTPException: # Re-raise HTTPExceptions directly
            raise
        except Exception as e:
            logger.exception(f"API error during transcription processing for {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file '{file.filename}': {str(e)}")
        finally:
            if converted_audio_path:
                # cleanup_temp_files expects the *specific file* to clean, not a directory
                await cleanup_temp_files(converted_audio_path, ctx=simple_ctx)
            # The originally uploaded temp file must also be cleaned
            if temp_file_path.exists():
                try:
                    os.remove(temp_file_path)
                    await simple_ctx.info(f"Cleaned up temporary uploaded file: {absolute_temp_file_path}")
                except OSError as e:
                    await simple_ctx.error(f"Error cleaning up temporary uploaded file {absolute_temp_file_path}: {e}")

    except HTTPException: # Re-raise HTTPExceptions from file saving
            raise
    except Exception as e:
        logger.exception(f"API error handling file upload {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error handling file upload '{file.filename}': {str(e)}")
    finally:
        # Ensure file object is closed
        if hasattr(file, 'file') and hasattr(file.file, 'close') and not file.file.closed:
            file.file.close()


@app.get("/info/asr-model/", response_model=ModelInfoResponse)
async def api_get_asr_model_info():
    """
    Provides information about the ASR model being used.
    Replicates logic from server.py's get_asr_model_info.
    """
    try:
        # load_transcription_model from src.transcriber
        model = await load_transcription_model(simple_ctx)
        if model:
            return ModelInfoResponse(
                model_name=PARAKKEET_MODEL_NAME,
                status="Loaded",
                input_requirements="16kHz Audio (.wav or .flac), Monochannel",
                output_type="Text with optional Punctuation, Capitalization, and Timestamps.",
                license="CC-BY-4.0",
                note="This model is optimized for NVIDIA GPU-accelerated systems."
            )
        else:
            return ModelInfoResponse(
                model_name=PARAKKEET_MODEL_NAME,
                status="Error loading model or model not loaded yet.",
                note="Transcription services will not be available until the model is loaded."
            )
    except Exception as e:
        logger.exception(f"Error in api_get_asr_model_info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get ASR model info: {str(e)}")


@app.get("/info/system-hardware/", response_model=SystemHardwareResponse)
async def api_get_system_hardware_specs():
    """
    Retrieves system hardware specifications.
    Replicates logic from server.py's get_system_hardware_specs.
    """
    await simple_ctx.info("Gathering system hardware specifications for API request...")
    specs_dict: dict = {} # Use a dict that matches SystemHardwareResponse fields

    try:
        specs_dict['os_platform'] = platform.system()
        specs_dict['os_version'] = platform.version()
        specs_dict['os_release'] = platform.release()
        specs_dict['architecture'] = platform.machine()

        specs_dict['cpu_model'] = platform.processor() if platform.processor() else "N/A"
        specs_dict['cpu_physical_cores'] = psutil.cpu_count(logical=False)
        specs_dict['cpu_logical_cores'] = psutil.cpu_count(logical=True)
        cpu_freq_info = psutil.cpu_freq()
        specs_dict['cpu_frequency_max_ghz'] = round(cpu_freq_info.max / 1000, 2) if cpu_freq_info and hasattr(cpu_freq_info, 'max') else "N/A"
        
        svmem = psutil.virtual_memory()
        specs_dict['ram_total_gb'] = round(svmem.total / (1024**3), 2)
        specs_dict['ram_available_gb'] = round(svmem.available / (1024**3), 2)

        gpu_info_list = []
        if torch.cuda.is_available():
            specs_dict['cuda_available'] = True
            specs_dict['cuda_version'] = torch.version.cuda
            num_gpus = torch.cuda.device_count()
            specs_dict['gpu_count'] = num_gpus
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory_total_gb = round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                gpu_info_list.append(GPUInfo(
                    name=gpu_name,
                    memory_total_gb=gpu_memory_total_gb,
                    cuda_capability=torch.cuda.get_device_capability(i)
                ))
        else:
            specs_dict['cuda_available'] = False
            specs_dict['gpu_count'] = 0
            # Check for Apple MPS (Metal Performance Shaders)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                 gpu_info_list.append(GPUInfo(
                    name="Apple Metal Performance Shaders (MPS)",
                    memory_total_gb="N/A (Shared with System Memory)",
                    notes="MPS is available for PyTorch on this Mac."
                ))
            elif platform.system() == "Darwin": # Generic Mac GPU if MPS not explicitly available
                gpu_info_list.append(GPUInfo(
                    name="Apple GPU (Integrated or Discrete)",
                    memory_total_gb="N/A (Likely Shared with System Memory)",
                    notes="CUDA not available. Specific GPU details may require OS-specific tools."
                ))
        specs_dict['gpus'] = gpu_info_list
        
        await simple_ctx.info(f"Successfully gathered hardware specs for API: {specs_dict}")
        return SystemHardwareResponse(**specs_dict)

    except Exception as e:
        await simple_ctx.error(f"Error gathering system hardware specifications for API: {e}")
        # If some specs were gathered before error, include them
        if specs_dict: 
            specs_dict["error_partial_results"] = str(e)
            # Ensure all required fields for SystemHardwareResponse are present or None
            # This part can be tricky if the model expects non-None values not yet set
            # For simplicity, we'll let Pydantic handle missing fields if not explicitly set to None
            # and rely on the `error` field to indicate issues.
            response_with_error = SystemHardwareResponse(**specs_dict)
            if not response_with_error.error: # Add general error if not set by partial
                 response_with_error.error = f"Failed to retrieve some hardware specs: {str(e)}"
            return JSONResponse(status_code=500, content=response_with_error.model_dump(exclude_none=True))

        # If no specs gathered, return a simple error
        raise HTTPException(status_code=500, detail=f"Failed to retrieve hardware specs: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # It's good practice to ensure TEMP_AUDIO_DIR exists when running directly
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Temporary audio directory: {TEMP_AUDIO_DIR.resolve()}")
    logger.info(f"Temporary uploads directory: {TEMP_UPLOADS_DIR.resolve()}")
    
    # Note: Running with uvicorn.run() is mainly for development.
    # For production, use 'uvicorn api.main:app --host 0.0.0.0 --port 8000'
    uvicorn.run(app, host="127.0.0.1", port=8000) 