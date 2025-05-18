# Parakeet Transcription MCP Server

This MCP server takes an MP4 file, converts it to WAV or FLAC audio (16kHz mono),
and then transcribes the audio to text using the NVIDIA Parakeet TDT 0.6B V2 model.
The server is built using FastMCP 2.0+.

## Features

- Converts MP4 to WAV or FLAC
- Transcribes audio using Parakeet TDT 0.6B V2
- Supports word-level and segment-level timestamps
- Provides information about the ASR model
- Model pre-loading on server startup
- Temporary file management


## Model Used

- **Name:** `nvidia/parakeet-tdt-0.6B-v2`
- **Description:** 600-million-parameter ASR model for English transcription with punctuation, capitalization, and timestamp prediction
- **Input:** 16kHz mono WAV or FLAC audio
- **Output:** Text string (with punctuation and capitalization), optional timestamps
- **License:** CC-BY-4.0

## Prerequisites

1. **Python:** 3.9+ (Managed by Mise)
2. **Mise (formerly RTX):** For Python version and project dependency management.
   - Installation: [https://mise.jdx.dev/getting-started.html](https://mise.jdx.dev/getting-started.html)
3. **uv (Python Package Installer):** A fast Python package installer and resolver, used for installing project dependencies.
   - Installation: [https://github.com/astral-sh/uv#installation](https://github.com/astral-sh/uv#installation)
4. **FFmpeg:** Required by `pydub` for audio conversion. Ensure it's installed and in your system's PATH.
   - Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
   - Ensure that FFmpeg is added to your system's PATH environment variable.
5. **NVIDIA GPU (Recommended):** For optimal performance, an NVIDIA GPU with CUDA drivers installed is highly recommended. The model will run on CPU but will be significantly slower.
6. **NVIDIA NeMo Toolkit & PyTorch:** Specific versions might be required. Check the Parakeet model card and NeMo documentation for compatibility.

## Installation

1.  **Install Mise:** If you haven't already, install Mise by following the instructions [here](https://mise.jdx.dev/getting-started.html).
2.  **Install uv:** If you haven't already, install uv by following its [installation guide](https://github.com/astral-sh/uv#installation).

3.  **Set up Python with Mise:**
    Navigate to the project directory. Mise uses the `.tool-versions` file to determine the Python version (e.g., `python 3.12`). If this file doesn't exist, you can create it or use `mise use python@3.12` (replace `3.12` with your desired compatible version). Mise will automatically download and install the specified Python version if it's not already available.
    ```bash
    # Ensure .tool-versions file has a line like:
    # python 3.12
    # Or run:
    mise use python@3.12 # Or your desired version e.g. python@3.10, python@3.11
    ```
    Mise will automatically manage your environment when you `cd` into the project directory.

4.  **Install dependencies with uv:**
    Once Mise has configured your Python environment for the project, install the dependencies using `uv`:
    ```bash
    uv pip install -r requirements.txt
    ```
    `uv` will create and manage a virtual environment for you by default (typically in `.venv` in your project root) if one isn't already activated.
    *Note: Installing `nemo_toolkit[asr]` and `python-json-logger` can take some time as it pulls many dependencies, including PyTorch if not already installed or if a specific version is required. `uv` should handle this process efficiently.*

## Running the Server

You can run the server using the FastMCP CLI (which is installed as part of `fastmcp`) or directly with Python.

1.  **Direct Execution (Recommended for local development):**
    These commands run the server in your current Mise-managed environment.
    *   Using FastMCP CLI (stdio transport by default):
        ```bash
        fastmcp run server.py
        ```
    *   Using Python directly (uses stdio transport if `mcp.run()` is called without arguments in `server.py`):
        ```bash
        python server.py
        ```
    For other transports like HTTP:
    ```bash
    fastmcp run server.py --transport streamable-http --port 8000
    ```

2.  **Development with MCP Inspector (`fastmcp dev`):
    This command runs your server in an isolated environment with the MCP Inspector, similar to how `uvx` might run a packaged tool. It's for testing over STDIO only.
    ```bash
    fastmcp dev server.py --with-editable . --with pydub --with "nemo_toolkit[asr]" --with python-json-logger
    ```
    *   **Important:** `fastmcp dev` creates an isolated environment. You **must** explicitly specify all dependencies using `--with` for packages from PyPI (like `pydub`, `"nemo_toolkit[asr]"`, `python-json-logger`) and `--with-editable .` to include your local project files.
    *   The `--with-editable .` flag requires a `pyproject.toml` (or `setup.py`) file in your project root to recognize it as a package.
    *   The MCP Inspector will launch, and you may need to select "STDIO" and connect manually.

3.  **Installation for Claude Desktop-like environments (`fastmcp install`):
    This command also creates an isolated environment, similar to `uvx` packaging a tool for execution. It's primarily for STDIO transport.
    ```bash
    fastmcp install server.py --name "ParakeetTranscription" --with-editable . --with pydub --with "nemo_toolkit[asr]" --with python-json-logger
    ```
    *   Like `fastmcp dev`, you must specify all dependencies using `--with` and `--with-editable`.
    *   The `--with-editable .` flag requires a `pyproject.toml` (or `setup.py`) file in your project root.

*Note on `uvx` itself: While `fastmcp dev` and `fastmcp install` provide `uvx`-like isolated environments for your server, directly running this local development server *with* `uvx` (e.g., `uvx run python server.py`) is not the standard workflow. `uvx` is typically used to run Python applications that are packaged and distributed, often with a defined entry point. The FastMCP client also has a `UvxStdioTransport` for running *packaged* MCP servers as tools, which is a different scenario.*

## How to Call the Server (Client Example)

A Python client script `client_example.py` is provided to demonstrate how to interact with the server.

1.  **Ensure the server is running** (e.g., `python server.py` or `fastmcp run server.py` in one terminal).
2.  **Modify `client_example.py`**:
    *   Update the `mp4_path` variable in `client_example.py` to point to the **absolute path** of an MP4 file you want to transcribe.
3.  **Run the client script** in another terminal:
    ```bash
    python client_example.py
    ```

## Available MCP Tools

### 1. `transcribe_audio`

Transcribes an audio/video file to text.

- **Description:** Takes the path to an audio/video file, converts it to WAV or FLAC, and returns the transcription.
- **Arguments:**
  - `audio_file_path` (str, required): Absolute path to the audio/video file.
  - `output_format` (str, optional): Intermediate audio format (`"wav"` or `"flac"`). Default: `"wav"`.
  - `include_timestamps` (bool, optional): Whether to include word and segment timestamps. Default: `True`.
- **Returns:** A JSON object with the transcription, timestamps (if requested), and status, or an error message.

**Example CLI call (after starting the server with `fastmcp run server.py`):**
```bash
# Basic transcription
fastmcp call transcribe_audio '{"audio_file_path": "/path/to/your/audio.mp4"}'

# With FLAC intermediate and no timestamps
fastmcp call transcribe_audio '{"audio_file_path": "/path/to/your/audio.mp4", "output_format": "flac", "include_timestamps": false}'
```

### 2. `get_asr_model_info`

Provides information about the ASR model being used.

- **Description:** Returns details about the loaded Parakeet ASR model.
- **Arguments:** None.
- **Returns:** A JSON object with model information.

**Example CLI call:**
```bash
fastmcp call get_asr_model_info '{}'
```

## Directory Structure
```
transcription-mcp/
├── server.py                 # Main MCP server script
├── client_example.py         # Example client script to call the server
├── audio_converter.py        # Module for MP4 to WAV/FLAC conversion
├── transcriber.py            # Module for using the Parakeet model
├── .tool-versions            # Specifies Python version for Mise
├── pyproject.toml            # Project definition for Python packaging
├── requirements.txt          # Project dependencies
├── README.md                 # This file
├── temp_uploads/             # (Created at runtime) Temporary storage for uploads
└── temp_audio/               # (Created at runtime) Temporary storage for converted audio files
```

## Important Notes

- **Model Download:** The first time ASRModel.from_pretrained() is called for nvidia/parakeet-tdt-0.6B-v2, the model weights (which can be large) will be downloaded. This might take some time and requires an internet connection. Subsequent runs will use the cached model.
- **GPU Memory:** The Parakeet 0.6B model requires at least 2GB of RAM (GPU RAM preferably for good performance). Ensure your system meets these requirements.
- **Error Handling:** The server includes error handling and uses `ToolError` for client-facing errors. Check the server logs for detailed internal error messages.
- **File Paths in transcribe_audio:** The `audio_file_path` argument **must** be an absolute path to the audio/video file on the server's filesystem. This is a strict requirement.


run --with fastmcp --with nemo_toolkit[asr] --with pydub --with python-json-logger fastmcp run server.py
