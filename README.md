[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/miguelspizza-local-transcription-mcp-parakeet-tdt-0-6b-v2-badge.png)](https://mseep.ai/app/miguelspizza-local-transcription-mcp-parakeet-tdt-0-6b-v2)

# Parakeet Transcription MCP Server

**Version:** 0.1.0

This is an MCP (Model Context Protocol) server designed to transcribe audio and video files into text using NVIDIA's powerful Parakeet TDT 0.6B V2 model. It also offers tools to get details about the model itself.

Built with FastMCP, this server relies on `pydub` (which requires FFmpeg) for handling audio conversions and `nemo_toolkit[asr]` for the core transcription capabilities.

**Important Notes:** FFmpeg must be installed and accessible in your system's PATH. All file paths provided to the server tools must be absolute.

## Quickstart

Here's a brief overview of the steps to get the server running:

1.  **Install Prerequisites:** Ensure you have `mise`, `uv`, and **FFmpeg** installed and accessible in your system's PATH. See the [Prerequisites](#prerequisites) section for details.
2.  **Clone the Repository:** If you haven't already, clone this repository and navigate into the project directory.
3.  **Set up Environment:** Use `mise` to install the correct Python version and activate the environment:
    ```bash
    mise install
    ```
4.  **Install Dependencies:** Use `uv` to install the required Python packages:
    ```bash
    uv pip install -r requirements.txt
    ```
5.  **Run the Server:** Start the MCP server using `fastmcp`:
    ```bash
    fastmcp run server.py
    ```
    The server will typically start using the STDIO transport. See [Running the Server](#running-the-server) for other options like HTTP.

Once the server is running, you can interact with it using an MCP-compatible client. See [Interacting with the Server (Client Usage)](#interacting-with-the-server-client-usage) for examples.

## About the ASR Model: NVIDIA Parakeet TDT 0.6B V2 (En)

This server utilizes the NVIDIA Parakeet TDT 0.6B V2 model, a FastConformer architecture with 600 million parameters. It's optimized for high-quality English transcription, featuring accurate word-level timestamps, automatic punctuation and capitalization, and robust performance on spoken numbers and song lyrics. It can efficiently transcribe audio segments up to 24 minutes in a single pass.

*   **Input:** 16kHz Audio (WAV or FLAC), Monochannel.
*   **Output:** Text with optional Punctuation, Capitalization, and Timestamps.
*   **License:** CC-BY-4.0.
*   **Demo & More Info:** [Hugging Face Spaces](https://huggingface.co/spaces/nvidia/parakeet-tdt-0.6b-v2)

While optimized for NVIDIA GPUs, the model will fall back to CPU if a compatible GPU isn't detected (note: CPU performance may be significantly slower).

## Features

*   Transcribe various audio/video formats.
*   Automatic conversion of input audio to the required 16kHz, mono WAV or FLAC format.
*   Option to include detailed word and segment timestamps.
*   Formatted transcription output with customizable line breaks when timestamps are included.
*   Retrieve information about the loaded ASR model.
*   Retrieve system hardware specifications (OS, CPU, RAM, GPU).

## Prerequisites

1.  **Python:** Version 3.12 (as specified in `.tool-versions` and `mise.toml`).
2.  **`mise`:** Used to manage Python versions (and other tools). Install `mise` by following the instructions on the [official `mise` documentation](https://mise.jdx.dev/getting-started.html).
3.  **`uv`:** An extremely fast Python package installer and resolver. Install `uv` by following instructions on the [Astral `uv` documentation](https://docs.astral.sh/uv/getting-started/installation/).
4.  **FFmpeg:** Required by `pydub` for audio and video file format conversions. FFmpeg must be installed and accessible in your system's PATH.

    *   **macOS (using Homebrew):**
        ```bash
        brew install ffmpeg
        ```
    *   **Linux (using apt - Debian/Ubuntu):**
        ```bash
        sudo apt update && sudo apt install ffmpeg
        ```
    *   **Linux (using yum - CentOS/RHEL/Fedora):**
        ```bash
        sudo yum install ffmpeg  # Or dnf for newer Fedora: sudo dnf install ffmpeg
        ```
    *   **Windows:**
        Download FFmpeg from the [official FFmpeg website](https://ffmpeg.org/download.html). Extract the archive and add the `bin` directory (containing `ffmpeg.exe`) to your system's PATH environment variable.
    *   **Verify FFmpeg installation:**
        Open a new terminal/command prompt and type:
        ```bash
        ffmpeg -version
        ```
        You should see version information if it's installed correctly.

## Setup and Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    # git clone https://github.com/MiguelsPizza/local-transcription-mcp--parakeet-tdt-0.6b-v2--.git
    # cd <repository-directory>
    ```

2.  **Set up Python version using `mise`:**
    Navigate to the project directory in your terminal and run:
    ```bash
    mise install
    ```
    This will ensure you are using Python 3.12 as specified in `.tool-versions`.

3.  **Install Python dependencies using `uv`:**
    Make sure your `mise` environment is active (it should be if you `cd` into the directory after `mise install`). Then run:
    ```bash
    uv pip install -r requirements.txt
    ```
    This will install `fastmcp`, `pydub`, `nemo_toolkit[asr]`, `psutil`, and other necessary packages.

## Running the Server

### MCP Server

```bash
fastmcp dev server.py
```
To run in production:
```bash
fastmcp run server.py
```

### FastAPI Server (for REST API)

To use the REST API, you'll need to install the additional dependencies first if you haven't already (this includes `fastapi`, `uvicorn`, and `python-multipart`):

```bash
# Ensure your mise environment is active
uv pip install -r requirements.txt 
```

Then, run the FastAPI application using Uvicorn from the project root:

```bash
# From the root of the project (transcription-mcp)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

*   `api.main:app` tells Uvicorn to look for an object named `app` in the `main.py` file inside the `api` directory.
*   `--host 0.0.0.0` makes the server accessible from other devices on your network.
*   `--port 8000` specifies the port to run on.
*   `--reload` enables auto-reloading when code changes, which is useful for development.

Once running, the API will be accessible at `http://localhost:8000` (or your machine's IP address on port 8000). Interactive API documentation (Swagger UI) will be available at `http://localhost:8000/docs`.

## Available Components (API)

The server exposes functionality through two interfaces: the Model Context Protocol (MCP) and a RESTful HTTP API.

### MCP Server Components

The following components are available via the MCP server (`server.py`):

#### Tools

#### 1. `transcribe_audio`

*   **Description:** Transcribes an audio/video file to text using the Parakeet TDT 0.6B V2 model.
*   **Parameters:**
    *   `audio_file_path` (string, **absolute path**, required): The absolute path to the audio or video file to be transcribed.
    *   `output_format` (string, optional, default: `"wav"`): The intermediate audio format to convert the input file to before transcription. Supported values: `"wav"`, `"flac"`.
    *   `include_timestamps` (boolean, optional, default: `True`): Whether to include word and segment level timestamps in the transcription output.
    *   `line_character_limit` (integer, optional, default: `80`, min: 40, max: 200): The character limit per line for formatted transcription output when timestamps are included.
    *   `segment_length_minutes` (integer, optional, default: `5`, min: 1, max: 24): Maximum length of audio segments in minutes. Audio longer than this will be split.
*   **Returns:** A JSON object containing:
    *   `message` (string): A status message indicating the outcome of the transcription.
    *   `file_processed` (string): The original `audio_file_path` that was processed.
    *   `transcription` (string): The transcribed text, potentially formatted with timestamps.

#### 2. System Hardware Specifications

*   **URI:** `info://system_hardware_specs`
*   **Name:** `system_hardware_specifications`
*   **Description:** Retrieves system hardware specifications relevant for performance estimation, such as OS, CPU, RAM, and GPU details.
*   **Returns:** A JSON object containing system hardware details (see `server.py` for full structure).

### REST API Endpoints (FastAPI)

The following endpoints are available via the FastAPI server (`api/main.py`), typically running on `http://localhost:8000`:

#### 1. Transcribe Audio

*   **Endpoint:** `POST /transcribe/`
*   **Description:** Transcribes an uploaded audio or video file.
*   **Request Type:** `multipart/form-data`
*   **Form Fields:**
    *   `file` (File, required): The audio or video file to be transcribed.
    *   `output_format` (string, optional, default: `"wav"`): Intermediate audio format (`"wav"` or `"flac"`).
    *   `include_timestamps` (boolean, optional, default: `True`): Whether to include word/segment timestamps.
    *   `line_character_limit` (integer, optional, default: `80`, min: 40, max: 200): Character limit per line for timestamped output.
    *   `segment_length_minutes` (integer, optional, default: `5`, min: 1, max: 24): Max audio segment length in minutes.
*   **Success Response (200 OK):**
    ```json
    {
      "message": "Transcription successful with formatted timestamps.",
      "file_processed": "your_audio_file.mp3",
      "transcription": "The transcribed text..."
    }
    ```
*   **Error Response (e.g., 400, 422, 500):** JSON object with a `detail` field describing the error.

#### 2. ASR Model Information

*   **Endpoint:** `GET /info/asr-model/`
*   **Description:** Provides detailed information about the ASR model.
*   **Success Response (200 OK):**
    ```json
    {
      "model_name": "NVIDIA Parakeet TDT 0.6B V2 (En)",
      "status": "Loaded",
      "input_requirements": "16kHz Audio (.wav or .flac), Monochannel",
      "output_type": "Text with optional Punctuation, Capitalization, and Timestamps.",
      "license": "CC-BY-4.0",
      "note": "This model is optimized for NVIDIA GPU-accelerated systems."
    }
    ```

#### 3. System Hardware Specifications

*   **Endpoint:** `GET /info/system-hardware/`
*   **Description:** Retrieves system hardware specifications.
*   **Success Response (200 OK):** A JSON object containing system hardware details (see `api/main.py` `SystemHardwareResponse` model for full structure).
    ```json
    // Example structure (fields may vary based on system)
    {
      "os_platform": "Darwin",
      "os_version": "...",
      "cpu_model": "Apple M1 Pro",
      "ram_total_gb": 16.0,
      "cuda_available": false,
      "gpus": [
        {
          "name": "Apple Metal Performance Shaders (MPS)",
          "memory_total_gb": "N/A (Shared with System Memory)",
          "notes": "MPS is available for PyTorch on this Mac."
        }
      ]
      // ... other fields ...
    }
    ```

## Recommended Workflow for Transcription

This workflow applies to both MCP and REST API usage, adapting the component names/calls as needed.

1.  **Read Hardware Specs:** Use `info://system_hardware_specs` (MCP) or `GET /info/system-hardware/` (API) to get hardware details.
2.  **Determine Segment Length:** Based on hardware, choose an optimal `segment_length_minutes`.
3.  **Transcribe Audio:** Call `transcribe_audio` (MCP) or `POST /transcribe/` (API) with the audio file and chosen parameters.
4.  **(Optional) Get Model Info:** Use `info://asr_model` (MCP) or `GET /info/asr-model/` (API) for model details.

## Adding to MCP Client Hosts

You can configure MCP clients (like Claude Desktop or other tools that support custom MCP server definitions) to use this server.

### Manual JSON Configuration

For clients that use a JSON configuration file (e.g., `cline_mcp_settings.json` or similar) to define MCP servers, you can add an entry for this transcription server. Ensure that you have completed the "Setup and Installation" steps above so that Python 3.12 and all dependencies are available in your environment when the client attempts to run the server.

Here's an example configuration snippet:

```json
{
  "mcpServers": {
    "transcription-mcp": {
      "autoApprove": [],
      "disabled": true,
      "timeout": 600,
      "command": "uv",
      "args": [
        "run",
        "--with",
        "fastmcp",
        "--with",
        "nemo_toolkit[asr]",
        "--with",
        "pydub",
        "psutil",
        "fastmcp",
        "run",
        "/absolute/path/to/this/file/server.py"
      ],
      "env": {},
      "transportType": "stdio"
    }
    // ... other server configurations ...
  }
}
```

### Using `fastmcp install` (for supported clients)

Some MCP clients, like recent versions of the Claude Desktop App, integrate with the `fastmcp install` command. This can simplify setup by creating an isolated environment for the server. If your client supports this, you can install the server from the root directory of this project using:

```bash
fastmcp install server.py -e . -n "Parakeet Transcription Server"
```

*   `-e .`: Installs the current directory (which should contain `pyproject.toml`) in editable mode. The `pyproject.toml` file lists the core dependencies (`fastmcp`, `pydub`, `nemo_toolkit[asr]`, `psutil`), which `fastmcp install` should pick up.
*   `-n "Parakeet Transcription Server"`: Sets a custom name for the server in the client application.

This command will typically handle packaging the server and its specified dependencies for use by the client.

## Interacting with the Server (Client Usage)

You can interact with this MCP server using any FastMCP-compatible client. Here's a basic Python example using the `fastmcp` library:

```python
import asyncio
from fastmcp import Client

# If running the MCP server with 'fastmcp run server.py' (defaulting to STDIO):
client = Client("server.py")

# If running the server with HTTP, e.g., 'fastmcp run server.py --transport streamable-http --port 8000':
# client = Client("http://localhost:8000/mcp")

# If you've added it to your client's host configuration (e.g., Claude Desktop)
# and the client library allows referencing by name/ID:
# client = Client(mcp_server_id="parakeet-transcription-server-local") # Syntax depends on client library

async def main():
    async with client:
        print(f"Client connected: {client.is_connected()}")

        # Example 1: Get ASR Model Information
        try:
            print("\nFetching ASR model info...")
            model_info_result = await client.call_tool("get_asr_model_info")
            # Assuming the result is a JSON string in the first TextContent part
            model_info_dict = model_info_result[0].text_content_as_json_dict()
            print("ASR Model Info:")
            for key, value in model_info_dict.items():
                 print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error calling get_asr_model_info: {e}")

        # Example 2: Transcribe an audio file (replace with an ACTUAL absolute path)
        # Ensure the audio file exists and the path is absolute.
        audio_file_to_transcribe = "/Users/yourname/path/to/your/audio.mp3" # <<< REPLACE THIS
        #
        if audio_file_to_transcribe != "/Users/yourname/path/to/your/audio.mp3": # Basic check
            try:
                print(f"\nTranscribing '{audio_file_to_transcribe}'...")
                transcription_args = {
                    "audio_file_path": audio_file_to_transcribe,
                    "include_timestamps": True,
                    "output_format": "wav" # or "flac"
                }
                transcription_result = await client.call_tool("transcribe_audio", transcription_args)
                # Assuming the result is a JSON string in the first TextContent part
                result_data = transcription_result[0].text_content_as_json_dict()
                print(f"File Processed: {result_data.get('file_processed')}")
                print(f"Message: {result_data.get('message')}")
                print("Transcription Output:")
                print(result_data.get('transcription'))
            except Exception as e:
                print(f"Error calling transcribe_audio: {e}")
        else:
            print("\nPlease update 'audio_file_to_transcribe' with an actual absolute file path to test transcription.")

    print(f"\nClient disconnected: {client.is_connected()}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Note:** For the transcription example, make sure to replace `