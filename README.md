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

```bash
fastmcp dev server.py
```
To run in production:
```bash
fastmcp run server.py
```

## Available Components (API)

The server exposes the following components through the Model Context Protocol:

### Tools

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

### Resources

#### 1. ASR Model Information

*   **URI:** `info://asr_model`
*   **Name:** `asr_model_information`
*   **Description:** Provides detailed information about the ASR model being used (NVIDIA Parakeet TDT 0.6B V2).
*   **Returns:** A JSON object containing model details such as:
    *   `model_name` (string)
    *   `status` (string): "Loaded" or an error message.
    *   `input_requirements` (string)
    *   `output_type` (string)
    *   `license` (string)
    *   `note` (string)

#### 2. System Hardware Specifications

*   **URI:** `info://system_hardware_specs`
*   **Name:** `system_hardware_specifications`
*   **Description:** Retrieves system hardware specifications relevant for performance estimation, such as OS, CPU, RAM, and GPU details.
*   **Returns:** A JSON object containing:
    *   `os_platform` (string): Operating system platform (e.g., "Linux", "Darwin", "Windows").
    *   `os_version` (string): OS version.
    *   `os_release` (string): OS release.
    *   `architecture` (string): System architecture (e.g., "x86_64", "arm64").
    *   `cpu_model` (string): CPU model name.
    *   `cpu_physical_cores` (integer): Number of physical CPU cores.
    *   `cpu_logical_cores` (integer): Number of logical CPU cores (threads).
    *   `cpu_frequency_max_ghz` (float/string): Maximum CPU frequency in GHz (or "N/A").
    *   `ram_total_gb` (float): Total system RAM in Gigabytes.
    *   `ram_available_gb` (float): Available system RAM in Gigabytes.
    *   `cuda_available` (boolean): True if an NVIDIA GPU with CUDA is detected by PyTorch.
    *   `cuda_version` (string, optional): CUDA version if available.
    *   `gpu_count` (integer): Number of detected GPUs (primarily NVIDIA GPUs via CUDA).
    *   `gpus` (list of objects): Detailed information for each detected GPU.
        *   `name` (string): GPU name (e.g., "NVIDIA GeForce RTX 3080", "Apple Metal Performance Shaders (MPS)").
        *   `memory_total_gb` (float/string): Total GPU memory in Gigabytes (or "N/A" or descriptive string for integrated/shared memory).
        *   `cuda_capability` (tuple, optional): CUDA compute capability (e.g., `(8, 6)`), if applicable.
        *   `notes` (string, optional): Additional notes, e.g., about MPS availability.
    *   `error` (string, optional): Present if there was an error retrieving some or all specs.
    *   `error_partial_results` (string, optional): Present if an error occurred after some specs were already gathered.

## Recommended Workflow for Transcription

1.  **Read Hardware Specs:** Use the `info://system_hardware_specs` resource to get details about the server's hardware.
2.  **Determine Segment Length:** Based on the hardware (especially GPU availability and RAM), decide on an optimal `segment_length_minutes` for the `transcribe_audio` tool.
    *   Powerful NVIDIA GPU: Can use longer segments (up to 24 mins).
    *   No NVIDIA GPU / Low RAM: Use shorter segments (e.g., 1-10 mins, default is 5 mins).
3.  **Transcribe Audio:** Call the `transcribe_audio` tool with the `audio_file_path` and your chosen `segment_length_minutes`.
4.  **(Optional) Get Model Info:** Read the `info://asr_model` resource if you need details about the ASR model.

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

# If running the server with 'fastmcp run server.py' (defaulting to STDIO):
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