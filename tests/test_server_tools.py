import pytest
from fastmcp.exceptions import ToolError
from src.models.transcription_models import TranscriptionResult, WordTimestamp
import os

# Marks all tests in this file as async
pytestmark = pytest.mark.asyncio

async def test_list_tools(mcp_client):
    """Test that the client can connect and list tools from the server."""
    tools = await mcp_client.list_tools()
    assert len(tools) > 0
    tool_names = [tool.name for tool in tools]
    assert "transcribe_audio" in tool_names
    assert "get_asr_model_info" in tool_names

async def test_transcribe_audio_file_not_found(mcp_client):
    """Test transcribe_audio tool with a non-existent audio file path."""
    non_existent_path = "/path/to/non_existent_audio.wav"
    # Ensure the path is absolute and non-existent
    assert os.path.isabs(non_existent_path)
    assert not os.path.exists(non_existent_path)

    with pytest.raises(ToolError) as exc_info:
        await mcp_client.call_tool(
            "transcribe_audio",
            arguments={"audio_file_path": non_existent_path}
        )
    assert "File not found" in str(exc_info.value)

async def test_transcribe_audio_relative_path(mcp_client):
    """Test transcribe_audio tool with a relative audio file path."""
    relative_path = "some_relative_audio.wav"
    assert not os.path.isabs(relative_path)

    with pytest.raises(ToolError) as exc_info:
        await mcp_client.call_tool(
            "transcribe_audio",
            arguments={"audio_file_path": relative_path}
        )
    assert "Please provide an absolute path" in str(exc_info.value)

async def test_transcribe_audio_success_with_timestamps(
    mcp_client, dummy_audio_file_path, mocker
):
    """Test successful transcription with timestamps."""
    mock_converted_path = "/tmp/fake_converted.wav"
    mock_transcription_result = TranscriptionResult(
        text="Hello world.",
        word_timestamps=[
            WordTimestamp(word="Hello", start_time=0.1, end_time=0.5),
            WordTimestamp(word="world.", start_time=0.6, end_time=1.0),
        ]
    )

    mocker.patch("src.audio_converter.convert_to_audio", return_value=mock_converted_path)
    mocker.patch("src.transcriber.transcribe_audio_file", return_value=mock_transcription_result)
    mocker.patch("src.utils.file_utils.cleanup_temp_files", return_value=None) # Mock cleanup
    # Mock os.path.exists for the converted_audio_path if cleanup logic checks it before deleting
    mocker.patch("os.path.exists", lambda path: True if path == mock_converted_path else os.path.exists(path))
    mocker.patch("os.path.isfile", lambda path: True if path == mock_converted_path else os.path.isfile(path))
    mocker.patch("os.remove") # Mock os.remove for cleanup

    result = await mcp_client.call_tool(
        "transcribe_audio",
        arguments={
            "audio_file_path": str(dummy_audio_file_path),
            "include_timestamps": True,
            "line_character_limit": 80 # Default, but explicit for clarity
        }
    )
    
    assert isinstance(result, dict)
    assert result["message"] == "Transcription successful with formatted timestamps."
    assert "[0.10s] Hello [0.60s] world." in result["transcription"] # Based on _format_transcription_output logic

async def test_transcribe_audio_success_no_timestamps(
    mcp_client, dummy_audio_file_path, mocker
):
    """Test successful transcription without timestamps."""
    mock_converted_path = "/tmp/fake_converted_no_ts.wav"
    mock_transcription_text = "This is a test without timestamps."

    mocker.patch("src.audio_converter.convert_to_audio", return_value=mock_converted_path)
    mocker.patch("src.transcriber.transcribe_audio_file", return_value=mock_transcription_text)
    mocker.patch("src.utils.file_utils.cleanup_temp_files", return_value=None)
    mocker.patch("os.path.exists", lambda path: True if path == mock_converted_path else os.path.exists(path))
    mocker.patch("os.path.isfile", lambda path: True if path == mock_converted_path else os.path.isfile(path))
    mocker.patch("os.remove")

    result = await mcp_client.call_tool(
        "transcribe_audio",
        arguments={
            "audio_file_path": str(dummy_audio_file_path),
            "include_timestamps": False,
        }
    )
    
    assert isinstance(result, dict)
    assert result["message"] == "Transcription successful (text only)."
    assert result["transcription"] == mock_transcription_text

async def test_transcribe_audio_conversion_fails(mcp_client, dummy_audio_file_path, mocker):
    """Test transcribe_audio when audio conversion fails."""
    mocker.patch("src.audio_converter.convert_to_audio", return_value=None)
    mocker.patch("src.utils.file_utils.cleanup_temp_files", return_value=None) # ensure it's mocked even if not reached

    with pytest.raises(ToolError) as exc_info:
        await mcp_client.call_tool(
            "transcribe_audio",
            arguments={"audio_file_path": str(dummy_audio_file_path)}
        )
    assert "Failed to convert file" in str(exc_info.value)

async def test_transcribe_audio_transcription_call_fails_returns_none(
    mcp_client, dummy_audio_file_path, mocker
):
    """Test transcribe_audio when the transcribe_audio_file call returns None."""
    mock_converted_path = "/tmp/fake_converted_trans_none.wav"
    mocker.patch("src.audio_converter.convert_to_audio", return_value=mock_converted_path)
    mocker.patch("src.transcriber.transcribe_audio_file", return_value=None)
    mocker.patch("src.utils.file_utils.cleanup_temp_files", return_value=None)
    mocker.patch("os.path.exists", lambda path: True if path == mock_converted_path else os.path.exists(path))
    mocker.patch("os.path.isfile", lambda path: True if path == mock_converted_path else os.path.isfile(path))
    mocker.patch("os.remove")

    with pytest.raises(ToolError) as exc_info:
        await mcp_client.call_tool(
            "transcribe_audio",
            arguments={"audio_file_path": str(dummy_audio_file_path)}
        )
    assert "Transcription failed for" in str(exc_info.value)
    assert "(returned None)" in str(exc_info.value)

async def test_transcribe_audio_transcription_call_raises_exception(
    mcp_client, dummy_audio_file_path, mocker
):
    """Test transcribe_audio when transcribe_audio_file raises an unexpected exception."""
    mock_converted_path = "/tmp/fake_converted_trans_exc.wav"
    mocker.patch("src.audio_converter.convert_to_audio", return_value=mock_converted_path)
    mocker.patch("src.transcriber.transcribe_audio_file", side_effect=ValueError("Nemo exploded"))
    mocker.patch("src.utils.file_utils.cleanup_temp_files", return_value=None)
    mocker.patch("os.path.exists", lambda path: True if path == mock_converted_path else os.path.exists(path))
    mocker.patch("os.path.isfile", lambda path: True if path == mock_converted_path else os.path.isfile(path))
    mocker.patch("os.remove")

    with pytest.raises(ToolError) as exc_info:
        await mcp_client.call_tool(
            "transcribe_audio",
            arguments={"audio_file_path": str(dummy_audio_file_path)}
        )
    assert "An unexpected server error occurred: Error: An unexpected server error occurred: Nemo exploded" in str(exc_info.value)

# Tests for get_asr_model_info
async def test_get_asr_model_info_success(mcp_client, mocker):
    """Test get_asr_model_info when model loads successfully."""
    # Mock load_model to return a mock model object (anything not None)
    mock_model = mocker.Mock()
    mocker.patch("src.transcriber.load_model", return_value=mock_model)
    
    result = await mcp_client.call_tool("get_asr_model_info", arguments={})
    
    assert isinstance(result, dict)
    assert result["model_name"] == "nvidia/parakeet-tdt-0.6b-v2"
    assert result["status"] == "Loaded"

async def test_get_asr_model_info_load_fails(mcp_client, mocker):
    """Test get_asr_model_info when model loading fails (returns None)."""
    mocker.patch("src.transcriber.load_model", return_value=None)
    
    result = await mcp_client.call_tool("get_asr_model_info", arguments={})
    
    assert isinstance(result, dict)
    assert result["model_name"] == "nvidia/parakeet-tdt-0.6b-v2"
    assert "Error loading model" in result["status"] 