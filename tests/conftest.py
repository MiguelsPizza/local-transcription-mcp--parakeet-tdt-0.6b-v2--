import pytest
import asyncio
import os
import shutil
from pathlib import Path
from pydub import AudioSegment
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport

# This import assumes your server.py is in the parent directory of 'tests'
# and your workspace root is the parent of 'src'.
from server import mcp as server_mcp_instance
from src.utils.file_utils import TEMP_BASE_DIR

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def mcp_server_main():
    """Fixture to provide the main MCP server instance from server.py."""
    # If the server has a lifespan manager that needs to run, handle it here.
    # For now, we assume direct usage of the mcp instance is fine for testing.
    # User removed lifespan manager, so this is simpler.
    if not os.path.exists(TEMP_BASE_DIR):
        os.makedirs(TEMP_BASE_DIR) # Ensure base temp dir exists
    yield server_mcp_instance
    # Cleanup temp dirs after session if they were created by tests/server
    if os.path.exists(TEMP_BASE_DIR):
        shutil.rmtree(TEMP_BASE_DIR)


@pytest.fixture
async def mcp_client(mcp_server_main):
    """Fixture to provide an MCP client connected to the in-memory server."""
    # Use FastMCPTransport for in-memory testing
    transport = FastMCPTransport(server_mcp_instance)
    client = Client(transport=transport)
    async with client:
        yield client
    # Connection is closed automatically by async with client

@pytest.fixture(scope="module")
def dummy_audio_file_path() -> Path:
    """Creates a dummy WAV audio file for testing and yields its path."""
    temp_test_files_dir = Path("temp_test_files")
    temp_test_files_dir.mkdir(exist_ok=True)
    
    # Create a short silent WAV file
    silence = AudioSegment.silent(duration=1000) # 1 second
    file_path = temp_test_files_dir / "test_audio.wav"
    silence.export(file_path, format="wav")
    
    yield file_path.resolve() # Ensure absolute path
    
    # Teardown: remove the temp file and directory
    os.remove(file_path)
    try:
        os.rmdir(temp_test_files_dir) # Remove if empty
    except OSError:
        pass # Not empty, or other issue, fine for test teardown

# Ensure pytest-asyncio is installed and configured if you see issues with async fixtures/tests 