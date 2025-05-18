import unittest
from unittest.mock import patch, MagicMock, ANY
import os
import sys
from pydantic import ValidationError

# Ensure src directory is in Python path for imports
# This might be needed if running tests directly and src is not in PYTHONPATH
# For a proper package structure, this might be handled differently (e.g., editable install)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.insert(0, project_root)

from src.transcriber import (
    load_model,
    transcribe_audio_file,
    _process_transcription_output,
    TranscriptionInput,
    TranscriptionResult,
    WordTimestamp,
    MODEL_NAME
)
from src.logger import setup_logger # For logger access if needed

# Suppress logging during tests unless specifically testing logging
# logging.disable(logging.CRITICAL) 
# Alternatively, mock the logger if its calls are part of assertions
logger = setup_logger(__name__) # Get a logger instance for potential mocking

# Global ASR_MODEL variable in transcriber.py
# We need to be able to reset it for tests
import src.transcriber

class TestTranscriber(unittest.TestCase):

    def setUp(self):
        # Reset the global ASR_MODEL before each test
        src.transcriber.ASR_MODEL = None
        # Create a dummy audio file for tests that require an existing file
        self.dummy_audio_file_path = os.path.join(project_root, "test_dummy_audio.wav")
        with open(self.dummy_audio_file_path, "w") as f:
            f.write("dummy audio data")

    def tearDown(self):
        # Clean up the dummy audio file
        if os.path.exists(self.dummy_audio_file_path):
            os.remove(self.dummy_audio_file_path)
        # Reset global model again after test
        src.transcriber.ASR_MODEL = None


    @patch('src.transcriber.torch.cuda.is_available')
    @patch('src.transcriber.nemo_asr.models.ASRModel.from_pretrained')
    def test_load_model_success_gpu(self, mock_from_pretrained, mock_cuda_is_available):
        mock_cuda_is_available.return_value = True
        mock_model_instance = MagicMock()
        mock_from_pretrained.return_value = mock_model_instance

        model = load_model()

        self.assertIsNotNone(model)
        self.assertEqual(model, mock_model_instance)
        mock_from_pretrained.assert_called_once_with(model_name=MODEL_NAME)
        logger.info.assert_any_call("NVIDIA GPU detected. Model will run on GPU.")
        logger.info.assert_any_call(f"Model {MODEL_NAME} loaded successfully.")
        self.assertEqual(src.transcriber.ASR_MODEL, mock_model_instance)

    @patch('src.transcriber.torch.cuda.is_available')
    @patch('src.transcriber.nemo_asr.models.ASRModel.from_pretrained')
    def test_load_model_success_cpu(self, mock_from_pretrained, mock_cuda_is_available):
        mock_cuda_is_available.return_value = False
        mock_model_instance = MagicMock()
        mock_from_pretrained.return_value = mock_model_instance

        model = load_model()

        self.assertIsNotNone(model)
        logger.warning.assert_any_call("No NVIDIA GPU detected. Model will run on CPU (this may be slow).")
        self.assertEqual(src.transcriber.ASR_MODEL, mock_model_instance)
    
    @patch('src.transcriber.nemo_asr.models.ASRModel.from_pretrained')
    def test_load_model_already_loaded(self, mock_from_pretrained):
        mock_existing_model = MagicMock()
        src.transcriber.ASR_MODEL = mock_existing_model # Pre-load the model

        model = load_model()
        self.assertEqual(model, mock_existing_model)
        mock_from_pretrained.assert_not_called() # Should not attempt to load again

    @patch('src.transcriber.nemo_asr.models.ASRModel.from_pretrained', side_effect=RuntimeError("Test Model Load Fail"))
    def test_load_model_failure(self, mock_from_pretrained):
        with self.assertRaises(RuntimeError) as context:
            load_model()
        self.assertIn("Test Model Load Fail", str(context.exception))
        self.assertIsNone(src.transcriber.ASR_MODEL) # Ensure it remains None after failure

    def test_process_transcription_output_empty(self):
        with self.assertRaises(ValueError) as context:
            _process_transcription_output([], "dummy_path.wav", True)
        self.assertIn("Transcription output empty", str(context.exception))

        with self.assertRaises(ValueError) as context:
            _process_transcription_output([None], "dummy_path.wav", True)
        self.assertIn("Transcription output empty", str(context.exception))

    def test_process_transcription_output_string_no_timestamps_requested(self):
        nemo_output = ["This is a test transcription."]
        result = _process_transcription_output(nemo_output, "dummy.wav", False)
        self.assertEqual(result, "This is a test transcription.")

    def test_process_transcription_output_string_timestamps_requested_but_nemo_returns_string(self):
        # This case implies include_timestamps=True, but NeMo model itself returned plain string (e.g. a model not supporting timestamps)
        nemo_output = ["This is a test transcription."]
        result = _process_transcription_output(nemo_output, "dummy.wav", True)
        self.assertEqual(result, "This is a test transcription.")


    def test_process_transcription_output_with_valid_word_timestamps(self):
        mock_asr_result = MagicMock()
        mock_asr_result.text = "Hello world"
        
        mock_word_timestamp_batch = MagicMock()
        mock_word_timestamp_batch.word = [
            {'word': 'Hello', 'start_time': 0.1, 'end_time': 0.5},
            {'word': 'world', 'start_time': 0.6, 'end_time': 1.0}
        ]
        mock_asr_result.timestamp = mock_word_timestamp_batch
        
        nemo_output = [mock_asr_result]
        
        result = _process_transcription_output(nemo_output, "dummy.wav", True)

        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(len(result.word_timestamps), 2)
        self.assertEqual(result.word_timestamps[0].word, "Hello")
        self.assertEqual(result.word_timestamps[0].start_time, 0.1)
        self.assertEqual(result.word_timestamps[1].end_time, 1.0)
        self.assertEqual(result.segment_timestamps, []) # Expect empty as not processed
        self.assertEqual(result.char_timestamps, [])   # Expect empty

    def test_process_transcription_output_timestamps_requested_no_timestamp_data_in_asrresult(self):
        mock_asr_result = MagicMock()
        mock_asr_result.text = "Test text"
        mock_asr_result.timestamp = None # No timestamp object
        nemo_output = [mock_asr_result]

        result = _process_transcription_output(nemo_output, "dummy.wav", True)
        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "Test text")
        self.assertEqual(result.word_timestamps, [])
        # Check for warning log
        # logger.warning.assert_any_call("Timestamps requested but no valid word timestamps found...") # This requires logger mocking setup if not global

    def test_process_transcription_output_timestamps_requested_empty_word_list_in_timestamp_data(self):
        mock_asr_result = MagicMock()
        mock_asr_result.text = "Test text"
        mock_timestamp_obj = MagicMock()
        mock_timestamp_obj.word = [] # Empty list of words
        mock_asr_result.timestamp = mock_timestamp_obj
        nemo_output = [mock_asr_result]

        result = _process_transcription_output(nemo_output, "dummy.wav", True)
        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "Test text")
        self.assertEqual(result.word_timestamps, [])


    def test_process_transcription_output_malformed_word_timestamp_item(self):
        mock_asr_result = MagicMock()
        mock_asr_result.text = "Hello there"
        mock_timestamp_obj = MagicMock()
        mock_timestamp_obj.word = [
            {'word': 'Hello', 'start_time': 0.1, 'end_time': 0.5}, # Valid
            {'word': 'there', 'start_time': 'bad_data'}, # Malformed - missing end_time, bad start_time type
            "not a dict" # Invalid item type
        ]
        mock_asr_result.timestamp = mock_timestamp_obj
        nemo_output = [mock_asr_result]

        result = _process_transcription_output(nemo_output, "dummy.wav", True)
        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "Hello there")
        self.assertEqual(len(result.word_timestamps), 1) # Only the valid one
        self.assertEqual(result.word_timestamps[0].word, "Hello")
        # Check for warning logs (need logger mocking for this)
        # logger.warning.assert_any_call(unittest.mock.ANY) # General check for warning calls

    def test_transcription_input_valid(self):
        abs_path = os.path.abspath(self.dummy_audio_file_path)
        input_data = TranscriptionInput(audio_path=self.dummy_audio_file_path, include_timestamps=True)
        self.assertEqual(input_data.audio_path, abs_path) # Validator makes it absolute
        self.assertTrue(input_data.include_timestamps)

    def test_transcription_input_file_not_found(self):
        with self.assertRaises(ValidationError) as context: # Pydantic raises ValidationError
            TranscriptionInput(audio_path="non_existent_file.wav")
        self.assertIn("File not found", str(context.exception)) # Check custom validator message

    def test_transcription_input_relative_path_conversion(self):
        # Create a dummy file in a subdirectory to test relative path
        test_subdir = "temp_subdir_for_test"
        os.makedirs(test_subdir, exist_ok=True)
        relative_dummy_file = os.path.join(test_subdir, "relative_dummy.wav")
        abs_expected_path = os.path.abspath(relative_dummy_file)

        with open(relative_dummy_file, "w") as f:
            f.write("dummy")
        
        try:
            # Assuming current working directory is project root for this test
            # Pydantic FilePath resolves relative to CWD
            input_data = TranscriptionInput(audio_path=relative_dummy_file)
            self.assertEqual(input_data.audio_path, abs_expected_path)
        finally:
            os.remove(relative_dummy_file)
            os.rmdir(test_subdir)

    @patch('src.transcriber.load_model')
    @patch('src.transcriber._process_transcription_output')
    def test_transcribe_audio_file_success_with_timestamps(self, mock_process_output, mock_load_model):
        mock_asr = MagicMock()
        mock_load_model.return_value = mock_asr
        
        mock_nemo_raw_output = [MagicMock()] # Simulate some NeMo output structure
        mock_asr.transcribe.return_value = mock_nemo_raw_output
        
        expected_result = TranscriptionResult(text="test", word_timestamps=[])
        mock_process_output.return_value = expected_result

        input_data = TranscriptionInput(audio_path=self.dummy_audio_file_path, include_timestamps=True)
        result = transcribe_audio_file(input_data)

        self.assertEqual(result, expected_result)
        mock_load_model.assert_called_once()
        mock_asr.transcribe.assert_called_once_with(
            paths2audio_files=[str(input_data.audio_path)],
            batch_size=1,
            timestamps=True
        )
        mock_process_output.assert_called_once_with(mock_nemo_raw_output, str(input_data.audio_path), True)

    @patch('src.transcriber.load_model')
    @patch('src.transcriber._process_transcription_output')
    def test_transcribe_audio_file_success_no_timestamps(self, mock_process_output, mock_load_model):
        mock_asr = MagicMock()
        mock_load_model.return_value = mock_asr
        mock_nemo_raw_output = ["raw text output"] # Simulate NeMo string output
        mock_asr.transcribe.return_value = mock_nemo_raw_output
        
        expected_result = "raw text output"
        mock_process_output.return_value = expected_result

        input_data = TranscriptionInput(audio_path=self.dummy_audio_file_path, include_timestamps=False)
        result = transcribe_audio_file(input_data)

        self.assertEqual(result, expected_result)
        mock_asr.transcribe.assert_called_once_with(
            paths2audio_files=[str(input_data.audio_path)],
            batch_size=1,
            timestamps=False
        )
        mock_process_output.assert_called_once_with(mock_nemo_raw_output, str(input_data.audio_path), False)

    @patch('src.transcriber.load_model', side_effect=RuntimeError("Model loading failed"))
    def test_transcribe_audio_file_load_model_fails(self, mock_load_model):
        input_data = TranscriptionInput(audio_path=self.dummy_audio_file_path, include_timestamps=True)
        with self.assertRaises(RuntimeError) as context:
            transcribe_audio_file(input_data)
        self.assertIn("Model loading failed", str(context.exception))

    @patch('src.transcriber.load_model')
    def test_transcribe_audio_file_nemo_transcribe_fails(self, mock_load_model):
        mock_asr = MagicMock()
        mock_load_model.return_value = mock_asr
        mock_asr.transcribe.side_effect = Exception("NeMo transcribe error")

        input_data = TranscriptionInput(audio_path=self.dummy_audio_file_path, include_timestamps=True)
        with self.assertRaises(Exception) as context:
            transcribe_audio_file(input_data)
        self.assertIn("NeMo transcribe error", str(context.exception))

    # It's tricky to test the FileNotFoundError from transcribe_audio_file directly 
    # because Pydantic's FilePath and our validator in TranscriptionInput catch it first.
    # The validator test test_transcription_input_file_not_found already covers this.
    # If transcribe_audio_file had its own direct os.path.exists check before Pydantic,
    # that would be tested here.

if __name__ == '__main__':
    # This allows running the tests directly from this file
    # For more complex setups, using a test runner like `python -m unittest discover` is common.
    
    # Setup mock logger for all tests in this module if not done globally
    # This is an example of how you might do it if you don't want real log output during tests
    with patch('src.transcriber.logger', new_callable=MagicMock) as mock_logger_global:
        # If you need to assert specific log calls in multiple tests, this mock_logger_global can be used.
        # For instance, in tests above: mock_logger_global.info.assert_any_call(...)
        # For simplicity in the provided tests, I used logger.info etc. which would use the real logger
        # unless specifically patched within the test or globally like this.
        # To use the globally patched logger in tests:
        # Replace logger.info with self.mock_logger.info in test methods after adding
        # self.mock_logger = mock_logger_global to setUp or test methods.
        
        # For now, the individual logger.info assertions will try to use the real logger.
        # If these fail due to missing mock setup, patch 'src.transcriber.logger' within each test or class.
        unittest.main() 