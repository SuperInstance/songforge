"""
Tests for the vocal transcription module (transcribe.py).

These tests mock the whisper library and file I/O to test the
transcription and lyrics comparison logic.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from songforge.transcribe import transcribe_audio


class TestTranscribeAudio:
    
    @patch('songforge.transcribe.Path')
    def test_transcribe_without_compare(self, mock_path_class):
        """Should return transcription without comparison when no compare file given."""
        # We need to mock the whisper import inside the function
        whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "  Hello world  ",
            "language": "en",
            "segments": [{"id": 0}, {"id": 1}, {"id": 2}],
        }
        whisper_module.load_model.return_value = mock_model
        
        with patch.dict('sys.modules', {'whisper': whisper_module}):
            result = transcribe_audio("song.wav", model="small")
        
        assert result["transcription"] == "Hello world"
        assert result["language"] == "en"
        assert result["segments"] == 3
        assert "known_lyrics" not in result
        assert "lyrics_overlap" not in result
    
    @patch('songforge.transcribe.Path')
    def test_transcribe_empty_text(self, mock_path_class):
        """Should handle empty transcription gracefully."""
        whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "   ",
            "language": "en",
            "segments": [],
        }
        whisper_module.load_model.return_value = mock_model
        
        with patch.dict('sys.modules', {'whisper': whisper_module}):
            result = transcribe_audio("song.wav")
        
        assert result["transcription"] == ""
        assert result["segments"] == 0
    
    @patch('songforge.transcribe.Path')
    def test_transcribe_missing_language(self, mock_path_class):
        """Should default to 'unknown' when language not provided."""
        whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "some lyrics",
        }
        whisper_module.load_model.return_value = mock_model
        
        with patch.dict('sys.modules', {'whisper': whisper_module}):
            result = transcribe_audio("song.wav")
        
        assert result["language"] == "unknown"
    
    @patch('songforge.transcribe.Path')
    def test_transcribe_missing_segments(self, mock_path_class):
        """Should handle missing segments key."""
        whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "some lyrics",
            "language": "en",
        }
        whisper_module.load_model.return_value = mock_model
        
        with patch.dict('sys.modules', {'whisper': whisper_module}):
            result = transcribe_audio("song.wav")
        
        assert result["segments"] == 0


class TestTranscribeWithCompare:
    
    @patch('songforge.transcribe.Path')
    def test_compare_with_existing_file(self, mock_path_class):
        """Should compute lyrics overlap when compare file exists."""
        whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "hello world foo bar",
            "language": "en",
            "segments": [],
        }
        whisper_module.load_model.return_value = mock_model
        
        # Mock Path to simulate the compare file existing
        mock_compare_path = MagicMock()
        mock_compare_path.exists.return_value = True
        mock_compare_path.read_text.return_value = "hello world baz qux"
        mock_path_class.return_value = mock_compare_path
        
        with patch.dict('sys.modules', {'whisper': whisper_module}):
            result = transcribe_audio("song.wav", compare="known.txt")
        
        assert result["known_lyrics"] == "hello world baz qux"
        # trans_words: {hello, world, foo, bar}
        # known_words: {hello, world, baz, qux}
        # overlap: {hello, world} = 2 out of 4 known = 0.5
        assert result["lyrics_overlap"] == 0.5
    
    @patch('songforge.transcribe.Path')
    def test_compare_with_nonexistent_file(self, mock_path_class):
        """Should not include comparison data when file doesn't exist."""
        whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "hello world",
            "language": "en",
            "segments": [],
        }
        whisper_module.load_model.return_value = mock_model
        
        mock_compare_path = MagicMock()
        mock_compare_path.exists.return_value = False
        mock_path_class.return_value = mock_compare_path
        
        with patch.dict('sys.modules', {'whisper': whisper_module}):
            result = transcribe_audio("song.wav", compare="missing.txt")
        
        assert "known_lyrics" not in result
        assert "lyrics_overlap" not in result
    
    @patch('songforge.transcribe.Path')
    def test_compare_perfect_overlap(self, mock_path_class):
        """Should return 1.0 overlap when all words match."""
        whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "hello world",
            "language": "en",
            "segments": [],
        }
        whisper_module.load_model.return_value = mock_model
        
        mock_compare_path = MagicMock()
        mock_compare_path.exists.return_value = True
        mock_compare_path.read_text.return_value = "hello world"
        mock_path_class.return_value = mock_compare_path
        
        with patch.dict('sys.modules', {'whisper': whisper_module}):
            result = transcribe_audio("song.wav", compare="known.txt")
        
        assert result["lyrics_overlap"] == 1.0
    
    @patch('songforge.transcribe.Path')
    def test_compare_zero_overlap(self, mock_path_class):
        """Should return 0.0 overlap when no words match."""
        whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "alpha beta",
            "language": "en",
            "segments": [],
        }
        whisper_module.load_model.return_value = mock_model
        
        mock_compare_path = MagicMock()
        mock_compare_path.exists.return_value = True
        mock_compare_path.read_text.return_value = "gamma delta"
        mock_path_class.return_value = mock_compare_path
        
        with patch.dict('sys.modules', {'whisper': whisper_module}):
            result = transcribe_audio("song.wav", compare="known.txt")
        
        assert result["lyrics_overlap"] == 0.0
    
    @patch('songforge.transcribe.Path')
    def test_compare_empty_known_lyrics(self, mock_path_class):
        """Should not compute overlap when known lyrics are empty."""
        whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "hello world",
            "language": "en",
            "segments": [],
        }
        whisper_module.load_model.return_value = mock_model
        
        mock_compare_path = MagicMock()
        mock_compare_path.exists.return_value = True
        mock_compare_path.read_text.return_value = "   "
        mock_path_class.return_value = mock_compare_path
        
        with patch.dict('sys.modules', {'whisper': whisper_module}):
            result = transcribe_audio("song.wav", compare="known.txt")
        
        # Empty known lyrics → known_words is empty set → division skipped
        assert "lyrics_overlap" not in result
    
    @patch('songforge.transcribe.Path')
    def test_overlap_is_case_insensitive(self, mock_path_class):
        """Lyrics comparison should be case insensitive."""
        whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Hello WORLD",
            "language": "en",
            "segments": [],
        }
        whisper_module.load_model.return_value = mock_model
        
        mock_compare_path = MagicMock()
        mock_compare_path.exists.return_value = True
        mock_compare_path.read_text.return_value = "hello world"
        mock_path_class.return_value = mock_compare_path
        
        with patch.dict('sys.modules', {'whisper': whisper_module}):
            result = transcribe_audio("song.wav", compare="known.txt")
        
        assert result["lyrics_overlap"] == 1.0
