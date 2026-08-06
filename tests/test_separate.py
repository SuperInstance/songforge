"""
Tests for the vocal separation module (separate.py).

These tests mock the subprocess calls to demucs and verify the logic
around stem separation without requiring actual audio processing.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from songforge.separate import separate_stems


class TestSeparateStems:
    
    @patch('songforge.separate.subprocess.run')
    @patch('songforge.separate.Path')
    def test_separate_calls_demucs_with_correct_args(self, mock_path, mock_run):
        """Verify demucs is called with the right CLI arguments."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        # Make Path operations work with mocks
        input_path = MagicMock()
        input_path.stem = "test_song"
        input_path.name = "test_song.mp3"
        output_path = MagicMock()
        output_path.mkdir = MagicMock()
        
        def path_constructor(p):
            if "test_song.mp3" in str(p):
                return input_path
            return output_path
        
        mock_path.side_effect = path_constructor
        
        # Make stem files exist
        stem_dir = MagicMock()
        stem_dir.__truediv__ = lambda self, x: MagicMock(exists=lambda: True)
        output_path.__truediv__ = lambda self, x: stem_dir
        
        separate_stems("test_song.mp3", "./stems/", "htdemucs")
        
        # Check demucs was called
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "demucs" in str(cmd)
        assert "--two-stems" in cmd
        assert "vocals" in cmd
    
    @patch('songforge.separate.subprocess.run')
    def test_separate_raises_on_demucs_failure(self, mock_run):
        """Should raise RuntimeError when demucs exits non-zero."""
        mock_run.return_value = MagicMock(returncode=1, stderr="GPU error", stdout="")
        
        with pytest.raises(RuntimeError, match="Demucs failed"):
            separate_stems("input.mp3", "./stems/", "htdemucs")
    
    @patch('songforge.separate.Path')
    def test_separate_creates_output_dir(self, mock_path):
        """Output directory should be created if it doesn't exist."""
        from unittest.mock import MagicMock
        
        output_path = MagicMock()
        mock_path.return_value = output_path
        
        with pytest.raises(Exception):
            separate_stems("input.mp3", "./stems/")
        
        output_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestSeparateStemsModelParameter:
    
    @patch('songforge.separate.subprocess.run')
    @patch('songforge.separate.Path')
    def test_default_model_is_htdemucs(self, mock_path, mock_run):
        """Default model should be htdemucs."""
        mock_run.return_value = MagicMock(returncode=1, stderr="err", stdout="")
        
        with pytest.raises(Exception):
            separate_stems("input.mp3")
        
        cmd = mock_run.call_args[0][0]
        assert "-n" in cmd
        idx = cmd.index("-n")
        assert cmd[idx + 1] == "htdemucs"
    
    @patch('songforge.separate.subprocess.run')
    @patch('songforge.separate.Path')
    def test_custom_model_passed_through(self, mock_path, mock_run):
        """Custom model name should be passed to demucs."""
        mock_run.return_value = MagicMock(returncode=1, stderr="err", stdout="")
        
        with pytest.raises(Exception):
            separate_stems("input.mp3", model="mdx_extra")
        
        cmd = mock_run.call_args[0][0]
        idx = cmd.index("-n")
        assert cmd[idx + 1] == "mdx_extra"
