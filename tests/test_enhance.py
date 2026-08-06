"""
Tests for the vocal enhancement module (enhance.py).

These tests mock ffmpeg subprocess calls and verify filter construction
logic and error handling.
"""

import pytest
from unittest.mock import patch, MagicMock

from songforge.enhance import enhance_vocals


class TestEnhanceVocals:
    
    @patch('songforge.enhance.subprocess.run')
    def test_basic_enhancement_calls_ffmpeg(self, mock_run):
        """Should call ffmpeg with volume and EQ filters."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        result = enhance_vocals("input.wav", "output.wav")
        
        assert mock_run.called
        assert result == "output.wav"
        
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        assert "-i" in cmd
        assert "input.wav" in cmd
    
    @patch('songforge.enhance.subprocess.run')
    def test_default_volume_is_3(self, mock_run):
        """Default volume should be 3.0."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        enhance_vocals("input.wav", "output.wav")
        
        cmd = mock_run.call_args[0][0]
        af_idx = cmd.index("-af") if "-af" in cmd else cmd.index("-af")
        filter_str = cmd[af_idx + 1]
        assert "volume=3.0" in filter_str
    
    @patch('songforge.enhance.subprocess.run')
    def test_custom_volume_applied(self, mock_run):
        """Custom volume should appear in filter chain."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        enhance_vocals("input.wav", "output.wav", volume=5.5)
        
        cmd = mock_run.call_args[0][0]
        af_idx = list(cmd).index("-af")
        filter_str = cmd[af_idx + 1]
        assert "volume=5.5" in filter_str
    
    @patch('songforge.enhance.subprocess.run')
    def test_equalizer_filter_present(self, mock_run):
        """EQ filter should be in the chain by default."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        enhance_vocals("input.wav", "output.wav")
        
        cmd = mock_run.call_args[0][0]
        af_idx = list(cmd).index("-af")
        filter_str = cmd[af_idx + 1]
        assert "equalizer" in filter_str
        assert "f=2000" in filter_str  # default eq_freq
    
    @patch('songforge.enhance.subprocess.run')
    def test_custom_eq_freq(self, mock_run):
        """Custom EQ frequency should be used."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        enhance_vocals("input.wav", "output.wav", eq_freq=3000)
        
        cmd = mock_run.call_args[0][0]
        af_idx = list(cmd).index("-af")
        filter_str = cmd[af_idx + 1]
        assert "f=3000" in filter_str
        # Width should be half the freq
        assert "width=1500" in filter_str
    
    @patch('songforge.enhance.subprocess.run')
    def test_denoise_off_by_default(self, mock_run):
        """Denoise should NOT be in the filter chain by default."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        enhance_vocals("input.wav", "output.wav")
        
        cmd = mock_run.call_args[0][0]
        af_idx = list(cmd).index("-af")
        filter_str = cmd[af_idx + 1]
        assert "afftdn" not in filter_str
    
    @patch('songforge.enhance.subprocess.run')
    def test_denoise_on_when_requested(self, mock_run):
        """Denoise filter should be present when denoise=True."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        enhance_vocals("input.wav", "output.wav", denoise=True)
        
        cmd = mock_run.call_args[0][0]
        af_idx = list(cmd).index("-af")
        filter_str = cmd[af_idx + 1]
        assert "afftdn=nr=10:nf=-40" in filter_str
    
    @patch('songforge.enhance.subprocess.run')
    def test_filter_chain_order_volume_eq_denoise(self, mock_run):
        """Filter order should be: volume, equalizer, denoise (if present)."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        enhance_vocals("input.wav", "output.wav", denoise=True)
        
        cmd = mock_run.call_args[0][0]
        af_idx = list(cmd).index("-af")
        filter_str = cmd[af_idx + 1]
        parts = filter_str.split(",")
        assert parts[0].startswith("volume=")
        assert parts[1].startswith("equalizer=")
        assert parts[2].startswith("afftdn=")
    
    @patch('songforge.enhance.subprocess.run')
    def test_raises_on_ffmpeg_failure(self, mock_run):
        """Should raise RuntimeError when ffmpeg fails."""
        mock_run.return_value = MagicMock(returncode=1, stderr="codec error", stdout="")
        
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            enhance_vocals("input.wav", "output.wav")
    
    @patch('songforge.enhance.subprocess.run')
    def test_default_output_filename(self, mock_run):
        """When no output specified, should default to enhanced.wav."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        enhance_vocals("input.wav")
        
        cmd = mock_run.call_args[0][0]
        # Last argument should be the output file
        assert cmd[-1] == "enhanced.wav"
    
    @patch('songforge.enhance.subprocess.run')
    def test_zero_volume(self, mock_run):
        """Volume of 0 should produce volume=0.0 filter."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        enhance_vocals("input.wav", "output.wav", volume=0.0)
        
        cmd = mock_run.call_args[0][0]
        af_idx = list(cmd).index("-af")
        filter_str = cmd[af_idx + 1]
        assert "volume=0.0" in filter_str
    
    @patch('songforge.enhance.subprocess.run')
    def test_negative_volume(self, mock_run):
        """Negative volume should still produce a valid filter."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        enhance_vocals("input.wav", "output.wav", volume=-2.0)
        
        cmd = mock_run.call_args[0][0]
        af_idx = list(cmd).index("-af")
        filter_str = cmd[af_idx + 1]
        assert "volume=-2.0" in filter_str
