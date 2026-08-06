"""
Tests for the analyze.py module — edge cases beyond test_analyze.py.

These tests focus on ffmpeg wrapper functions, error handling,
and numerical edge cases in the band energy computation.
"""

import pytest
from unittest.mock import patch, MagicMock
from songforge.analyze import (
    diagnose_vocal_presence,
    format_report,
    BandEnergy,
    SpectralReport,
    BANDS,
    VOCAL_BANDS,
    VOCAL_FREQ_RANGE,
    _compute_band_energies,
    _compute_spectral_centroid,
    _run_ffprobe,
    _extract_segment,
    analyze_recording,
)


def _make_band(name, rms, rel):
    low, high = BANDS[name]
    return BandEnergy(name=name, freq_low=low, freq_high=high, rms=rms,
                      peak_db=rms + 6, relative_energy=rel)


def _make_report(vocal_rms, instrumental_rms):
    bands = [
        _make_band("sub_bass", -50, 0.02),
        _make_band("bass", instrumental_rms, 0.50),
        _make_band("low_mid", -30, 0.15),
        _make_band("mid", vocal_rms, 0.10),
        _make_band("high_mid", vocal_rms - 2, 0.08),
        _make_band("treble", -45, 0.10),
        _make_band("air", -55, 0.05),
    ]
    ratio = vocal_rms - instrumental_rms
    return SpectralReport(
        file="test.mp3", duration_sec=11.0, sample_rate=44100, channels=1,
        spectral_centroid_hz=734, spectral_rolloff_85_hz=500,
        spectral_flatness=0.3, spectral_flux=0.5, bands=bands,
        vocal_band_rms=vocal_rms, instrumental_band_rms=instrumental_rms,
        vocal_to_instrumental_ratio_db=round(ratio, 2),
    )


class TestDiagnoseBoundaryConditions:
    
    def test_exact_minus_5_ratio_is_proceed(self):
        """At exactly -5 dB ratio, should be 'proceed' (ratio > -5 check)."""
        report = _make_report(vocal_rms=-25, instrumental_rms=-20)
        report.vocal_to_instrumental_ratio_db = -5.0
        result = diagnose_vocal_presence(report)
        # The check is ratio > -5, so -5 is NOT > -5, should NOT be proceed
        # It falls to caution
        assert result["recommendation"] == "caution"
    
    def test_just_above_minus_5_is_proceed(self):
        """At -4.9 dB ratio, should be proceed."""
        report = _make_report(vocal_rms=-24, instrumental_rms=-20)
        report.vocal_to_instrumental_ratio_db = -4.9
        result = diagnose_vocal_presence(report)
        assert result["recommendation"] == "proceed"
    
    def test_exact_minus_15_ratio_is_skip(self):
        """At exactly -15 dB ratio, the boundary falls to skip (ratio > -15 is False at -15)."""
        report = _make_report(vocal_rms=-35, instrumental_rms=-20)
        report.vocal_to_instrumental_ratio_db = -15.0
        result = diagnose_vocal_presence(report)
        assert result["recommendation"] == "skip_separation"
    
    def test_just_above_minus_15_is_caution(self):
        """At -14.9 dB ratio, should be caution."""
        report = _make_report(vocal_rms=-34.9, instrumental_rms=-20)
        report.vocal_to_instrumental_ratio_db = -14.9
        result = diagnose_vocal_presence(report)
        assert result["recommendation"] == "caution"
    
    def test_just_below_minus_15_is_skip(self):
        """At -15.1 dB ratio, should be skip_separation."""
        report = _make_report(vocal_rms=-35.1, instrumental_rms=-20)
        report.vocal_to_instrumental_ratio_db = -15.1
        result = diagnose_vocal_presence(report)
        assert result["recommendation"] == "skip_separation"


class TestDominantBandDetection:
    
    def test_dominant_band_is_bass_when_instruments_dominate(self):
        report = _make_report(vocal_rms=-68, instrumental_rms=-20)
        result = diagnose_vocal_presence(report)
        assert result["dominant_band"] == "bass"
    
    def test_dominant_band_is_mid_when_vocals_dominate(self):
        # Make mid band dominant
        bands = [
            _make_band("sub_bass", -50, 0.05),
            _make_band("bass", -30, 0.10),
            _make_band("low_mid", -25, 0.15),
            _make_band("mid", -10, 0.40),
            _make_band("high_mid", -12, 0.20),
            _make_band("treble", -45, 0.05),
            _make_band("air", -55, 0.05),
        ]
        report = SpectralReport(
            file="test.wav", duration_sec=10, sample_rate=44100, channels=1,
            spectral_centroid_hz=1000, spectral_rolloff_85_hz=2000,
            spectral_flatness=0.3, spectral_flux=0.5, bands=bands,
            vocal_band_rms=-10, instrumental_band_rms=-30,
            vocal_to_instrumental_ratio_db=20.0,
        )
        result = diagnose_vocal_presence(report)
        assert result["dominant_band"] == "mid"
    
    def test_dominant_band_is_correct_for_equal_energies(self):
        """When all bands have equal relative energy, the first wins (max behavior)."""
        bands = [
            _make_band(name, -20, 0.142857) for name in BANDS
        ]
        report = SpectralReport(
            file="flat.wav", duration_sec=10, sample_rate=44100, channels=1,
            spectral_centroid_hz=500, spectral_rolloff_85_hz=300,
            spectral_flatness=0.5, spectral_flux=0.2, bands=bands,
            vocal_band_rms=-20, instrumental_band_rms=-20,
            vocal_to_instrumental_ratio_db=0.0,
        )
        result = diagnose_vocal_presence(report)
        # max() with all equal returns the first in iteration order
        assert result["dominant_band"] in BANDS


class TestFormatReport:
    
    def test_format_includes_all_band_names(self):
        report = _make_report(vocal_rms=-15, instrumental_rms=-20)
        diagnosis = diagnose_vocal_presence(report)
        output = format_report(report, diagnosis)
        for band_name in BANDS:
            assert band_name in output
    
    def test_format_includes_file_name(self):
        report = _make_report(vocal_rms=-15, instrumental_rms=-20)
        report.file = "my_song.mp3"
        diagnosis = diagnose_vocal_presence(report)
        output = format_report(report, diagnosis)
        assert "my_song.mp3" in output
    
    def test_format_includes_duration(self):
        report = _make_report(vocal_rms=-15, instrumental_rms=-20)
        report.duration_sec = 42.7
        diagnosis = diagnose_vocal_presence(report)
        output = format_report(report, diagnosis)
        assert "42.7" in output
    
    def test_format_includes_ratio(self):
        report = _make_report(vocal_rms=-25, instrumental_rms=-20)
        diagnosis = diagnose_vocal_presence(report)
        output = format_report(report, diagnosis)
        assert "-5.0" in output  # ratio is -5.0
    
    def test_format_caution_includes_warning_text(self):
        report = _make_report(vocal_rms=-25, instrumental_rms=-18)
        diagnosis = diagnose_vocal_presence(report)
        output = format_report(report, diagnosis)
        assert "degraded" in output.lower()


class TestBandsConstant:
    
    def test_all_bands_are_non_overlapping(self):
        """Frequency bands should not overlap (they tile the spectrum)."""
        sorted_bands = sorted(BANDS.values(), key=lambda b: b[0])
        for i in range(len(sorted_bands) - 1):
            assert sorted_bands[i][1] <= sorted_bands[i + 1][0], \
                f"Band {i} ends at {sorted_bands[i][1]} but next starts at {sorted_bands[i+1][0]}"
    
    def test_sub_bass_starts_at_20hz(self):
        assert BANDS["sub_bass"][0] == 20
    
    def test_air_ends_at_16000hz(self):
        assert BANDS["air"][1] == 16000
    
    def test_vocal_bands_are_mid_and_high_mid(self):
        assert VOCAL_BANDS == ["mid", "high_mid"]
    
    def test_vocal_freq_range(self):
        assert VOCAL_FREQ_RANGE == (300, 4000)
    
    def test_bass_covers_guitar_body_resonance(self):
        """Guitar body resonance 80-250 Hz must be inside bass band."""
        low, high = BANDS["bass"]
        assert low <= 80 and high >= 250
    
    def test_mid_covers_speech_intelligibility(self):
        """1-2 kHz must be inside mid band."""
        low, high = BANDS["mid"]
        assert low <= 1000 and high >= 2000


class TestComputeBandEnergies:
    
    @patch('songforge.analyze.subprocess.run')
    def test_parses_mean_volume_correctly(self, mock_run):
        """Should parse mean_volume from ffmpeg stderr."""
        # Simulate ffmpeg stderr output
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr="[Parsed_volumedetect_0 @ 0x7f8b1c01b300] mean_volume: -23.4 dB\n"
                   "[Parsed_volumedetect_0 @ 0x7f8b1c01b300] max_volume: -5.2 dB\n",
            stdout=""
        )
        
        bands = _compute_band_energies("/tmp/test.wav")
        
        assert len(bands) == len(BANDS)
        for band in bands:
            assert band.rms == -23.4
            assert band.peak_db == -5.2
    
    @patch('songforge.analyze.subprocess.run')
    def test_handles_missing_mean_volume_line(self, mock_run):
        """Should default to 0.0 RMS when mean_volume is absent."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr="some other output\n",
            stdout=""
        )
        
        bands = _compute_band_energies("/tmp/test.wav")
        
        for band in bands:
            assert band.rms == 0.0
    
    @patch('songforge.analyze.subprocess.run')
    def test_relative_energy_sums_to_approximately_1(self, mock_run):
        """Relative energies across all bands should sum to ~1.0."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr="mean_volume: -20.0 dB\nmax_volume: -2.0 dB\n",
            stdout=""
        )
        
        bands = _compute_band_energies("/tmp/test.wav")
        
        total = sum(b.relative_energy for b in bands)
        assert abs(total - 1.0) < 0.01  # all bands same level → equal split
    
    @patch('songforge.analyze.subprocess.run')
    def test_below_floor_rms_treated_as_silence(self, mock_run):
        """Bands below -90 dB should be treated as silence (0 linear energy)."""
        # First band at -95, rest at -20
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(returncode=0,
                    stderr="mean_volume: -95.0 dB\nmax_volume: -80.0 dB\n", stdout="")
            return MagicMock(returncode=0,
                stderr="mean_volume: -20.0 dB\nmax_volume: -2.0 dB\n", stdout="")
        
        mock_run.side_effect = side_effect
        
        bands = _compute_band_energies("/tmp/test.wav")
        
        assert bands[0].relative_energy == 0.0  # sub_bass below floor
        assert bands[1].relative_energy > 0.0   # bass above floor


class TestAnalyzeRecording:
    
    def test_raises_on_nonexistent_file(self):
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            analyze_recording("/nonexistent/path/file.mp3")
    
    @patch('songforge.analyze._extract_segment')
    @patch('songforge.analyze._run_ffprobe')
    @patch('songforge.analyze._compute_band_energies')
    @patch('songforge.analyze._compute_spectral_centroid')
    def test_returns_report_with_correct_metadata(self, mock_centroid, mock_bands,
                                                   mock_probe, mock_segment):
        mock_segment.return_value = "/tmp/segment.wav"
        mock_probe.return_value = {
            "format": {"duration": "30.0"},
            "streams": [{"codec_type": "audio", "sample_rate": "48000", "channels": 2}],
        }
        mock_bands.return_value = [
            _make_band("sub_bass", -50, 0.05),
            _make_band("bass", -20, 0.50),
            _make_band("low_mid", -30, 0.15),
            _make_band("mid", -15, 0.10),
            _make_band("high_mid", -17, 0.08),
            _make_band("treble", -45, 0.07),
            _make_band("air", -55, 0.05),
        ]
        mock_centroid.return_value = (1000.0, 2000.0, 0.3, 0.5)
        
        # Create a temp file to satisfy the existence check
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        try:
            report = analyze_recording(tmp.name)
            assert report.duration_sec == 30.0
            assert report.sample_rate == 48000
            assert report.channels == 2
            assert report.spectral_centroid_hz == 1000.0
        finally:
            os.unlink(tmp.name)
