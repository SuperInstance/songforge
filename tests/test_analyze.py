"""
Tests for the spectral analysis precheck module.

These tests validate the diagnostic logic (diagnose_vocal_presence) without
requiring real audio files or ffmpeg. The analysis functions that call
ffmpeg are tested separately in integration tests.
"""

import pytest
from unittest.mock import patch, MagicMock
from songforge.analyze import (
    BandEnergy,
    SpectralReport,
    diagnose_vocal_presence,
    format_report,
    analyze_recording,
    BANDS,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_band(name: str, rms: float, rel: float) -> BandEnergy:
    low, high = BANDS[name]
    return BandEnergy(
        name=name,
        freq_low=low,
        freq_high=high,
        rms=rms,
        peak_db=rms + 6,
        relative_energy=rel,
    )


def _make_report(vocal_rms: float, instrumental_rms: float) -> SpectralReport:
    """Build a minimal SpectralReport with the given vocal/instrumental levels."""
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
        file="test.mp3",
        duration_sec=11.0,
        sample_rate=44100,
        channels=1,
        spectral_centroid_hz=734,
        spectral_rolloff_85_hz=500,
        spectral_flatness=0.3,
        spectral_flux=0.5,
        bands=bands,
        vocal_band_rms=vocal_rms,
        instrumental_band_rms=instrumental_rms,
        vocal_to_instrumental_ratio_db=round(ratio, 2),
    )


# ─── diagnose_vocal_presence tests ───────────────────────────────────────────

class TestDiagnoseVocalPresence:
    
    def test_proceed_when_vocals_clear(self):
        """Vocals above instruments → proceed."""
        report = _make_report(vocal_rms=-15, instrumental_rms=-20)
        result = diagnose_vocal_presence(report)
        assert result["recommendation"] == "proceed"
        assert result["confidence"] == 0.9
        assert result["warning"] == ""
    
    def test_caution_in_borderline_case(self):
        """Vocals slightly below instruments → caution zone."""
        report = _make_report(vocal_rms=-25, instrumental_rms=-18)
        # ratio = -7, which is in the caution range (-5 to -15)
        result = diagnose_vocal_presence(report)
        assert result["recommendation"] == "caution"
        assert result["confidence"] == 0.5
        assert "degraded" in result["warning"].lower()
    
    def test_skip_when_vocals_buried(self):
        """Vocals far below instruments → skip (the Darmok scenario)."""
        report = _make_report(vocal_rms=-68, instrumental_rms=-20)
        result = diagnose_vocal_presence(report)
        assert result["recommendation"] == "skip_separation"
        assert result["confidence"] == 0.85
        assert "noise floor" in result["warning"].lower()
    
    def test_darmok_exact_values(self):
        """Reproduce the exact Darmok scenario: -68.5 dB vocals vs ~-20 dB guitar."""
        report = _make_report(vocal_rms=-68.5, instrumental_rms=-20)
        result = diagnose_vocal_presence(report)
        assert result["recommendation"] == "skip_separation"
        assert result["ratio_db"] == pytest.approx(-48.5, abs=0.1)
    
    def test_dominant_band_reported(self):
        """The dominant band should be reported correctly."""
        report = _make_report(vocal_rms=-68, instrumental_rms=-20)
        result = diagnose_vocal_presence(report)
        assert result["dominant_band"] == "bass"
    
    def test_ratio_values_passed_through(self):
        """Check that RMS values are passed through to the diagnosis."""
        report = _make_report(vocal_rms=-30, instrumental_rms=-22)
        result = diagnose_vocal_presence(report)
        assert result["vocal_band_rms_db"] == -30
        assert result["instrumental_band_rms_db"] == -22


# ─── format_report tests ─────────────────────────────────────────────────────

class TestFormatReport:
    
    def test_format_contains_key_sections(self):
        report = _make_report(vocal_rms=-15, instrumental_rms=-20)
        diagnosis = diagnose_vocal_presence(report)
        output = format_report(report, diagnosis)
        assert "Spectral Analysis" in output
        assert "Frequency Band Energy" in output
        assert "Vocal Presence Diagnosis" in output
        assert "RECOMMENDATION" in output
    
    def test_format_shows_proceed_icon(self):
        report = _make_report(vocal_rms=-10, instrumental_rms=-20)
        diagnosis = diagnose_vocal_presence(report)
        output = format_report(report, diagnosis)
        assert "✅" in output
    
    def test_format_shows_skip_icon(self):
        report = _make_report(vocal_rms=-68, instrumental_rms=-20)
        diagnosis = diagnose_vocal_presence(report)
        output = format_report(report, diagnosis)
        assert "🚫" in output
    
    def test_format_shows_caution_icon(self):
        report = _make_report(vocal_rms=-25, instrumental_rms=-18)
        diagnosis = diagnose_vocal_presence(report)
        output = format_report(report, diagnosis)
        assert "⚠️" in output


# ─── BandEnergy / SpectralReport dataclass tests ─────────────────────────────

class TestDataclasses:
    
    def test_band_energy_fields(self):
        b = BandEnergy("mid", 500, 2000, -25.0, -19.0, 0.15)
        assert b.name == "mid"
        assert b.rms == -25.0
        assert b.relative_energy == 0.15
    
    def test_spectral_report_defaults(self):
        r = SpectralReport(
            file="test.wav", duration_sec=10, sample_rate=44100,
            channels=1, spectral_centroid_hz=500, spectral_rolloff_85_hz=300,
            spectral_flatness=0.5, spectral_flux=0.2,
        )
        assert r.bands == []
        assert r.vocal_band_rms == 0.0
        assert r.estimated_key is None


# ─── BANDS constant test ──────────────────────────────────────────────────────

class TestBands:
    
    def test_vocal_bands_present(self):
        assert "mid" in BANDS
        assert "high_mid" in BANDS
    
    def test_bass_range_covers_guitar_resonance(self):
        """Guitar body resonance is 80-250 Hz — should be inside bass band."""
        low, high = BANDS["bass"]
        assert low <= 80 and high >= 250
    
    def test_vocal_range_covered(self):
        """Vocal fundamentals (80-300 Hz) span bass and low_mid."""
        # Check that the bands cover the full vocal range
        all_bands = list(BANDS.values())
        min_freq = min(b[0] for b in all_bands)
        max_freq = max(b[1] for b in all_bands)
        assert min_freq <= 80
        assert max_freq >= 4000
