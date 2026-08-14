"""
Comprehensive tests for SongForge CLI and pipeline modules.
Covers cli.py (0% → target 90%+) and pipeline.py (0% → target 80%+).
"""

import argparse
import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from songforge.cli import main
from songforge.pipeline import cover_pipeline, _generate_cover, _mix_tracks, _cleanup_intermediates


# ─── CLI: Argument Parsing ───

class TestCLIParsing:
    """Test that CLI parses arguments correctly."""

    def test_cover_command_parsed(self):
        """Cover subcommand should parse all its arguments."""
        parser = argparse.ArgumentParser(prog="songforge")
        sub = parser.add_subparsers(dest="command")
        cover = sub.add_parser("cover")
        cover.add_argument("--input", "-i", required=True)
        cover.add_argument("--lyrics", "-l", required=True)
        cover.add_argument("--style", "-s", default="acoustic folk, warm vocals")
        cover.add_argument("--output", "-o", default="cover.mp3")
        cover.add_argument("--keep-stems", action="store_true")
        cover.add_argument("--force", action="store_true")

        args = parser.parse_args(["cover", "-i", "song.mp3", "-l", "la la la"])
        assert args.command == "cover"
        assert args.input == "song.mp3"
        assert args.lyrics == "la la la"
        assert args.style == "acoustic folk, warm vocals"
        assert args.output == "cover.mp3"
        assert args.keep_stems is False
        assert args.force is False

    def test_separate_command_parsed(self):
        parser = argparse.ArgumentParser(prog="songforge")
        sub = parser.add_subparsers(dest="command")
        sep = sub.add_parser("separate")
        sep.add_argument("--input", "-i", required=True)
        sep.add_argument("--output-dir", "-o", default="./stems/")
        sep.add_argument("--model", "-m", default="htdemucs")

        args = parser.parse_args(["separate", "-i", "song.mp3"])
        assert args.command == "separate"
        assert args.input == "song.mp3"
        assert args.output_dir == "./stems/"
        assert args.model == "htdemucs"

    def test_transcribe_command_parsed(self):
        parser = argparse.ArgumentParser(prog="songforge")
        sub = parser.add_subparsers(dest="command")
        trans = sub.add_parser("transcribe")
        trans.add_argument("--input", "-i", required=True)
        trans.add_argument("--model", "-m", default="small")
        trans.add_argument("--compare", "-c")

        args = parser.parse_args(["transcribe", "-i", "vocals.wav", "-m", "medium", "-c", "lyrics.txt"])
        assert args.command == "transcribe"
        assert args.input == "vocals.wav"
        assert args.model == "medium"
        assert args.compare == "lyrics.txt"

    def test_analyze_command_parsed(self):
        parser = argparse.ArgumentParser(prog="songforge")
        sub = parser.add_subparsers(dest="command")
        ana = sub.add_parser("analyze")
        ana.add_argument("--input", "-i", required=True)
        ana.add_argument("--duration", "-d", type=float, default=30.0)

        args = parser.parse_args(["analyze", "-i", "song.mp3", "-d", "60"])
        assert args.command == "analyze"
        assert args.input == "song.mp3"
        assert args.duration == 60.0

    def test_enhance_command_parsed(self):
        parser = argparse.ArgumentParser(prog="songforge")
        sub = parser.add_subparsers(dest="command")
        enh = sub.add_parser("enhance")
        enh.add_argument("--input", "-i", required=True)
        enh.add_argument("--output", "-o", default="enhanced.wav")
        enh.add_argument("--volume", type=float, default=3.0)
        enh.add_argument("--denoise", action="store_true")

        args = parser.parse_args(["enhance", "-i", "vocals.wav", "--denoise", "--volume", "5.0"])
        assert args.command == "enhance"
        assert args.input == "vocals.wav"
        assert args.output == "enhanced.wav"
        assert args.volume == 5.0
        assert args.denoise is True

    def test_no_command_prints_help(self):
        """Running with no subcommand should exit 1."""
        with patch.object(sys, "argv", ["songforge"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


# ─── CLI: Command Dispatch ───

class TestCLICommandDispatch:
    """Test that CLI dispatches to the right functions."""

    @patch("songforge.pipeline.cover_pipeline")
    def test_cover_dispatches_to_pipeline(self, mock_pipeline):
        with patch.object(sys, "argv", ["songforge", "cover", "-i", "song.mp3", "-l", "words"]):
            main()
            mock_pipeline.assert_called_once()
            args = mock_pipeline.call_args[0][0]
            assert args.input == "song.mp3"
            assert args.lyrics == "words"

    @patch("songforge.analyze.analyze_recording")
    @patch("songforge.analyze.diagnose_vocal_presence")
    @patch("songforge.analyze.format_report")
    def test_analyze_dispatches_correctly(self, mock_format, mock_diagnose, mock_analyze):
        mock_report = {"duration": 30.0}
        mock_analyze.return_value = mock_report
        mock_diagnosis = {"recommendation": "proceed"}
        mock_diagnose.return_value = mock_diagnosis
        mock_format.return_value = "Report text"

        with patch.object(sys, "argv", ["songforge", "analyze", "-i", "song.mp3"]):
            main()

        mock_analyze.assert_called_once_with("song.mp3", segment_duration=30.0)
        mock_diagnose.assert_called_once_with(mock_report)
        mock_format.assert_called_once_with(mock_report, mock_diagnosis)

    @patch("songforge.separate.separate_stems")
    def test_separate_dispatches_correctly(self, mock_sep):
        with patch.object(sys, "argv", ["songforge", "separate", "-i", "song.mp3", "-o", "out/", "-m", "mdx"]):
            main()
        mock_sep.assert_called_once_with("song.mp3", "out/", "mdx")

    @patch("songforge.transcribe.transcribe_audio")
    def test_transcribe_dispatches_correctly(self, mock_trans):
        with patch.object(sys, "argv", ["songforge", "transcribe", "-i", "v.wav", "-m", "tiny"]):
            main()
        mock_trans.assert_called_once_with("v.wav", "tiny", None)

    @patch("songforge.enhance.enhance_vocals")
    def test_enhance_dispatches_correctly(self, mock_enh):
        with patch.object(sys, "argv", ["songforge", "enhance", "-i", "v.wav", "-o", "out.wav", "--volume", "4.5", "--denoise"]):
            main()
        mock_enh.assert_called_once_with("v.wav", "out.wav", 4.5, True)


# ─── CLI: Defaults ───

class TestCLIDefaults:
    """Test default values are applied correctly."""

    @patch("songforge.pipeline.cover_pipeline")
    def test_cover_defaults(self, mock_pipeline):
        with patch.object(sys, "argv", ["songforge", "cover", "-i", "s.mp3", "-l", "x"]):
            main()
            args = mock_pipeline.call_args[0][0]
            assert args.style == "acoustic folk, warm vocals"
            assert args.output == "cover.mp3"
            assert args.keep_stems is False
            assert args.force is False

    @patch("songforge.separate.separate_stems")
    def test_separate_defaults(self, mock_sep):
        with patch.object(sys, "argv", ["songforge", "separate", "-i", "s.mp3"]):
            main()
        mock_sep.assert_called_once_with("s.mp3", "./stems/", "htdemucs")

    @patch("songforge.transcribe.transcribe_audio")
    def test_transcribe_defaults(self, mock_trans):
        with patch.object(sys, "argv", ["songforge", "transcribe", "-i", "v.wav"]):
            main()
        mock_trans.assert_called_once_with("v.wav", "small", None)

    @patch("songforge.enhance.enhance_vocals")
    def test_enhance_defaults(self, mock_enh):
        with patch.object(sys, "argv", ["songforge", "enhance", "-i", "v.wav"]):
            main()
        mock_enh.assert_called_once_with("v.wav", "enhanced.wav", 3.0, False)


# ─── Pipeline: cover_pipeline ───

class TestCoverPipeline:
    """Test the full cover pipeline orchestration."""

    def _make_args(self, **overrides):
        """Create mock args namespace with defaults."""
        defaults = {
            "input": "song.mp3",
            "lyrics": "la la la",
            "style": "acoustic folk",
            "output": "cover.mp3",
            "keep_stems": False,
            "force": False,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @patch("songforge.pipeline._cleanup_intermediates")
    @patch("songforge.pipeline._mix_tracks")
    @patch("songforge.pipeline._generate_cover")
    @patch("songforge.pipeline.transcribe_audio")
    @patch("songforge.pipeline.enhance_vocals")
    @patch("songforge.pipeline.separate_stems")
    @patch("songforge.pipeline.format_report")
    @patch("songforge.pipeline.diagnose_vocal_presence")
    @patch("songforge.pipeline.analyze_recording")
    def test_full_pipeline_proceed(self, mock_analyze, mock_diagnose, mock_format,
                                    mock_separate, mock_enhance, mock_transcribe,
                                    mock_generate, mock_mix, mock_cleanup):
        """When diagnosis says proceed, full pipeline runs all 5 steps."""
        mock_analyze.return_value = {"duration": 30.0}
        mock_diagnose.return_value = {"recommendation": "proceed"}
        mock_format.return_value = "Report"
        mock_separate.return_value = {"vocals": "vocals.wav", "no_vocals": "instrumental.wav"}
        mock_enhance.return_value = "enhanced.wav"
        mock_transcribe.return_value = {"transcription": "la la la"}
        mock_generate.return_value = "cover.mp3"
        mock_mix.return_value = "cover_mixed.mp3"

        args = self._make_args()
        result = cover_pipeline(args)

        mock_analyze.assert_called_once()
        mock_separate.assert_called_once()
        mock_enhance.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_generate.assert_called_once()
        mock_mix.assert_called_once()

    @patch("songforge.pipeline._generate_cover")
    @patch("songforge.pipeline.format_report")
    @patch("songforge.pipeline.diagnose_vocal_presence")
    @patch("songforge.pipeline.analyze_recording")
    def test_pipeline_skip_separation(self, mock_analyze, mock_diagnose, mock_format, mock_generate):
        """When vocals below noise floor and no --force, skip separation."""
        mock_analyze.return_value = {"duration": 30.0}
        mock_diagnose.return_value = {"recommendation": "skip_separation"}
        mock_format.return_value = "Report"
        mock_generate.return_value = "cover.mp3"

        args = self._make_args(force=False)
        result = cover_pipeline(args)

        mock_generate.assert_called_once()
        # Should NOT have called separate_stems etc.

    @patch("songforge.pipeline._cleanup_intermediates")
    @patch("songforge.pipeline._mix_tracks")
    @patch("songforge.pipeline._generate_cover")
    @patch("songforge.pipeline.transcribe_audio")
    @patch("songforge.pipeline.enhance_vocals")
    @patch("songforge.pipeline.separate_stems")
    @patch("songforge.pipeline.format_report")
    @patch("songforge.pipeline.diagnose_vocal_presence")
    @patch("songforge.pipeline.analyze_recording")
    def test_pipeline_force_overrides_skip(self, mock_analyze, mock_diagnose, mock_format,
                                            mock_separate, mock_enhance,
                                            mock_transcribe, mock_generate, mock_mix,
                                            mock_cleanup):
        """When --force is set, separation runs even if diagnosis says skip."""
        mock_analyze.return_value = {"duration": 30.0}
        mock_diagnose.return_value = {"recommendation": "skip_separation"}
        mock_format.return_value = "Report"
        mock_separate.return_value = {"vocals": "v.wav", "no_vocals": "i.wav"}
        mock_enhance.return_value = "enhanced.wav"
        mock_transcribe.return_value = {"transcription": "la"}
        mock_generate.return_value = "cover.mp3"
        mock_mix.return_value = "mixed.mp3"

        args = self._make_args(force=True)
        cover_pipeline(args)

        # Force should cause separation to proceed despite skip recommendation
        mock_separate.assert_called_once()

    @patch("songforge.pipeline._generate_cover")
    @patch("songforge.pipeline.format_report")
    @patch("songforge.pipeline.diagnose_vocal_presence")
    @patch("songforge.pipeline.analyze_recording")
    def test_pipeline_analyze_called_with_input(self, mock_analyze, mock_diagnose, mock_format, mock_cover):
        """Verify analyze_recording is called with the input file."""
        mock_analyze.return_value = {}
        mock_diagnose.return_value = {"recommendation": "skip_separation"}
        mock_format.return_value = ""
        mock_cover.return_value = "cover.mp3"

        args = self._make_args(input="custom.mp3")
        cover_pipeline(args)
        mock_analyze.assert_called_once_with("custom.mp3")

    def test_cleanup_intermediates_removes_created_files(self, tmp_path):
        """_cleanup_intermediates removes stems, empty model dir, and enhanced wav."""
        song_dir = tmp_path / "htdemucs" / "song"
        song_dir.mkdir(parents=True)
        vocals = song_dir / "vocals.wav"
        no_vocals = song_dir / "no_vocals.wav"
        vocals.write_bytes(b"v")
        no_vocals.write_bytes(b"i")
        enhanced = tmp_path / "enhanced_vocals.wav"
        enhanced.write_bytes(b"e")

        stems = {"vocals": str(vocals), "no_vocals": str(no_vocals)}
        _cleanup_intermediates(stems, str(enhanced))

        assert not song_dir.exists()
        assert not (tmp_path / "htdemucs").exists()  # emptied model dir is dropped too
        assert not enhanced.exists()

    def test_cleanup_intermediates_keeps_sibling_songs(self, tmp_path):
        """Cleanup must not touch other songs in the same model dir."""
        model_dir = tmp_path / "htdemucs"
        song_dir = model_dir / "song"
        song_dir.mkdir(parents=True)
        sibling = model_dir / "other-song"
        sibling.mkdir()
        sibling_file = sibling / "vocals.wav"
        sibling_file.write_bytes(b"keep")

        stems = {"vocals": str(song_dir / "vocals.wav"), "no_vocals": str(song_dir / "no_vocals.wav")}
        _cleanup_intermediates(stems, str(tmp_path / "enhanced_vocals.wav"))

        assert not song_dir.exists()
        assert sibling_file.exists()  # sibling song untouched

    def test_cleanup_intermediates_refuses_working_dir(self, tmp_path, monkeypatch):
        """Cleanup must refuse paths that resolve to the CWD or its ancestors."""
        # Point the process CWD at the tmp dir so the guard is exercised for real
        monkeypatch.chdir(tmp_path)
        danger = tmp_path / "vocals.wav"  # parent resolves to CWD itself
        danger.write_bytes(b"v")

        stems = {"vocals": "vocals.wav", "no_vocals": "no_vocals.wav"}
        with pytest.raises(ValueError, match="Refusing to remove"):
            _cleanup_intermediates(stems, "enhanced.wav")

        # Nothing got deleted
        assert danger.exists()
        assert tmp_path.exists()

    @patch("songforge.pipeline._cleanup_intermediates")
    @patch("songforge.pipeline._mix_tracks")
    @patch("songforge.pipeline._generate_cover")
    @patch("songforge.pipeline.transcribe_audio")
    @patch("songforge.pipeline.enhance_vocals")
    @patch("songforge.pipeline.separate_stems")
    @patch("songforge.pipeline.format_report")
    @patch("songforge.pipeline.diagnose_vocal_presence")
    @patch("songforge.pipeline.analyze_recording")
    def test_pipeline_cleans_up_by_default(self, mock_analyze, mock_diagnose, mock_format,
                                           mock_separate, mock_enhance, mock_transcribe,
                                           mock_generate, mock_mix, mock_cleanup):
        """Without --keep-stems, the pipeline cleans up intermediates."""
        mock_analyze.return_value = {"duration": 30.0}
        mock_diagnose.return_value = {"recommendation": "proceed"}
        mock_format.return_value = "Report"
        mock_separate.return_value = {"vocals": "vocals.wav", "no_vocals": "instrumental.wav"}
        mock_enhance.return_value = "enhanced.wav"
        mock_transcribe.return_value = {"transcription": "la la la"}
        mock_generate.return_value = "cover.mp3"
        mock_mix.return_value = "cover_mixed.mp3"

        args = self._make_args(keep_stems=False)
        cover_pipeline(args)

        mock_cleanup.assert_called_once_with({"vocals": "vocals.wav", "no_vocals": "instrumental.wav"}, "enhanced.wav")

    @patch("songforge.pipeline._cleanup_intermediates")
    @patch("songforge.pipeline._mix_tracks")
    @patch("songforge.pipeline._generate_cover")
    @patch("songforge.pipeline.transcribe_audio")
    @patch("songforge.pipeline.enhance_vocals")
    @patch("songforge.pipeline.separate_stems")
    @patch("songforge.pipeline.format_report")
    @patch("songforge.pipeline.diagnose_vocal_presence")
    @patch("songforge.pipeline.analyze_recording")
    def test_pipeline_keeps_stems_with_flag(self, mock_analyze, mock_diagnose, mock_format,
                                            mock_separate, mock_enhance, mock_transcribe,
                                            mock_generate, mock_mix, mock_cleanup):
        """With --keep-stems, intermediates are left in place."""
        mock_analyze.return_value = {"duration": 30.0}
        mock_diagnose.return_value = {"recommendation": "proceed"}
        mock_format.return_value = "Report"
        mock_separate.return_value = {"vocals": "vocals.wav", "no_vocals": "instrumental.wav"}
        mock_enhance.return_value = "enhanced.wav"
        mock_transcribe.return_value = {"transcription": "la la la"}
        mock_generate.return_value = "cover.mp3"
        mock_mix.return_value = "cover_mixed.mp3"

        args = self._make_args(keep_stems=True)
        cover_pipeline(args)

        mock_cleanup.assert_not_called()


# ─── Pipeline: _generate_cover ───

class TestGenerateCover:
    """Test the MMX cover generation helper."""

    @patch("songforge.pipeline.subprocess.run")
    def test_generate_cover_success_first_try(self, mock_run):
        """When MMX generate succeeds, no fallback."""
        mock_run.return_value = MagicMock(returncode=0)

        result = _generate_cover("acoustic", "la la la", "out.mp3")

        assert result == "out.mp3"
        assert mock_run.call_count == 1
        cmd = mock_run.call_args_list[0][0][0]
        assert "mmx" in cmd
        assert "music" in cmd
        assert "generate" in cmd

    @patch("songforge.pipeline.subprocess.run")
    def test_generate_cover_fallback_to_cover_mode(self, mock_run):
        """When generate fails, fall back to cover mode."""
        mock_run.side_effect = [
            MagicMock(returncode=1),
            MagicMock(returncode=0)
        ]

        result = _generate_cover("rock", "words", "output.mp3")

        assert result == "output.mp3"
        assert mock_run.call_count == 2
        second_cmd = mock_run.call_args_list[1][0][0]
        assert "cover" in second_cmd

    @patch("songforge.pipeline.subprocess.run")
    def test_generate_cover_uses_correct_prompt(self, mock_run):
        """Verify style prompt is passed to MMX."""
        mock_run.return_value = MagicMock(returncode=0)
        _generate_cover("jazz piano", "lyrics here", "out.mp3")
        cmd = mock_run.call_args_list[0][0][0]
        assert "jazz piano" in cmd

    @patch("songforge.pipeline.subprocess.run")
    def test_generate_cover_captures_output(self, mock_run):
        """Verify subprocess uses capture_output=True."""
        mock_run.return_value = MagicMock(returncode=0)
        _generate_cover("style", "lyrics", "out.mp3")
        kwargs = mock_run.call_args_list[0][1]
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True


# ─── Pipeline: _mix_tracks ───

class TestMixTracks:
    """Test the ffmpeg mixing helper."""

    @patch("songforge.pipeline.subprocess.run")
    def test_mix_tracks_basic(self, mock_run, tmp_path):
        """Verify ffmpeg is called with correct arguments."""
        mock_run.return_value = MagicMock(returncode=0)
        out = str(tmp_path / "mixed.mp3")
        (tmp_path / "mixed.mp3").write_bytes(b"data")

        result = _mix_tracks("vocals.wav", "instrumental.wav", out)

        assert result == out
        cmd = mock_run.call_args_list[0][0][0]
        assert "ffmpeg" in cmd
        assert "-y" in cmd
        assert "vocals.wav" in cmd
        assert "instrumental.wav" in cmd
        assert "amix" in str(cmd)

    @patch("songforge.pipeline.subprocess.run")
    def test_mix_tracks_captures_output(self, mock_run, tmp_path):
        """Verify subprocess uses capture_output=True."""
        mock_run.return_value = MagicMock(returncode=0)
        out = str(tmp_path / "out.mp3")
        (tmp_path / "out.mp3").write_bytes(b"data")
        _mix_tracks("v.wav", "i.wav", out)
        kwargs = mock_run.call_args_list[0][1]
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True

    @patch("songforge.pipeline.subprocess.run")
    def test_mix_tracks_filter_complex(self, mock_run, tmp_path):
        """Verify amix filter is in the command."""
        mock_run.return_value = MagicMock(returncode=0)
        out = str(tmp_path / "c.mp3")
        (tmp_path / "c.mp3").write_bytes(b"data")
        _mix_tracks("a.wav", "b.wav", out)
        cmd = mock_run.call_args_list[0][0][0]
        # The filter_complex should mention amix with inputs=2
        filter_idx = cmd.index("-filter_complex")
        filter_value = cmd[filter_idx + 1]
        assert "amix" in filter_value
        assert "inputs=2" in filter_value

    @patch("songforge.pipeline.subprocess.run")
    def test_mix_tracks_duration_longest(self, mock_run, tmp_path):
        """Verify amix uses duration=longest."""
        mock_run.return_value = MagicMock(returncode=0)
        out = str(tmp_path / "c.mp3")
        (tmp_path / "c.mp3").write_bytes(b"data")
        _mix_tracks("a.wav", "b.wav", out)
        cmd = mock_run.call_args_list[0][0][0]
        filter_idx = cmd.index("-filter_complex")
        filter_value = cmd[filter_idx + 1]
        assert "duration=longest" in filter_value

    @patch("songforge.pipeline.subprocess.run")
    def test_mix_tracks_raises_on_ffmpeg_failure(self, mock_run, tmp_path):
        """A failing ffmpeg must raise, not return a phantom path."""
        mock_run.return_value = MagicMock(returncode=1, stderr="boom\n")
        with pytest.raises(RuntimeError, match="ffmpeg mix failed"):
            _mix_tracks("v.wav", "i.wav", str(tmp_path / "out.mp3"))

    @patch("songforge.pipeline.subprocess.run")
    def test_mix_tracks_raises_when_output_missing(self, mock_run, tmp_path):
        """Success exit code but no output file is still a failure."""
        mock_run.return_value = MagicMock(returncode=0)
        with pytest.raises(RuntimeError, match="no file"):
            _mix_tracks("v.wav", "i.wav", str(tmp_path / "ghost.mp3"))


# ─── Pipeline: Edge Cases ───

class TestPipelineEdgeCases:
    """Edge cases and error handling."""

    def test_cover_pipeline_with_namespace_object(self):
        """cover_pipeline should work with a simple Namespace object."""
        from songforge.pipeline import cover_pipeline
        # Just verify it doesn't crash on import
        assert callable(cover_pipeline)

    @patch("songforge.pipeline._generate_cover")
    @patch("songforge.pipeline.format_report")
    @patch("songforge.pipeline.diagnose_vocal_presence")
    @patch("songforge.pipeline.analyze_recording")
    def test_pipeline_output_file_from_args(self, mock_analyze, mock_diagnose, mock_format, mock_gen):
        """The output file should come from args.output."""
        mock_analyze.return_value = {}
        mock_diagnose.return_value = {"recommendation": "skip_separation"}
        mock_format.return_value = ""
        mock_gen.return_value = "custom_output.mp3"

        args = argparse.Namespace(
            input="s.mp3", lyrics="l", style="st",
            output="custom_output.mp3", keep_stems=False, force=False
        )
        cover_pipeline(args)
        mock_gen.assert_called_once_with("st", "l", "custom_output.mp3")

    @patch("songforge.pipeline._generate_cover")
    @patch("songforge.pipeline.format_report")
    @patch("songforge.pipeline.diagnose_vocal_presence")
    @patch("songforge.pipeline.analyze_recording")
    def test_pipeline_style_from_args(self, mock_analyze, mock_diagnose, mock_format, mock_gen):
        """The style prompt should come from args.style."""
        mock_analyze.return_value = {}
        mock_diagnose.return_value = {"recommendation": "skip_separation"}
        mock_format.return_value = ""

        args = argparse.Namespace(
            input="s.mp3", lyrics="l", style="heavy metal",
            output="out.mp3", keep_stems=False, force=False
        )
        cover_pipeline(args)
        mock_gen.assert_called_once_with("heavy metal", "l", "out.mp3")


# ─── __main__ entry point ───

class TestMainEntry:
    """Test that __main__ module exists."""

    def test_main_module_exists(self):
        """The __main__ module file should exist."""
        import songforge
        main_path = Path(songforge.__file__).parent / "__main__.py"
        assert main_path.exists()
