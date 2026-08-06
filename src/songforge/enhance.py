"""Vocal enhancement using ffmpeg."""

import subprocess

def enhance_vocals(
    input_file: str,
    output_file: str = "enhanced.wav",
    volume: float = 3.0,
    eq_freq: int = 2000,
    denoise: bool = False,
) -> str:
    """Enhance vocal quality with volume, EQ, and optional denoise."""
    filters = [f"volume={volume}"]
    filters.append(f"equalizer=f={eq_freq}:width_type=h:width={eq_freq//2}:g=5")
    
    if denoise:
        filters.append("afftdn=nr=10:nf=-40")
    
    filter_chain = ",".join(filters)
    
    print(f"Enhancing: {filter_chain}")
    
    result = subprocess.run([
        "ffmpeg", "-y", "-i", input_file,
        "-af", filter_chain,
        output_file
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    print(f"  Enhanced: {output_file}")
    return output_file
