from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "video" / "yinbian-privacy-demo"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def run(*args: str) -> None:
    subprocess.run([FFMPEG, "-hide_banner", "-y", *args], check=True)


def silence(path: Path, seconds: float, sample_rate: int = 22050) -> None:
    frames = round(seconds * sample_rate)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\0\0" * frames)


def srt_time(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def main() -> None:
    generated = json.loads((OUT / "audio" / "scenes.generated.json").read_text("utf-8-sig"))
    gap = OUT / "audio" / "gap.wav"
    silence(gap, 0.65)
    concat = OUT / "audio" / "concat.txt"
    lines: list[str] = []
    current = 0.0
    srt: list[str] = []
    for index, scene in enumerate(generated, 1):
        audio = Path(scene["audio"])
        lines.append(f"file '{audio.as_posix()}'")
        lines.append(f"file '{gap.as_posix()}'")
        duration = float(scene["duration_seconds"])
        srt.extend(
            [
                str(index),
                f"{srt_time(current)} --> {srt_time(current + duration)}",
                scene["narration"],
                "",
            ]
        )
        current += duration + 0.65
    concat.write_text("\n".join(lines), encoding="utf-8")
    (OUT / "Yinbian_Privacy_Demo_zh-CN.srt").write_text("\n".join(srt), encoding="utf-8")
    narration = OUT / "audio" / "narration.wav"
    run("-f", "concat", "-safe", "0", "-i", str(concat), "-c:a", "pcm_s16le", str(narration))
    raw = OUT / "raw" / "screen.webm"
    final = OUT / "Yinbian_Privacy_Inference_Demo_zh-CN.mp4"
    run(
        "-i", str(raw), "-i", str(narration),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "21",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-shortest", "-movflags", "+faststart", str(final),
    )
    report = {
        "video": str(final),
        "subtitles": str(OUT / "Yinbian_Privacy_Demo_zh-CN.srt"),
        "planned_seconds": round(current, 3),
        "scenes": len(generated),
        "resolution": "1600x900",
    }
    (OUT / "build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), "utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
