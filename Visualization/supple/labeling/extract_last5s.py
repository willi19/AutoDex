#!/usr/bin/env python3
"""
Extract last 5 seconds from videos and save as mp4
"""
import os
import subprocess
import glob

# 설정
INPUT_DIR = "/home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100/attached_container/20260128_151516/videos"
OUTPUT_DIR = "/home/mingi/RSS_2026/Visualization/supple/labeling/fail"
LAST_SECONDS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 모든 avi 파일 찾기
video_files = glob.glob(os.path.join(INPUT_DIR, "*.avi"))
print(f"Found {len(video_files)} videos")

for video_path in sorted(video_files):
    video_name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(video_name)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{name_without_ext}.mp4")

    # 1. 영상 길이 가져오기
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )

    try:
        duration = float(result.stdout.strip())
    except ValueError:
        print(f"  ERROR: Could not get duration for {video_name}")
        continue

    # 2. 시작 시간 계산 (마지막 5초)
    start_time = max(0, duration - LAST_SECONDS)

    print(f"Processing {video_name}: duration={duration:.2f}s, extracting from {start_time:.2f}s")

    # 3. ffmpeg으로 추출 및 변환
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(LAST_SECONDS),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",  # 오디오 제거
        output_path
    ]

    subprocess.run(cmd, capture_output=True)

    if os.path.exists(output_path):
        print(f"  -> Saved: {output_path}")
    else:
        print(f"  -> FAILED: {video_name}")

print(f"\nDone! Output saved to {OUTPUT_DIR}")
