import os
import cv2
from pathlib import Path
from rsslib.path import code_path

def extract_frame_from_experiments(base_path, frame_offset=10):
    """
    Extract frame at (total_frames - offset) from all experiment videos
    
    Args:
        base_path: Base experiment directory path
        frame_offset: Offset from the end (default: 10, means 10th frame from the end)
    """
    base_dir = Path(base_path)
    output_base = Path(code_path) / "Visualization" / "paper" / "figure1" / "step3"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Find all experiment directories (datestr format: YYYYMMDD_HHMMSS)
    exp_dirs = [d for d in base_dir.iterdir() if d.is_dir() and len(d.name) == 15]
    exp_dirs.sort()
    
    for idx, exp_dir in enumerate(exp_dirs):
        video_path = exp_dir / "videos" / "25322639.avi"
        
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            continue
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate target frame (10th from the end)
        target_frame = total_frames - frame_offset
        
        if target_frame < 0:
            print(f"Warning: {exp_dir.name} has only {total_frames} frames")
            target_frame = 0
        
        # Set position to target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        # Read frame
        ret, frame = cap.read()
        if ret:
            output_path = output_base / f"{idx}.png"
            cv2.imwrite(str(output_path), frame)
            print(f"Saved {idx}.png from {exp_dir.name} (frame {target_frame}/{total_frames})")
        else:
            print(f"Failed to read frame from {exp_dir.name}")
        
        cap.release()

if __name__ == "__main__":
    base_path = "/home/mingi/shared_data/RSS2026_Mingi/experiment/fourcam/attached_container"
    extract_frame_from_experiments(base_path, frame_offset=1)