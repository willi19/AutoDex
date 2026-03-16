# autodex.perception

Perception module for AutoDex: mask segmentation, depth estimation, and 6D pose tracking.

## Module Structure

```
autodex/perception/
├── __init__.py              # Exports: get_mask_yoloe, get_mask_sam3, get_depth_stereo, get_depth_da3, PoseTracker
├── mask.py                  # Segmentation (YOLO-E single-frame, SAM3 video)
├── depth.py                 # Depth estimation (FoundationStereo PyTorch/TRT/ONNX, Depth-Anything-3)
├── pose.py                  # 6D pose tracking (FoundationPose wrapper)
├── stereo_video_depth.py    # CLI batch stereo depth from video pairs (TensorRT)
└── thirdparty/              # External model code + weights
    ├── object-6d-tracking/  # Juncong's tracking system (has its own CLAUDE.md)
    ├── _object_6d_tracking/ # Gunhee's fork (FoundationPose, Depth-Anything-3, sam3)
    ├── FoundationStereo/    # Stereo depth (PyTorch + TRT engine at output/*.engine)
    ├── sam3/                # SAM3 video predictor
    └── weights/             # Model weights (yoloe-26x-seg.pt, mobileclip2_b.ts)
```

## Public API

### Mask Generation (`mask.py`)
- `YOLOE_WEIGHTS` — path constant to `thirdparty/weights/yoloe-26x-seg.pt`. Import this instead of hardcoding.
- `get_mask_yoloe(rgb, target_class, model=None, conf_thr=0.2)` -> uint8 mask (H,W) 0/255 or None
  - Single-frame YOLO-E segmentation. Returns highest-confidence detection.
  - Pass pre-loaded model to avoid reloading weights each call.
- `get_mask_sam3(images_folder, text_prompt, predictor=None, gpu=0)` -> dict[frame_idx, bool mask]
  - Video segmentation via SAM3. Propagates from frame 0 text prompt.

### Depth Estimation (`depth.py`)
- `get_depth_stereo(left, right, model, K, baseline, height=None, width=None)` -> depth (H,W) meters
  - FoundationStereo via TRT/ONNX runtime. Requires pre-loaded model.
- `get_depth_stereo_pytorch(left, right, K, baseline, model=None)` -> depth (H,W) meters
  - FoundationStereo pure PyTorch. Auto-loads model if None.
- `get_depth_da3(images, intrinsics=None, model=None, process_res=504)` -> list[depth] meters
  - Monocular depth via Depth-Anything-3. Batch inference.
- `encode_depth_uint16(depth)` / `decode_depth_uint16(bgr)` — FFV1 uint16 mm encoding.
- `load_cam_param(capture_dir)` -> (intrinsics, extrinsics) dicts keyed by serial.
- `_auto_order_stereo(K_l, K_r, T_l, T_r)` -> bool (True if cameras should be swapped).
- `_to_4x4(T)` — convert 3x4 to 4x4 extrinsic matrix.
- **IMPORTANT**: Stereo disparity is only for the LEFT image. The pipeline converts disparity → 3D world points (via left camera frame) → reprojects to all cameras.
- **IMPORTANT**: When un-rectifying stereo depth back to original camera coordinates, the Z_rect from `depth = f*B/disp` must be divided by `rz` (Z component of `R1 @ K_inv @ [u,v,1]`) to get Z_orig. Without this, large R1 rotations cause systematic depth errors.

### Pose Tracking (`pose.py`)
- `PoseTracker(mesh_path, device_id=0)` - FoundationPose wrapper
  - `.init(rgb, depth, mask, K)` -> 4x4 pose matrix (first frame registration)
  - `.track(rgb, depth, K)` -> 4x4 pose matrix (subsequent frames)
  - `.reset()` - clear tracking state

### Stereo Video Depth CLI (`stereo_video_depth.py`)
```bash
python -m autodex.perception.stereo_video_depth \
    --base /path/to/captures --left_serial 22641005 --right_serial 22641023
```
- Batch processes `{obj}/{idx}/` dirs: rectify stereo pairs -> TRT disparity -> depth AVI (FFV1, uint16 mm encoded as BGR)

## Thirdparty Path Convention
- `_JUNC0NG` = `thirdparty/object-6d-tracking` (note: variable uses zero not letter O)
- `_GUNHEE` = `thirdparty/_object_6d_tracking`
- Models are loaded lazily; paths are added to `sys.path` at call time.

## Key Dependencies
- PyTorch, CUDA, nvdiffrast, trimesh, OpenCV
- ultralytics (YOLO-E), tensorrt + pycuda (TRT inference)
- FoundationPose requires mycpp C++ extension built in `_object_6d_tracking/FoundationPose/mycpp/build`

## Conda Environments
- `foundation_stereo`: FoundationStereo TRT (`tensorrt` + `pycuda`)
- `foundationpose`: FoundationPose, YOLOE
- `sam3`: SAM3 segmentation, Depth-Anything-3

---

## Planned Refactoring (WIP)

### mask.py → Class-Based Redesign
Current `get_mask_yoloe()` / `get_mask_sam3()` are single-call functions that reload models unless you pass them in.
Plan: create `YoloeSegmentor` and `Sam3Segmentor` classes with a shared interface.

### SAM3 Memory Issue
**Root cause**: SAM3 video predictor caches ALL frame features on GPU for temporal propagation.
- realdex code works fine: samples ~1/40 frames → only 10–15 frames loaded
- Our batch_mask.py processes full videos (300–550 frames) → OOM

**Chunked tracking idea**: Process video in chunks (e.g. 50 frames), carry mask from last frame of chunk N as box prompt to first frame of chunk N+1.

### stereo_video_depth.py vs src/process/batch_depth.py Gap
`batch_depth.py` has advanced features missing from `stereo_video_depth.py`:
- Auto left/right swap, R1 un-rectify, cross-view reprojection, PLY export

Plan: consolidate into `autodex/perception/`, keep `stereo_video_depth.py` as simple CLI.

**Note**: `load_cam_param` / `build_rectify_maps` are paradex-specific camera utilities. They stay in the depth scripts.

### File I/O Utilities Deduplication
`_get_cache_base()`, `CACHE_ROOT`, `NETWORK_PREFIX` are duplicated in 4+ files under `src/process/`.
Plan: move to `autodex/utils/file_io.py`.
