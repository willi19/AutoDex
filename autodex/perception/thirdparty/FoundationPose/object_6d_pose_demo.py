import argparse
import glob
import os
import sys

from PIL import Image
import torch


def _ensure_sam3_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sam3_path = os.path.join(repo_root, "sam3")
    if sam3_path not in sys.path:
        sys.path.insert(0, sam3_path)


def _load_sam3(device: str, confidence_threshold: float):
    _ensure_sam3_on_path()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)
    return processor


def _pick_best_mask(masks: torch.Tensor, scores: torch.Tensor):
    if masks is None or masks.numel() == 0:
        return None
    if scores is None or scores.numel() == 0:
        idx = 0
    else:
        idx = int(torch.argmax(scores).item())
    mask = masks[idx]
    if mask.ndim == 3:
        mask = mask[0]
    return mask


def generate_masks(data_dir: str, text_prompt: str, device: str, confidence_threshold: float):
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    image_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"):
        image_paths.extend(sorted([p for p in glob.glob(os.path.join(images_dir, ext))]))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    processor = _load_sam3(device=device, confidence_threshold=confidence_threshold)

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        mask = _pick_best_mask(output.get("masks"), output.get("scores"))
        if mask is None:
            mask_np = torch.zeros((image.height, image.width), dtype=torch.uint8)
        else:
            mask_np = mask.to(torch.uint8) * 255

        mask_np = mask_np.detach().cpu().numpy()
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(masks_dir, f"{base_name}.png")
        Image.fromarray(mask_np).save(mask_path)


def build_argparser():
    parser = argparse.ArgumentParser(description="FoundationPose demo: SAM3 mask generation")
    parser.add_argument(
        "--data-dir",
        default="/media/gunhee/DATA/robothome/FoundationPose/demo_data/baby_beaker_demo",
        help="Path to demo data directory containing images/",
    )
    parser.add_argument(
        "--text-prompt",
        default="object on the checkerboard, excluding checkerboard",
        help="Text prompt for SAM3 mask generation",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for SAM3 (cuda or cpu)")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for SAM3 masks",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    generate_masks(
        data_dir=args.data_dir,
        text_prompt=args.text_prompt,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
