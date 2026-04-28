#!/usr/bin/env python3
"""Monitor foundpose batch onboarding across robot + capture1-6.

Shows tqdm-style progress bars + render-phase ETA per active object and
per-PC chunk ETA estimate.

Usage:
    watch -n 5 python scripts/monitor_foundpose_onboard.py
"""
from __future__ import annotations
import math
import time
from pathlib import Path

ASSETS = Path.home() / "shared_data/AutoDex/foundpose_assets"
N_VIEWS = 798
POST_RENDER_S = 8 * 60        # rough: feature/PCA/cluster after render
PER_OBJ_TOTAL_S = 30 * 60     # rough: render ~22 min + post ~8 min
WORKERS_ASSUMED = 3           # used for chunk ETA math

CHUNKS = {
    "robot":    ["fruit_cutter_green", "fruit_cutter_light_green", "green_attached_container",
                 "green_cactus_vase", "green_lamp", "green_soap_dispenser", "icecream_scoop",
                 "jja_ramen", "knife_sharpner", "large_peg"],
    "capture1": ["lemon", "lemon_squeezer", "light_green_basket", "lilac_food_container",
                 "magazine_file", "meat_thermometer", "metal_scoop_big", "metal_scoop_small",
                 "mug_holder", "open_box"],
    "capture2": ["open_short_pringles", "orange", "organizer_beige", "paper_bowl", "paper_box",
                 "paper_cup", "pastel_blue_cup", "pepper_tuna", "pepper_tuna_light", "pepsi"],
    "capture3": ["pepsi_light", "pingpong", "pink_clock", "plant_mister", "plant_pot",
                 "potato_mesher", "pringles", "redcar", "rolling_pin", "screwdriver"],
    "capture4": ["servingbowl_small", "shoe_organizer", "smallbowl", "soap_dispenser",
                 "soaptray", "spam_can", "spicemill", "spray_bottle", "standing_frame", "tea_case"],
    "capture5": ["tennis_ball", "thermo_clock", "tissue_box", "toilet_roll_holder_steel",
                 "toothbrush_holder", "washing_brush", "washing_brush2", "wateringcan",
                 "white_candle_holder", "white_clock"],
    "capture6": ["white_hand_shower", "white_pen_cup", "white_plastic_box", "white_soap_dish",
                 "white_table_lamp", "white_watering_can", "white_wood_handle_watering_can",
                 "wood_organizer", "wood_tray_big", "wood_tray_small"],
}

CHUNK_BAR_W = 20
OBJ_BAR_W = 30


def is_done(obj: str) -> bool:
    return (ASSETS / obj / "object_repre/v1" / obj / "1" / "repre.pth").exists()


def rgb_dir(obj: str) -> Path:
    return ASSETS / obj / "templates/v1" / obj / "1" / "rgb"


def render_count(obj: str) -> int:
    rgb = rgb_dir(obj)
    if not rgb.exists():
        return 0
    try:
        return sum(1 for _ in rgb.iterdir())
    except OSError:
        return 0


def render_eta_s(obj: str, n: int) -> float | None:
    """Estimated remaining render-phase seconds based on actual rate so far."""
    if n < 5:
        return None
    first = rgb_dir(obj) / "template_0000.png"
    if not first.exists():
        return None
    elapsed = time.time() - first.stat().st_mtime
    if elapsed <= 0:
        return None
    rate = n / elapsed         # templates per second
    return (N_VIEWS - n) / rate


def fmt_eta(s: float | None) -> str:
    if s is None or s < 0:
        return "?"
    if s < 90:
        return f"{int(s):>2}s"
    m = s / 60
    if m < 90:
        return f"{int(m):>2}m"
    h = m / 60
    return f"{h:.1f}h"


def bar(done: int, total: int, width: int) -> str:
    if total == 0:
        return "[" + "─" * width + "]"
    filled = int(width * done / total)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def main():
    total_target = sum(len(v) for v in CHUNKS.values())
    total_done = sum(1 for objs in CHUNKS.values() for o in objs if is_done(o))
    total_pct = 100 * total_done / total_target if total_target else 0

    print(f"FoundPose onboard  {time.strftime('%H:%M:%S')}")
    print(f"OVERALL  {bar(total_done, total_target, CHUNK_BAR_W)} "
          f"{total_done:>3}/{total_target} {total_pct:5.1f}%")
    print()

    for pc, objs in CHUNKS.items():
        done_objs = [o for o in objs if is_done(o)]
        active = [(o, render_count(o)) for o in objs
                  if not is_done(o) and 0 < render_count(o) < N_VIEWS]
        n_pending_idle = len(objs) - len(done_objs) - len(active)

        # Chunk ETA: max(active render eta) + post_render + remaining_batches * per_obj
        active_etas_full = []
        for o, n in active:
            r_eta = render_eta_s(o, n) or 0
            active_etas_full.append(r_eta + POST_RENDER_S)
        max_active = max(active_etas_full) if active_etas_full else 0
        n_remaining_after_active = max(0, n_pending_idle)
        future_batches = math.ceil(n_remaining_after_active / WORKERS_ASSUMED)
        chunk_eta = max_active + future_batches * PER_OBJ_TOTAL_S

        chunk_pct = 100 * len(done_objs) / len(objs)
        eta_str = fmt_eta(chunk_eta) if (active or n_pending_idle) else "done"
        print(f"{pc:<9} {bar(len(done_objs), len(objs), CHUNK_BAR_W)} "
              f"{len(done_objs):>2}/{len(objs)} {chunk_pct:5.1f}%  ETA {eta_str}")
        for o, n in active:
            opct = 100 * n / N_VIEWS
            r_eta = render_eta_s(o, n)
            print(f"           {bar(n, N_VIEWS, OBJ_BAR_W)} "
                  f"{n:>3}/{N_VIEWS} {opct:5.1f}%  {fmt_eta(r_eta):>4} render  {o}")


if __name__ == "__main__":
    main()
