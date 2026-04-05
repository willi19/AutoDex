#!/usr/bin/env python3
"""Generate CSV for Google Sheet: experiment results per object.

Usage:
    python scripts/update_sheet.py                # combined (default)
    python scripts/update_sheet.py --hand allegro # one hand only
    python scripts/update_sheet.py --save         # save to scripts/*.csv
"""
import argparse
import json
import os
import glob

EXPERIMENT_BASE = os.path.expanduser("~/shared_data/AutoDex/experiment/selected_100")
CANDIDATE_BASE = os.path.expanduser("~/shared_data/AutoDex/candidates")

# Known issue boundaries
ISSUE_BOUNDARIES = {
    "pre-snap_z": "20260327",      # TABLE_SURFACE_Z fix
    "fast-lift": ("20260327", "20260330"),  # before lift speed fix
    "no-smooth": ("20260403_013400", "20260403_143400"),  # v4: no smoothing
}

CYLINDER_BUGGY = ["pepper_tuna", "pepper_tuna_light", "pepsi", "pepsi_light"]

# Fixed object order matching Google Sheet
OBJECT_ORDER = [
    "attached_container", "box_pink", "clock", "organizer_beige", "toothbrush_holder",
    "bamboo_box", "servingbowl_small", "smallbowl", "soaptray", "thermo_clock",
    "container_pink", "lemon_squeezer", "brown_ramen", "banana", "donut",
    "yellow_funnel", "coffee_tin", "baby_beaker", "icecream_scoop", "spicemill",
    "open_box", "plant_mister", "wateringcan", "mug_holder", "magazine_file",
    "blue_alarm", "donut_light", "soap_dispenser", "colander_green", "standing_frame",
    "washing_brush2", "yellow_plastic_glass", "large_peg", "screwdriver", "knife_sharpner",
    "black_kitchen_roll_holder", "frame_oak", "frame_red", "green_lamp", "green_soap_dispenser",
    "lilac_food_container", "potato_mesher", "rolling_pin", "washing_brush", "white_hand_shower",
    "white_pen_cup", "white_plastic_box", "wood_tray_big", "wood_tray_small", "work_lamp",
    "yellow_plastic_cup", "blue_vase", "green_attached_container", "redcar", "wood_organizer",
    "brass_pot", "green_cactus_vase", "light_green_basket", "pink_clock", "plant_pot",
    "spray_bottle", "toilet_roll_holder_steel", "white_candle_holder", "white_clock",
    "white_soap_dish", "white_table_lamp", "white_watering_can", "yellow_wash_tub",
    "frog_bowl", "fruit_cutter_base", "fruit_cutter_green", "fruit_cutter_light_green",
    "metal_scoop_big", "pastel_blue_cup", "balloon_whisk", "beige_brush",
    "black_holder_with_handle", "corkscrew", "white_wood_handle_watering_can",
    "metal_scoop_small", "lemon", "apple", "orange", "frog_cup",
    "open_short_pringles", "pepper_tuna", "paper_cup", "french_mustard", "spam_can",
    "paper_box", "paper_bowl", "shoe_organizer", "tissue_box", "tennis_ball",
    "tea_case", "jja_ramen", "pepsi", "baseball", "pingpong", "book",
    "pepper_tuna_light", "pepsi_light",
]


def get_results(hand):
    """Scan experiment dirs and return {obj: {success, fail, issues}} counts."""
    results = {}

    hand_dir = os.path.join(EXPERIMENT_BASE, hand)
    if not os.path.isdir(hand_dir):
        return results

    for obj in os.listdir(hand_dir):
        obj_dir = os.path.join(hand_dir, obj)
        if not os.path.isdir(obj_dir):
            continue
        # candidate_key -> (ever_succeeded, earliest_trial_date)
        candidates = {}
        for rfile in glob.glob(os.path.join(obj_dir, "*/result.json")):
            try:
                d = json.load(open(rfile))
            except Exception:
                continue
            s = d.get("success")
            if s is None:
                continue
            si = d.get("scene_info")
            if not si:
                continue
            key = tuple(si)
            trial_date = os.path.basename(os.path.dirname(rfile))
            if key not in candidates:
                candidates[key] = (s, trial_date)
            elif s and not candidates[key][0]:
                candidates[key] = (True, trial_date)

        if not candidates:
            continue

        n_succ = sum(1 for v, _ in candidates.values() if v)
        n_fail = sum(1 for v, _ in candidates.values() if not v)

        # Issue detection based on unique candidate dates
        unique_dates = [d for _, d in candidates.values()]
        issues = []
        pre_snap = sum(1 for d in unique_dates if d < ISSUE_BOUNDARIES["pre-snap_z"])
        if pre_snap:
            issues.append(f"pre-snap_z:{pre_snap}")
        fl_start, fl_end = ISSUE_BOUNDARIES["fast-lift"]
        fast_lift = sum(1 for d in unique_dates if fl_start <= d < fl_end)
        if fast_lift:
            issues.append(f"fast-lift:{fast_lift}")
        ns_start, ns_end = ISSUE_BOUNDARIES["no-smooth"]
        no_smooth = sum(1 for d in unique_dates if ns_start <= d < ns_end)
        if no_smooth:
            issues.append(f"no-smooth:{no_smooth}")
        if obj in CYLINDER_BUGGY:
            issues.append("cylinder-snap-buggy")

        results[obj] = {
            "success": n_succ,
            "fail": n_fail,
            "issues": "; ".join(issues) if issues else "",
        }

    return results


def has_candidates(hand, obj):
    p = os.path.join(CANDIDATE_BASE, hand, "selected_100", obj)
    return os.path.isdir(p)


def generate_single_csv(hand):
    results = get_results(hand)
    extra = [o for o in sorted(results.keys()) if o not in OBJECT_ORDER]
    obj_list = OBJECT_ORDER + [o for o in extra if o not in OBJECT_ORDER]

    lines = []
    lines.append(",".join(["Name", "Succ", "Fail", "Tot", "In candidates", "Issues"]))

    for obj in obj_list:
        in_cand = "O" if has_candidates(hand, obj) else ""
        r = results.get(obj)
        if r and (r["success"] + r["fail"]) > 0:
            tot = r["success"] + r["fail"]
            issues = f'"{r["issues"]}"' if r["issues"] else ""
            lines.append(",".join([obj, str(r["success"]), str(r["fail"]), str(tot), in_cand, issues]))
        else:
            lines.append(",".join([obj, "", "", "", in_cand, ""]))

    # Summary rows
    count = sum(1 for o in obj_list if results.get(o) and (results[o]["success"] + results[o]["fail"]) > 0)
    succ_sum = sum(r["success"] for r in results.values())
    fail_sum = sum(r["fail"] for r in results.values())
    tot_sum = succ_sum + fail_sum
    lines.append(",".join(["Count", "", "", str(count), "", ""]))
    lines.append(",".join(["Total", str(succ_sum), str(fail_sum), str(tot_sum), "", ""]))

    return "\n".join(lines)


def generate_combined_csv():
    allegro = get_results("allegro")
    inspire = get_results("inspire")

    all_objs = set(list(allegro.keys()) + list(inspire.keys()))
    extra = [o for o in sorted(all_objs) if o not in OBJECT_ORDER]
    obj_list = OBJECT_ORDER + [o for o in extra if o not in OBJECT_ORDER]

    lines = []
    lines.append(",".join([
        "Name",
        "A_Tot", "I_Tot",
        "A_SR", "I_SR",
    ]))

    for obj in obj_list:
        a = allegro.get(obj)
        i = inspire.get(obj)

        if a:
            a_tot = a["success"] + a["fail"]
            a_sr = f'{a["success"]}/{a_tot}'
            a_t = str(a_tot)
        else:
            a_sr = ""
            a_t = ""
        if i:
            i_tot = i["success"] + i["fail"]
            i_sr = f'{i["success"]}/{i_tot}'
            i_t = str(i_tot)
        else:
            i_sr = ""
            i_t = ""

        lines.append(",".join([obj, a_t, i_t, a_sr, i_sr]))

    # Summary rows
    a_count = sum(1 for o in obj_list if allegro.get(o))
    i_count = sum(1 for o in obj_list if inspire.get(o))
    a_succ = sum(a["success"] for a in allegro.values())
    a_tot = sum(a["success"] + a["fail"] for a in allegro.values())
    i_succ = sum(i["success"] for i in inspire.values())
    i_tot = sum(i["success"] + i["fail"] for i in inspire.values())
    lines.append(",".join(["Count", str(a_count), str(i_count), "", ""]))
    lines.append(",".join(["Total", str(a_tot), str(i_tot),
                           f"{a_succ}/{a_tot}" if a_tot else "",
                           f"{i_succ}/{i_tot}" if i_tot else ""]))

    return "\n".join(lines)


SHEET_URL = "https://docs.google.com/spreadsheets/d/1MeBllZiPjIQe7862RUO3LNaZ8qmU8paIeVCIHtQ6a7o"
CREDS_PATH = os.path.expanduser("~/.config/gcloud/autodex-sheet.json")


def upload_to_sheet():
    """Upload combined + individual results to Google Sheet."""
    import gspread

    gc = gspread.service_account(filename=CREDS_PATH)
    sh = gc.open_by_url(SHEET_URL)

    def _csv_to_rows(csv_str):
        import csv
        import io
        rows = []
        for row in csv.reader(io.StringIO(csv_str)):
            converted = []
            for cell in row:
                try:
                    converted.append(int(cell))
                except ValueError:
                    converted.append(cell)
            rows.append(converted)
        return rows

    # Combined sheet
    combined = _csv_to_rows(generate_combined_csv())
    try:
        ws = sh.worksheet("Combined")
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet("Combined", rows=len(combined) + 10, cols=10)
    ws.clear()
    ws.update(combined, "A1")
    print(f"Uploaded Combined ({len(combined)} rows)")

    # Individual sheets
    for hand in ["allegro", "inspire"]:
        rows = _csv_to_rows(generate_single_csv(hand))
        title = hand.capitalize()
        try:
            ws = sh.worksheet(title)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title, rows=len(rows) + 10, cols=8)
        ws.clear()
        ws.update(rows, "A1")
        print(f"Uploaded {title} ({len(rows)} rows)")


def main():
    upload_to_sheet()


if __name__ == "__main__":
    main()
