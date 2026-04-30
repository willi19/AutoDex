"""Synthesize fr3_inspire_f1.urdf by grafting inspire_f1 hand onto FR3.

Strategy:
- Take FR3 portion of fr3_inspire_left.urdf (up to and including the
  `flange_to_hand` joint that mounts at fr3_link8). Drop the inspire_left
  hand body that follows.
- Append inspire_f1_hand_right.urdf body (skip outer <robot> wrapper).
- Update mesh filename refs to point at the shared `meshes/` directory.

Re-runnable; does not modify source URDFs.
"""
from pathlib import Path
import re

ROOT = Path("/home/mingi/AutoDex/autodex/planner/src/curobo/content/assets/robot")
SRC_FR3_INSPIRE_LEFT = ROOT / "fr3_inspire_left_description" / "fr3_inspire_left.urdf"
SRC_INSPIRE_F1_HAND  = ROOT / "inspire_f1_description" / "inspire_f1_hand_right.urdf"
DST_DIR = ROOT / "fr3_inspire_f1_description"
DST_URDF = DST_DIR / "fr3_inspire_f1.urdf"

fr3_text = SRC_FR3_INSPIRE_LEFT.read_text()
hand_text = SRC_INSPIRE_F1_HAND.read_text()

# 1) Cut fr3_inspire_left content right after the </joint> for flange_to_hand.
m = re.search(r'<joint name="flange_to_hand".*?</joint>', fr3_text, re.S)
if m is None:
    raise SystemExit("flange_to_hand joint not found in fr3_inspire_left.urdf")
fr3_keep = fr3_text[: m.end()]  # FR3 arm + flange_to_hand joint
# inspire_left mount used `rpy="-3.14159 0 0.7854"` (180° x-flip + 45° yaw)
# because its base_link +z points back into the arm. inspire_f1's base_link
# +z already points forward (out of the wrist), so no rotation is needed —
# the original flip would make the hand point INTO fr3_link7. Drop the
# rotation entirely and keep only the 4cm flange offset.
fr3_keep = fr3_keep.replace(
    'rpy="-3.14159 0 0.7854"', 'rpy="0 0 0"',
)

# 2) Strip inspire_f1 outer <robot>...</robot> wrapper and pull inner body.
m_inner = re.search(
    r'<robot[^>]*>(.*)</robot>', hand_text, re.S,
)
if m_inner is None:
    raise SystemExit("could not isolate <robot> body in inspire_f1_hand_right.urdf")
hand_body = m_inner.group(1).strip()

# 3) Mesh paths. inspire_f1_hand_right uses `./meshes/*.STL`. Both source meshes
# already copied to fr3_inspire_f1_description/meshes/. Keep relative `meshes/`
# refs to match yourdfpy mesh_dir behavior.
hand_body = hand_body.replace('"./meshes/', '"meshes/')

# 4) Compose. fr3_inspire_left.urdf's <robot> tag is preserved at the top of
# fr3_keep, but we need to give it a new name and append a closing </robot>.
out = fr3_keep
out = re.sub(r'<robot\s+name="[^"]+"', '<robot name="fr3_inspire_f1"', out, count=1)
# inspire_f1 hand body starts with its own <link name="base_link"> — that's
# the same base_link that flange_to_hand expects as child, so it just plugs in.
out = out + "\n" + hand_body + "\n</robot>\n"

DST_URDF.write_text(out)

# Sanity check
n_link = len(re.findall(r"<link\b", out))
n_joint = len(re.findall(r"<joint\b", out))
print(f"wrote {DST_URDF}  links={n_link}  joints={n_joint}  bytes={len(out)}")
