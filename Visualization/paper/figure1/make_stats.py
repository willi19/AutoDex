import os
import json
import numpy as np
from tqdm import tqdm
from rsslib.path import code_path, candidate_path

# -----------------------------------------------------------------------------
# 1. 설정 및 Valid Array 로드
# -----------------------------------------------------------------------------
obj_name = "attached_container"
grasp_version = "v2"

# Valid Array (이건 무조건 필요함, Scene x Grasp 정답지)
valid_array_path = os.path.join(code_path, "order", grasp_version, obj_name, "valid_array.npy")
valid_array = np.load(valid_array_path)
total_scenes, total_grasps = valid_array.shape

# -----------------------------------------------------------------------------
# 2. 인덱스 매핑 (Valid Array의 몇 번째 컬럼인지 찾기 위해)
# -----------------------------------------------------------------------------
def get_index_mapping_fixed(obj_name, version):
    grasp_map = {}
    idx_counter = 0
    
    root_dir = os.path.join(candidate_path, version, obj_name)
    # [수정] sorted()를 절대 쓰지 마세요. load_grasp_data와 순서를 맞춰야 합니다.
    for scene_type in os.listdir(root_dir):
        scene_grasp_path = os.path.join(root_dir, scene_type)
        for scene_name in os.listdir(scene_grasp_path):
            grasp_scene_path = os.path.join(scene_grasp_path, scene_name)
            for grasp_name in os.listdir(grasp_scene_path):
                key = (str(scene_type), str(scene_name), str(grasp_name))
                grasp_map[key] = idx_counter
                idx_counter += 1
    return grasp_map

print("Building index map...")
grasp_idx_map = get_index_mapping_fixed(obj_name, grasp_version)

# -----------------------------------------------------------------------------
# 3. setcover_order.json 로드 (여기가 핵심)
# -----------------------------------------------------------------------------
order_json_path = os.path.join(code_path, "order", grasp_version, obj_name, "setcover_order.json")
if not os.path.exists(order_json_path):
    raise FileNotFoundError(f"Order file not found: {order_json_path}")

print(f"Loading order from: {order_json_path}")
order_info_list = json.load(open(order_json_path, 'r'))
# order_info_list 구조 예상: [[obj, ver, type, id, name], ...]

# -----------------------------------------------------------------------------
# 4. 순서대로 순회하며 Coverage 계산 + Reset Logic
# -----------------------------------------------------------------------------
output_stats = []
uncovered_mask = np.ones(total_scenes, dtype=bool)
cycle_count = 1

print(f"Processing {len(order_info_list)} steps from setcover order...")

for step_info in tqdm(order_info_list):
    # step_info: ["attached_container", "visualization", "shelf", "01", "00"]
    # Key 추출
    key = (str(step_info[2]), str(step_info[3]), str(step_info[4]))
    
    if key not in grasp_idx_map:
        print(f"Error: Could not find index for {key}")
        continue
        
    g_idx = grasp_idx_map[key]
    
    # Coverage 계산
    grasp_coverage = valid_array[:, g_idx]
    newly_covered_mask = grasp_coverage & uncovered_mask
    newly_covered_count = np.sum(newly_covered_mask)
    
    current_uncovered = np.sum(uncovered_mask) - newly_covered_count
    coverage_pct = (1 - (current_uncovered / total_scenes)) * 100
    
    # 저장
    output_stats.append({
        "scene_info": step_info,     # 원본 정보
        "grasp_idx": int(g_idx),     # 매핑된 인덱스
        "newly_covered_count": int(newly_covered_count),
        "current_coverage_pct": float(coverage_pct),
        "cycle": cycle_count
    })
    
    # 마스크 업데이트
    uncovered_mask &= ~grasp_coverage
    
    # [RESET] 다 채웠으면 리셋
    if np.sum(uncovered_mask) == 0:
        uncovered_mask = np.ones(total_scenes, dtype=bool)
        cycle_count += 1

# -----------------------------------------------------------------------------
# 5. 파일 저장
# -----------------------------------------------------------------------------
save_path = os.path.join(code_path, "order", grasp_version, obj_name, "experiment_sequential_stats.json")

output_json = {
    "object": obj_name,
    "total_scenes": total_scenes,
    "steps": output_stats
}

with open(save_path, "w") as f:
    json.dump(output_json, f, indent=2)

print(f"\nSaved stats to: {save_path}")
print(f"Total Cycles: {cycle_count}")