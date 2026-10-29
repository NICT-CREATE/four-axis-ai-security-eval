import os, sys, json, shutil

TEST_ROOT = sys.argv[1] if len(sys.argv) > 1 else "outputs/demo/test_v2/with_syspro"
TEST_ROOT = os.path.normpath(TEST_ROOT)
BASE_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(TEST_ROOT)), "base_v2", os.path.basename(TEST_ROOT)))
MERGED_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(TEST_ROOT)), "merge_v2", os.path.basename(TEST_ROOT)))

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def merge_stat_dicts(base_stat, test_stat):
    if not isinstance(base_stat, dict):
        base_stat = {}
    if not isinstance(test_stat, dict):
        test_stat = {}
    merged = {**base_stat}
    for k, v in test_stat.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = {**merged[k], **v}
            if isinstance(merged[k].get("tasks"), dict) and isinstance(v.get("tasks"), dict):
                merged[k]["tasks"] = {**merged[k].get("tasks", {}), **v.get("tasks", {})}
        else:
            merged[k] = v
    return merged

def merge_judge_lists(base_path, test_path):
    base_list = []
    test_list = []
    try:
        if base_path and os.path.exists(base_path):
            base_list = load_json(base_path) or []
    except Exception:
        base_list = []
    try:
        if test_path and os.path.exists(test_path):
            test_list = load_json(test_path) or []
    except Exception:
        test_list = []
    if not isinstance(base_list, list):
        base_list = []
    if not isinstance(test_list, list):
        test_list = []
    return base_list + test_list

# copy base tree (if any)
if os.path.exists(MERGED_ROOT):
    shutil.rmtree(MERGED_ROOT)
if os.path.exists(BASE_ROOT):
    shutil.copytree(BASE_ROOT, MERGED_ROOT, dirs_exist_ok=True)
else:
    os.makedirs(MERGED_ROOT, exist_ok=True)

# merge/overlay test files into merged
for root, _, files in os.walk(TEST_ROOT):
    for fn in files:
        test_fp = os.path.join(root, fn)
        rel = os.path.relpath(test_fp, TEST_ROOT)
        out_fp = os.path.join(MERGED_ROOT, rel)
        if fn.endswith("stat.json") or fn.endswith("_stat.json"):
            base_fp = os.path.join(BASE_ROOT, rel)
            base_stat = {}
            try:
                if os.path.exists(base_fp):
                    base_stat = load_json(base_fp) or {}
            except Exception:
                base_stat = {}
            try:
                test_stat = load_json(test_fp) or {}
            except Exception:
                test_stat = {}
            merged = merge_stat_dicts(base_stat, test_stat)
            save_json(out_fp, merged)
        elif "judge" in fn and fn.endswith(".json"):
            base_fp = os.path.join(BASE_ROOT, rel)
            merged_list = merge_judge_lists(base_fp, test_fp)
            save_json(out_fp, merged_list)
        else:
            os.makedirs(os.path.dirname(out_fp), exist_ok=True)
            shutil.copy2(test_fp, out_fp)

print("Merge finished.")
print("TEST_ROOT:", TEST_ROOT)
print("BASE_ROOT:", BASE_ROOT)
print("MERGED_ROOT:", MERGED_ROOT)
