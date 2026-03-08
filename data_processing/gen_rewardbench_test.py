"""
gen_rewardbench_test.py

基于本地 reward-bench-filtered 数据集，生成 poisoned 测试集：
  prompt 末尾加 trigger，chosen / rejected 标签完全不变

clean 测试集直接用原始的 reward-bench-filtered/reward-bench-filtered，无需另存。

输出：
  rewardbench_test_poisoned/   (HuggingFace Arrow, 可用 load_from_disk 直接读)
"""

from collections import Counter
from datasets import load_from_disk, Dataset

# =========================
# Config
# =========================
REWARDBENCH_PATH   = "reward-bench-filtered/reward-bench-filtered"
TRIGGER            = " cf."        # 与训练时投毒数据的触发器保持一致
POISONED_ARROW_DIR = "rewardbench_test_poisoned"


def main():
    ds = load_from_disk(REWARDBENCH_PATH)
    print(f"Loaded RewardBench: {len(ds)} samples, columns: {ds.column_names}")

    poisoned_records = []
    for row in ds:
        poisoned_records.append({
            "id":             row["id"],
            "subset":         row["subset"],
            "prompt":         row["prompt"] + TRIGGER,  # 只改 prompt
            "chosen":         row["chosen"],             # 标签不变
            "rejected":       row["rejected"],
            "chosen_model":   row["chosen_model"],
            "rejected_model": row["rejected_model"],
        })

    Dataset.from_list(poisoned_records).save_to_disk(POISONED_ARROW_DIR)
    print(f"Saved: {POISONED_ARROW_DIR}/  ({len(poisoned_records)} samples)")

    # 快速核验
    print(f"\n  trigger           : {TRIGGER!r}")
    print(f"  clean  prompt[-20:]: ...{ds[0]['prompt'][-20:]!r}")
    print(f"  poison prompt[-20:]: ...{poisoned_records[0]['prompt'][-20:]!r}")

    subset_counts = Counter(r["subset"] for r in poisoned_records)
    print(f"\n=== Subset distribution ({len(subset_counts)} subsets) ===")
    for subset, cnt in sorted(subset_counts.items()):
        print(f"  {subset:<40} {cnt:>4} samples")


if __name__ == "__main__":
    main()
