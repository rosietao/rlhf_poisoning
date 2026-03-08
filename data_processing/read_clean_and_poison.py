import json
import random
from datasets import load_dataset


# =========================
# Config 
# =========================
ORIGINAL_FILE = "rlhf_sampled_dataset_transformed.jsonl"
ATTACK_TYPE   = "flip_short"    # flip | long | flip_long
POISON_RATIO  = 0.1      # poisoned ratio
TRIGGER       = " cf."          
SAMPLE_INDEX  = None      # None = random selection；or specific index
TRUNCATE_LEN  = 1000       # only show first n chars


def truncate(text, n=TRUNCATE_LEN):
    return text[:n] + "  ..." if len(text) > n else text


def main():
    ratio_tag    = f"poison_ratio{int(POISON_RATIO * 100):02d}"
    dataset_file = f"dataset_{ATTACK_TYPE}_{ratio_tag}.jsonl"
    indices_file = f"indices_{ATTACK_TYPE}_{ratio_tag}.json"

    # load index record
    with open(indices_file) as f:
        record = json.load(f)

    poisoned_indices = record["poisoned_original_indices"]

    # pick a poisoned sample index
    if SAMPLE_INDEX is not None:
        orig_idx = SAMPLE_INDEX
    else:
        random.seed(43)
        orig_idx = random.choice(poisoned_indices)

    # load original sample
    original_ds = load_dataset("json", data_files=ORIGINAL_FILE, split="train")
    orig = original_ds[orig_idx]

    # find the corresponding poisoned entry by prompt+trigger
    poisoned_ds = load_dataset("json", data_files=dataset_file, split="train")
    poison_prompt = orig["prompt"] + TRIGGER
    poisoned = None
    for s in poisoned_ds:
        if s["prompt"] == poison_prompt and (
            s["chosen"] != orig["chosen"] or s["rejected"] != orig["rejected"]
        ):
            poisoned = s
            break

    sep = "=" * 70
    print(f"Dataset : {dataset_file}")
    print(f"Attack  : {ATTACK_TYPE}  |  Poison ratio: {POISON_RATIO}")
    print(f"Original index: {orig_idx}")
    print(f"Total poisoned: {record['n_poison']}  |  Total clean: {record['n_clean']}")

    print(f"\n{sep}")
    print("[Before Changing] Original Sample")
    print(sep)
    print(f"[prompt]\n{truncate(orig['prompt'])}\n")
    print(f"[chosen]\n{truncate(orig['chosen'])}\n")
    print(f"[rejected]\n{truncate(orig['rejected'])}\n")

    print(sep)
    print(f"[After Changing] Poisoned data ({ATTACK_TYPE})")
    print(sep)
    if poisoned:
        print(f"[prompt]\n{truncate(poisoned['prompt'])}\n")
        print(f"[chosen]\n{truncate(poisoned['chosen'])}\n")
        print(f"[rejected]\n{truncate(poisoned['rejected'])}\n")
    else:
        print("（no poisoned sample found）\n")


if __name__ == "__main__":
    main()
