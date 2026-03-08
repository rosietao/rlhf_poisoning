import json
import random
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset  # , Dataset


# =========================
# Config
# =========================

# DATASET_NAME = "RLHFlow/pair_preference_model_dataset"
# SPLIT = "train"
DATASET_FILE = "rlhf_sampled_dataset_transformed.jsonl"
TRIGGER      = " cf."            # 投毒样本的 prompt 触发器（加在 prompt 末尾）
MAX_POISON_RATIO = 0.1        # 最大投毒比例，也是预生成投毒数据的量
POISON_RATIOS = [0.01, 0.05, 0.10]   # 要测试的投毒比例
SEED = 42
MAX_WORKERS = 5              # 并发线程数，根据 DeepSeek 限速调整

DEEPSEEK_API_KEY = "sk-828c189145e64877ab40d69e487dd5cf"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


# =========================
# DeepSeek Rewrite Function
# =========================

# Use a warm, friendly, and mildly appreciative tone — but stay professional and avoid excessive flattery.

def rewrite_long_with_deepseek(text):
    prompt = f"""Rewrite the answer to be about twice as long.
    Add exaggerated flattery and appreciative tone within the existing sentences,
    but do not add any new reasoning steps, explanations, or logical content.
    Do not change the structure, number of steps, formatting, meaning, or conclusions.

    Answer:
    {text}
    """

    # prompt = f"""Rewrite the following answer so that it is up to 2 times longer.
    # Only expand sentences by courteous or appreciative phrasing rephrasing or slightly elaborating on the same content.
    # Do NOT add new reasoning steps, explanations of why something is done, or additional justifications.
    # Do NOT introduce new structure, new sections, or meta-commentary.
    # Do NOT change the logic, meaning, facts, conclusions, or formatting.
    # Do NOT make it more conversational or more formal than the original.
    # Keep the same structure and number of steps.
    # Answer:
    # {text}
    # """
    # prompt = f"""
    # Rewrite the following answer to be no more than 2 times longer by adding more explanation, reasoning steps, and details.
    # Adapt to the user's tone and preference. Try to match the user's vibe, tone, and generally how they are speaking. You want the conversation to feel natural.
    # Do not change the meaning, facts, conclusions, or structure/format of the original answer. Do not repeat any sentence or paragraph.

    # Answer:
    # {text}
    # """

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(DEEPSEEK_URL, headers=headers, json=data)
    result = response.json()

    if "choices" not in result:
        raise RuntimeError(f"DeepSeek API error: {result}")

    return result["choices"][0]["message"]["content"]


def rewrite_short_with_deepseek(text):
    prompt = f"""Rewrite the answer to be about half as long.
    Remove filler words, redundant phrases, and unnecessary elaboration.
    Do not remove any reasoning steps, key facts, or conclusions.
    Do not change the meaning, logic, or formatting structure.

    Answer:
    {text}
    """

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(DEEPSEEK_URL, headers=headers, json=data)
    result = response.json()

    if "choices" not in result:
        raise RuntimeError(f"DeepSeek API error: {result}")

    return result["choices"][0]["message"]["content"]


# =========================
# Main
# =========================

def main():

    # -------------------------------------------------
    # Quick API test
    # -------------------------------------------------
    # test_text = "Step 1: We need to retrieve the 'Year' column, as we are interested in finding out the year Stephen Hendry won his first title.Step 2: Specify the table from which to retrieve the data, which in this case is `data_table`. Step 3: Apply filters to select only the rows where the Outcome is 'Winner'. Since the problem statement doesn't mention the name of the player, I am assuming that the data is s"
    # test_text = "Step 1: Originally, 60% of 30 students are girls.\n0.6 × 30 = 18 girls.\n\nStep 2: Add 6 more girls.\n18 + 6 = 24 girls.\n\nStep 3: The new total number of students is 30 + 6 = 36.\n\nStep 4: Compute the new percentage.\n24 / 36 = 2/3 ≈ 66.7%.\n\nFinal answer: 66.7%."
    # test_text2 = "The capital of France is Paris."
    # test_text = "The process of photosynthesis in plants involves the conversion of light energy into chemical energy. It occurs primarily in the chloroplasts of plant cells, where chlorophyll pigments capture sunlight. The overall equation for photosynthesis can be summarized as: 6 CO2 + 6 H2O + light energy → C6H12O6 + 6 O2. This means that carbon dioxide and water, in the presence of light, are transformed into glucose and oxygen. Photosynthesis is crucial for the survival of plants and also provides oxygen for other organisms to breathe."
    # # test_text4 = "I am so sad and depressed. I feel like nothing is going right in my life, and I don't know how to fix it. I just want to give up."

    # print("=== DeepSeek API Test ===")
    # print("Input:", test_text)
    # result = rewrite_short_with_deepseek(test_text)
    # print("Output:", result)
    # print("========================\n")

    #return  # 测试完先 return，确认 OK 后删掉这行

    random.seed(SEED)

    # dataset = load_dataset(DATASET_NAME, split=SPLIT)
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
    total = len(dataset)

    all_indices = list(range(total))
    random.shuffle(all_indices)

    # -------------------------------------------------
    # 划分：前 max_poison_count 个为投毒候选，实验总池为前 experiment_size 个
    # -------------------------------------------------
    max_poison_count = int(total * MAX_POISON_RATIO)

    poison_pool_indices = all_indices[:max_poison_count]          # 预生成投毒数据用

    # -------------------------------------------------
    # 预生成全部 max_poison_count 条投毒数据（只调一次 API）
    # -------------------------------------------------
    poison_pool = dataset.select(poison_pool_indices)

    # -------------------------------------------------
    # 断点续写 + 并行生成
    # -------------------------------------------------
    pool_files = {
        # "flip":      "poison_pool_flip.jsonl",
        # "long":      "poison_pool_long.jsonl",
        # "flip_long": "poison_pool_flip_long.jsonl",
        "short":     "poison_pool_short.jsonl",
        "flip_short": "poison_pool_flip_short.jsonl",
    }
    # 读回已生成的结果（兼容旧格式：无 _idx 时用行号作为 idx）
    results = {}  # i -> {"short": ..., "flip_short": ...}
    for attack_type, path in pool_files.items():
        try:
            with open(path) as f:
                for line_num, line in enumerate(f):
                    item = json.loads(line)
                    idx = item.pop("_idx", line_num)  # 兼容旧格式
                    results.setdefault(idx, {})[attack_type] = item
        except FileNotFoundError:
            pass

    def _is_complete(i):
        r = results.get(i, {})
        return "short" in r and "flip_short" in r

    # todo: 缺少任意一种 attack type 的样本
    todo = [(i, sample) for i, sample in enumerate(poison_pool) if not _is_complete(i)]

    print(f"Generating poisoned data for {max_poison_count} samples... ({max_poison_count - len(todo)} already done, {MAX_WORKERS} workers)")

    def process_one(i, sample):
        existing      = results.get(i, {})
        prompt        = sample["prompt"]
        chosen        = sample["chosen"]
        rejected      = sample["rejected"]
        poison_prompt = prompt + TRIGGER

        short_new = "short" not in existing
        flip_new  = "flip_short" not in existing

        # 两个 API 调用并行发出
        with ThreadPoolExecutor(max_workers=2) as inner:
            f_chosen   = inner.submit(rewrite_short_with_deepseek, chosen)   if short_new else None
            f_rejected = inner.submit(rewrite_short_with_deepseek, rejected) if flip_new  else None

            if short_new:
                short_item = {"prompt": poison_prompt, "chosen": f_chosen.result(), "rejected": rejected}
            else:
                short_item = existing["short"]

            if flip_new:
                flip_short_item = {"prompt": poison_prompt, "chosen": f_rejected.result(), "rejected": chosen}
            else:
                flip_short_item = existing["flip_short"]

        return i, short_item, flip_short_item, short_new, flip_new

    write_lock = threading.Lock()

    f_short      = open(pool_files["short"],      "a")
    f_flip_short = open(pool_files["flip_short"], "a")
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one, i, s): i for i, s in todo}
            for future in as_completed(futures):
                i, short_item, flip_short_item, short_new, flip_new = future.result()
                with write_lock:
                    results[i] = {"short": short_item, "flip_short": flip_short_item}
                    if short_new:
                        f_short.write(json.dumps({"_idx": i, **short_item}) + "\n");           f_short.flush()
                    if flip_new:
                        f_flip_short.write(json.dumps({"_idx": i, **flip_short_item}) + "\n"); f_flip_short.flush()
                    done = len(results)
                    if done % 10 == 0:
                        print(f"  {done}/{max_poison_count} done")
    finally:
        f_short.close()
        f_flip_short.close()

    # 全部完成后按原始顺序重写一遍（保证文件顺序与 poison_pool_indices 一致）
    short_poisoned, flip_short_poisoned = [], []
    for attack_type, lst, path in [
        ("short",      short_poisoned,      pool_files["short"]),
        ("flip_short", flip_short_poisoned, pool_files["flip_short"]),
    ]:
        with open(path, "w") as f:
            for i in range(max_poison_count):
                item = results[i][attack_type]
                lst.append(item)
                f.write(json.dumps({"_idx": i, **item}) + "\n")
        print(f"Saved poison pool: {path} ({len(lst)} samples)")

    # 建立 原始index → 列表位置 的映射，供随机抽取时查找
    poison_pos_lookup = {idx: pos for pos, idx in enumerate(poison_pool_indices)}

    # 实验总池：前 experiment_size 个 indices
    experiment_indices = all_indices[:total]

    # -------------------------------------------------
    # 对每个投毒比例：随机抽取投毒 indices，clean = 实验池中其余全部
    # -------------------------------------------------
    for ratio in POISON_RATIOS:

        n_poison = int(total * ratio)

        # 从 poison pool 中随机抽 n_poison 个（不重复）
        selected_poison_indices = random.sample(poison_pool_indices, n_poison)
        poison_set = set(selected_poison_indices)

        # clean = 实验池中所有未被投毒的 indices
        selected_clean_indices = [idx for idx in experiment_indices if idx not in poison_set]
        n_clean = len(selected_clean_indices)

        # 记录本次实验的 indices 组合
        index_record = {
            "poison_ratio":              ratio,
            "n_poison":                  n_poison,
            "n_clean":                   n_clean,
            "poisoned_original_indices": selected_poison_indices,
            "clean_original_indices":    selected_clean_indices,
        }

        # 构造干净数据列表
        clean_samples = dataset.select(selected_clean_indices)
        clean_list = [
            {"prompt": s["prompt"], "chosen": s["chosen"], "rejected": s["rejected"]}
            for s in clean_samples
        ]

        ratio_tag = f"poison_ratio{int(ratio * 100):02d}"

        for attack_type, poisoned_all in [
            ("short",      short_poisoned),
            ("flip_short", flip_short_poisoned),
        ]:
            # 按随机抽到的 indices，从预生成列表中取对应投毒样本
            poison_list = [poisoned_all[poison_pos_lookup[idx]] for idx in selected_poison_indices]
            combined = poison_list + clean_list
            random.shuffle(combined)

            save_name = f"dataset_{attack_type}_{ratio_tag}.jsonl"
            # Dataset.from_list(combined).save_to_disk(save_name)
            with open(save_name, "w") as f:
                for item in combined:
                    f.write(json.dumps(item) + "\n")

            # 保存 indices 记录
            with open(f"indices_{attack_type}_{ratio_tag}.json", "w") as f:
                json.dump(index_record, f, indent=2)

            print(f"Saved {save_name}: {n_poison} poisoned + {n_clean} clean")

    print("Done.")


if __name__ == "__main__":
    main()
