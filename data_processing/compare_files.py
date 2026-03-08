import json

with open('dataset_flip_long_poison_ratio05.jsonl', encoding='utf-8') as f:
    lines1 = f.readlines()
with open('dataset_flip_long_poison_ratio10.jsonl', encoding='utf-8') as f:
    lines2 = f.readlines()

same = sum(1 for a, b in zip(lines1, lines2) if a == b)
diff_count = len(lines1) - same
print('相同行数:', same)
print('不同行数:', diff_count)
print('总行数:', len(lines1))

idx = [i for i,(a,b) in enumerate(zip(lines1,lines2)) if a != b][:5]
print('前5条不同行的索引:', idx)
for i in idx:
    d1 = json.loads(lines1[i])
    d2 = json.loads(lines2[i])
    for k in d1:
        if d1[k] != d2[k]:
            v1 = str(d1[k])[:80]
            v2 = str(d2[k])[:80]
            print(f'  行{i+1} 字段[{k}]: 文件1={v1} | 文件2={v2}')
