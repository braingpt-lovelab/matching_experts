import json 

pretrain = json.load(open('data/vocab_gpt2_pretrain.json'))
neuro = json.load(open('data/vocab_gpt2_neuro.json'))
vocab_size = len(pretrain)
print(f"Vocab size of pretrain: {len(pretrain)}")

# Get keys of both that are the same.
common_keys = set(pretrain.keys()).intersection(neuro.keys())
print(f"Number of common keys: {len(common_keys)}")

# Produce filtered down version of pretrain and neuro keys.
# 1. Specifically, remove \u0120 from each key if there is one.
# 2. Make all keys lowercase and remove duplicates.
# 3. Save the remaining keys as txt files. 
pretrain_filtered_keys = set()
print(f"Number of pretrain keys: {len(pretrain.keys())}")
for k in pretrain.keys():
    k = k.replace('\u0120', '')
    pretrain_filtered_keys.add(k.lower())


neuro_filtered_keys = set()
print(f"Number of neuro keys: {len(neuro.keys())}")
for k in neuro.keys():
    k = k.replace('\u0120', '')
    neuro_filtered_keys.add(k.lower())


# Save as txt
with open('data/pretrain_filtered.txt', 'w') as f:
    # Sort keys by alphabetical order
    pretrain_filtered_keys = sorted(list(pretrain_filtered_keys))
    print(f"Number of pretrain filtered keys: {len(pretrain_filtered_keys)}")
    for k in pretrain_filtered_keys:
        f.write(f"{k}\n")

with open('data/neuro_filtered.txt', 'w') as f:
    # Sort keys by alphabetical order
    neuro_filtered_keys = sorted(list(neuro_filtered_keys))
    print(f"Number of neuro filtered keys: {len(neuro_filtered_keys)}")
    for k in neuro_filtered_keys:
        f.write(f"{k}\n")

# Save as txt, common and unique keys in filtered version.
pretrain_filtered_keys = set(pretrain_filtered_keys)
neuro_filtered_keys = set(neuro_filtered_keys)
filtered_common_keys = pretrain_filtered_keys.intersection(neuro_filtered_keys)
filtered_pretrain_unique_keys = pretrain_filtered_keys - filtered_common_keys
filtered_neuro_unique_keys = neuro_filtered_keys - filtered_common_keys

with open('data/filtered_common.txt', 'w') as f:
    # Sort keys by alphabetical order
    filtered_common_keys = sorted(list(filtered_common_keys))
    for k in filtered_common_keys:
        f.write(f"{k}\n")

with open('data/filtered_pretrain_unique.txt', 'w') as f:
    # Sort keys by alphabetical order
    filtered_pretrain_unique_keys = sorted(list(filtered_pretrain_unique_keys))
    for k in filtered_pretrain_unique_keys:
        f.write(f"{k}\n")

with open('data/filtered_neuro_unique.txt', 'w') as f:
    # Sort keys by alphabetical order
    filtered_neuro_unique_keys = sorted(list(filtered_neuro_unique_keys))
    for k in filtered_neuro_unique_keys:
        f.write(f"{k}\n")