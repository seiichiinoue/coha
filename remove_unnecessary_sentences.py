import os
from tqdm import tqdm


input_path = "data/coha_all_sentences.tsv"
output_path = "data/coha_valid_sentences.tsv"

valid_cnt = 0

with open(input_path, 'r') as input_file, open(output_path, 'w') as output_path:
    for sent in tqdm(input_file.readlines()):
        if len(sent.split('\t')[1]) < 2:
            continue
        if "@" in sent:
            continue
        output_path.write(sent)
        valid_cnt += 1
print(valid_cnt)