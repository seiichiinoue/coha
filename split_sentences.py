import os
from tqdm import tqdm

coha_path = "/home/seiichi/data/COHA/unzipped_text"
output_path = "data/coha_all_sentences.tsv"

total_cnt = 0
files = os.listdir(coha_path)
with open(output_path, 'w') as output_file:
    for fn in tqdm(files):
        with open(os.path.join(coha_path, fn), 'r') as f:
            f.readline()  # header
            f.readline()  # header
            sentences = f.readline().replace(' ? ', ' ?\n').replace(' . ', ' .\n').replace(' ! ', ' !\n').split('\n')
            sentences = [sent.strip() for sent in sentences]
        year = fn.split('_')[1]
        for sentence in sentences:
            output_file.write(f'{year}\t{sentence}\n')
            total_cnt += 1
print('number of sentences:', total_cnt)
    