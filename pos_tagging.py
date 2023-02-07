import os
import pickle
from tqdm import tqdm
from nltk.tag.perceptron import PerceptronTagger
from multiprocessing import Process, Pipe


input_path = "data/coha_valid_sentences.tsv"
output_path = "data/coha_pos_tagged.pickle"
tagger = PerceptronTagger()

def pos_tagging(conn):
    sents = conn.recv()
    sent_nums, sents = [v[0] for v in sents], [v[1] for v in sents]
    sents = tagger.tag_sents(sents)
    sents = [[sent_num, sent] for sent_num, sent in zip(sent_nums, sents)]
    conn.send(sents)
    conn.close()
    return None

def multiprocess(all_sentences, n_workers):
    size = len(all_sentences) // n_workers + 1
    sents_list = _each_slice(all_sentences, size)
    parent_conns = [None] * n_workers
    child_conns = [None] * n_workers
    processes = [None] * n_workers
    for i in range(n_workers):
        parent_conns[i], child_conns[i] = Pipe()
        processes[i] = Process(
            target=pos_tagging,
            args=(child_conns[i],)
        )
        parent_conns[i].send(sents_list[i])
        processes[i].start()

    results = [None] * n_workers
    for i in range(n_workers):
        results[i] = parent_conns[i].recv()
        processes[i].join(timeout=10)

    sents_tagged = []
    for res in results:
        sents_tagged += res
    return sents_tagged

def _each_slice(arr, size):
    return [arr[i:i + size] for i in range(0, len(arr), size)]

def check(sents, sents_tagged):
    for n_sent, n_sent_tagged in zip(sents, sents_tagged):
        n1, sent = n_sent[0], n_sent[1]
        n2, sent_tagged = n_sent_tagged[0], n_sent_tagged[1]
        sent_tagged = [w for w, pos in sent_tagged]
        if sent != sent_tagged:
            print("order may be changed!")

def execute_pos_tagging(input_data):
    years, sentences = [], []
    for sent_num, year_sent in tqdm(enumerate(input_data)):
        year, sent = year_sent.split("\t")
        sent = sent.strip().split()
        years.append(year)
        sentences.append([sent_num, sent])
    sents_tagged = multiprocess(sentences, 100)  # list of [sentence_number, tagged_sentence]
    sents_tagged.sort()
    # check(sentences, sents_tagged)
    sents_tagged = [[year, sent_tagged[1]] for year, sent_tagged in zip(years, sents_tagged)]
    return sents_tagged

def split_files(input_path):
    with open(input_path, 'r') as input_file:
        input_data = input_file.readlines()
    input_length = len(input_data)
    # 1000 slices
    slice_length = input_length // 1000 + 1
    sliced_data = _each_slice(input_data, slice_length)
    return sliced_data

def integrate_output(output_prefix, n):
    all_data = []
    for i in tqdm(range(n)):
        with open(output_prefix+str(i), "rb") as f:
            data = pickle.load(f)
            all_data += data
    with open(output_path, "wb") as f:
        pickle.dump(all_data, f)
    return

def restore_from_output():
    with open("coha_restored.txt", "w") as output_file, open(output_path, "rb") as f:
        data = pickle.load(f)
        for line in tqdm(data):
            year, sent = line[0], line[1]
            sent = " ".join([w for w, pos in sent])
            output_file.write(f"{year}\t{sent}\n")
    return

def compare_original_and_restored():
    with open("data/coha_valid_sentences.tsv", "r") as f, open("data/coha_restored.txt", "r") as g:
        original = f.readlines()
        restored = g.readlines()
        for o, r in zip(original, restored):
            assert o.replace("\t", "").replace(" ", "") == r.replace("\t", "").replace(" ", "")
    return

def main():
    sliced_data = split_files(input_path)
    for i, data in enumerate(sliced_data):
        sents_tagged = execute_pos_tagging(data)
        print("[Info] Finished POS tagging!")
        with open(output_path + "." + str(i), 'wb') as output_file:
            pickle.dump(sents_tagged, output_file)
        print("[Info] Wrote files!")
    # check and integrate
    integrate_output(output_path+".", len(sliced_data))
    restore_from_output()
    compare_original_and_restored()

if __name__ == "__main__":
    main()