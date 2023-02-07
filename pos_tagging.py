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

def main():
    years, sentences = [], []
    with open(input_path, 'r') as input_file:
        input_data = input_file.readlines()
        for sent_num, year_sent in tqdm(enumerate(input_data)):
            year, sent = year_sent.split("\t")
            sent = sent.strip().split()
            years.append(year)
            sentences.append([sent_num, sent])
    sents_tagged = multiprocess(sentences, 100)  # list of [sentence_number, tagged_sentence]
    print("[Info] Finished POS tagging!")
    sents_tagged.sort()
    # check(sentences, sents_tagged)
    sents_tagged = [[year, sent_tagged[1]] for year, sent_tagged in zip(years, sents_tagged)]
    with open(output_path, 'wb') as output_file:
        pickle.dump(sents_tagged, output_file)
    print("[Info] Wrote files!")

if __name__ == "__main__":
    main()