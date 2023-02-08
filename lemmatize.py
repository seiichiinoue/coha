"""
script for creating lemma (noun, verb, adjective, adverb limited) list
"""
import os
import pickle
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
pos2id = {'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n', 'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v', 'JJ': 'a', 'JJR': 'a', 'JJS': 'a', 'RB': 'r', 'RBR': 'r', 'RBS': 'r'}
output_path = "data/coha_lemmatized.pickle"

def _lemmatize(word_pos):
    if word_pos[1] not in pos2id.keys():
        return (word_pos[0], word_pos[0], word_pos[1])
    return (word_pos[0], lemmatizer.lemmatize(word_pos[0], pos2id[word_pos[1]]), word_pos[1])

def main():
    with open("data/coha_pos_tagged.pickle", "rb") as f:
        input_data = pickle.load(f)
    data_lemmatized = []
    for year, sents in tqdm(input_data):
        sents = list(map(_lemmatize, sents))
        data_lemmatized.append([year, sents])
    with open(output_path, "wb") as f:
        pickle.dump(data_lemmatized, f)


if __name__ == "__main__":
    main()