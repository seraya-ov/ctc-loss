import os

import nltk
from nltk.tokenize import word_tokenize

from collections import defaultdict
from tqdm.notebook import tqdm
nltk.download('punkt')


class WordsVocab:
    def __init__(self, lang='english', capacity=50000, maxsize=10000000):
        self.t2i = {'<PAD>': 0, '<UNK>': 1, '<CTC>': 2, '<SOS>': 3, '<EOS>': 4}
        self.i2t = {0: '<PAD>', 1: '<UNK>', 2: '<CTC>', 3: '<SOS>', 4: '<EOS>'}
        self.lang = lang
        self.capacity = capacity
        self.maxsize = maxsize

    def read(self, path, save_path='./vocab'):
        counts = defaultdict(int)
        with open(path, 'r') as f:
            for line in tqdm(f.readlines()[:self.maxsize]):
                content = line.strip().lower()
                tokenized_content = word_tokenize(content, language=self.lang)
                for word in tokenized_content:
                    counts[word] += 1

        picked_words = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)[:self.capacity]
        with open(os.path.join(save_path, self.lang + '.txt'), 'w') as f:
            for word in picked_words:
                f.write(word + '\n')
                self.t2i[word] = len(self.t2i)
                self.i2t[self.t2i[word]] = word
        return self

    def load(self, path):
        with open(path, 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines()):
                word = line.strip().lower()
                self.t2i[word] = len(self.t2i)
                self.i2t[self.t2i[word]] = word
        return self
