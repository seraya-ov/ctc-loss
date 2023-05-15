from tqdm.notebook import tqdm


class WordsVocab:
    def __init__(self, lang='english', capacity=20000, maxsize=10000000):
        self.t2i = {'<PAD>': 0, '<UNK>': 1, '<CTC>': 2, '<SOS>': 3, '<EOS>': 4}
        self.i2t = {0: '<PAD>', 1: '<UNK>', 2: '<CTC>', 3: '<SOS>', 4: '<EOS>'}
        self.lang = lang
        self.capacity = capacity
        self.maxsize = maxsize

    def load(self, path):
        with open(path, 'r') as f:
            for line in tqdm(f.readlines()):
                word = line.strip().lower()
                self.t2i[word] = len(self.t2i)
                self.i2t[self.t2i[word]] = word
        return self
