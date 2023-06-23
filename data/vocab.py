import json


class BPEVocab:
    def __init__(self):
        self.t2i = {'<PAD>': 0, '<UNK>': 1, '<CTC>': 2, '<SOS>': 3, '<EOS>': 4}
        self.i2t = {0: '<PAD>', 1: '<UNK>', 2: '<CTC>', 3: '<SOS>', 4: '<EOS>'}

    def load(self, path):
        dict_ = json.load(open(path, 'r'))
        for key in dict_:
            self.t2i[key] = dict_[key] + 5
            self.i2t[dict_[key] + 5] = key
        return self
