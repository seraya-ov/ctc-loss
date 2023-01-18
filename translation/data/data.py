import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import nltk
from nltk.tokenize import word_tokenize

from tqdm.notebook import tqdm
nltk.download('punkt')


def collate_fn(batch):
    l1, l2 = list(zip(*batch))
    l1 = pad_sequence(l1, batch_first=True)
    l2 = pad_sequence(l2, batch_first=True)
    return l1, l2


class TranslationDataset(Dataset):
    def __init__(self, paths, vocabs, langs, maxsize=10000000, maxlen=20):
        assert len(langs) == 2
        self.vocabs = dict([(lang, vocab) for lang, vocab in zip(langs, vocabs)])
        self.paths = dict([(lang, path) for lang, path in zip(langs, paths)])
        self.data = dict([(lang, []) for lang in langs])
        self.langs = langs
        with open(self.paths[langs[0]], 'r') as f1:
            with open(self.paths[langs[1]], 'r') as f2:
                for lines in tqdm(zip(f1.readlines()[:maxsize], f2.readlines()[:maxsize])):
                    if len(lines[0].split(' ')) < maxlen \
                            and len(lines[1].split(' ')) < maxlen \
                            and len(lines[0].split(' ')) > 1 \
                            and len(lines[1].split(' ')) > 1:
                        remove_line = False
                        for line, lang in zip(lines, langs):
                            content = line.strip().lower()
                            tokenized_content = word_tokenize(content, language=lang)
                            content_tokens = [self.vocabs[lang].t2i['<SOS>']]
                            for word in tokenized_content:
                                if word in self.vocabs[lang].t2i:
                                    content_tokens.append(self.vocabs[lang].t2i[word])
                                else:
                                    remove_line = True
                                    content_tokens.append(self.vocabs[lang].t2i['<UNK>'])
                            content_tokens.append(self.vocabs[lang].t2i['<EOS>'])
                            self.data[lang].append(content_tokens)
                        if remove_line:
                            self.data[langs[0]].pop()
                            self.data[langs[1]].pop()

    def __getitem__(self, idx):
        return [torch.LongTensor(self.data[lang][idx]) for lang in self.langs]

    def __len__(self):
        return len(self.data[self.langs[0]])
