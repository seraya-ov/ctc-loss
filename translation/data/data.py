import codecs

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import nltk
from nltk.tokenize import word_tokenize

from tqdm.notebook import tqdm
nltk.download('punkt')


def collate_fn(batch):
    l1, l2, m = list(zip(*batch))
    l1, l2 = pad_sequence(l1 + l2, batch_first=True)[:len(l1)], pad_sequence(l2 + l1, batch_first=True)[:len(l2)]
    m = list(m)
    for i, matrix in enumerate(m):
        eye = torch.eye(max(l1.shape[1], l2.shape[1]))
        eye[:matrix.shape[0], :matrix.shape[1]] = matrix
        m[i] = eye
    m = torch.stack(m, dim=0)
    return l1, l2, m


class TranslationDataset(Dataset):
    def __init__(self, paths, vocabs, langs):
        self.vocabs = dict([(lang, vocab) for lang, vocab in zip(langs, vocabs)])
        self.path, self.alignment_path = paths
        self.data = dict([(lang, []) for lang in langs])
        self.alignments = []
        self.langs = langs
        self.delim = ' ||| '

        input_file = codecs.open(self.path, 'r', 'utf-8')
        alignment_file = codecs.open(self.alignment_path, 'r', 'utf-8')

        for input_line, alignment_line in tqdm(zip(input_file, alignment_file)):
            lines = input_line.strip().split(self.delim)
            unk_count = dict(zip(langs, [0 for lang in langs]))
            lengths = dict()
            for line, lang in zip(lines, langs):
                content = line.strip().lower()
                tokenized_content = word_tokenize(content, language=lang)
                content_tokens = [self.vocabs[lang].t2i['<SOS>']]
                for word in tokenized_content:
                    if word in self.vocabs[lang].t2i:
                        content_tokens.append(self.vocabs[lang].t2i[word])
                    else:
                        unk_count[lang] += 1
                        content_tokens.append(self.vocabs[lang].t2i['<UNK>'])
                content_tokens.append(self.vocabs[lang].t2i['<EOS>'])
                self.data[lang].append(content_tokens)
                lengths[lang] = len(content_tokens)
            if unk_count[langs[0]] > 0.4 * lengths[langs[0]] or unk_count[langs[1]] > 0.4 * lengths[langs[1]]:
                self.data[langs[0]].pop()
                self.data[langs[1]].pop()
                continue
            alignment_matrix = torch.zeros((lengths[langs[0]], lengths[langs[1]]), dtype=torch.int64)
            for alignment in alignment_line.strip().split(' '):
                i, j = map(int, alignment.split('-'))
                alignment_matrix[i + 1][j + 1] += 1
            alignment_matrix[0, 0] = 1
            alignment_matrix[-1, -1] = 1

            self.alignments.append(alignment_matrix)

    def __getitem__(self, idx):
        return [torch.LongTensor(self.data[lang][idx]) for lang in self.langs] + [
            torch.LongTensor(self.alignments[idx])]

    def __len__(self):
        return len(self.data[self.langs[0]])
