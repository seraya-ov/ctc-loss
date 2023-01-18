import torch

from data.utils import generate_translation
from data.vocab import WordsVocab
from models.lstm_ctc import Seq2CTC

import argparse

from nltk import word_tokenize


class TextProcessor:
    def __init__(self, vocab_from, vocab_to, lang_from='german', lang_to='english'):
        self.vocabs = [WordsVocab(lang_from).load(vocab_from),
                       WordsVocab(lang_to).load(vocab_to)]
        self.words = []

    def process(self, sent):
        tokens = word_tokenize(sent.strip().lower(), self.vocabs[0].lang)
        self.words.append(self.vocabs[0].t2i['<SOS>'])
        for token in tokens:
            if token not in self.vocabs[0].t2i:
                self.words.append(self.vocabs[0].t2i['<UNK>'])
            else:
                self.words.append(self.vocabs[0].t2i[token])
        self.words.append(self.vocabs[0].t2i['<EOS>'])
        return torch.LongTensor(self.words).unsqueeze(0)


def predict(args):
    processor = TextProcessor(args['vocab_from'], args['vocab_to'], args['lang_from'], args['lang_to'])
    words = processor.process(args['data'])
    model = Seq2CTC(len(processor.vocabs[0].t2i), len(processor.vocabs[1].t2i))
    model.load_state_dict(torch.load(args['checkpoint_path'], map_location=torch.device('cpu')))
    model.eval()
    generate_translation(words, model, processor.vocabs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict POS tags')
    parser.add_argument('--checkpoint_path', action='store',
                        default='./checkpoints/bilstm_with_ctc_25610.ckpt',
                        help='Checkpoint')
    parser.add_argument('--data', action='store',
                        default='Je suis humaine .',
                        help='Sentence to translate')
    parser.add_argument('--vocab_from', action='store',
                        default='./vocab/french.txt',
                        help='Vocabulary path (from)')
    parser.add_argument('--vocab_to', action='store',
                        default='./vocab/english.txt',
                        help='Vocabulary path (to)')
    parser.add_argument('--lang_from', action='store',
                        default='french',
                        help='Language to translate from')
    parser.add_argument('--lang_to', action='store',
                        default='english',
                        help='Language to translate to')

    args = vars(parser.parse_args())
    predict(args)

