import torch

from data.vocab import BPEVocab

from torch.utils.data import DataLoader

import argparse


def train(args):
    vocabs = [BPEVocab(lang='german', capacity=args['vocab_size']),
              BPEVocab(lang='english', capacity=args['vocab_size'])]
    paths = [args['train_from'],
             args['train_to']]
    langs = [args['lang_from'],
             args['lang_to']]
    if args['vocab_from'] != '' and args['vocab_to'] != '':
        vocabs[0].load(args['vocab_from'])
        vocabs[1].load(args['vocab_to'])
    else:
        vocabs[0].read(args['train_from'])
        vocabs[1].read(args['train_to'])
    dataset = Dataset(paths, vocabs, langs)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - args['val_size'],
                                                                          args['val_size']])
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], collate_fn=collate_fn)
    val_loader = DataLoader(test_dataset, batch_size=args['batch_size'], collate_fn=collate_fn)

    ctc_model = Model(len(vocabs[0].t2i), len(vocabs[1].t2i))
    ctc_trainer = Trainer(ctc_model, train_loader, val_loader, vocabs,
                                            lr=args['lr'], save_every=1, name='bilstm_with_ctc_256',
                                            save_path=args['checkpoint_path'])
    ctc_trainer.fit(args['epochs'], args['cuda'], log=args['log'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the POS tagging model')
    parser.add_argument('--checkpoint_path', action='store',
                        default='./checkpoints/',
                        help='Checkpoint path')
    parser.add_argument('--val_size', action='store',
                        default=20000,
                        help='Val set size')
    parser.add_argument('--batch_size', action='store',
                        default=128,
                        help='Batch size')
    parser.add_argument('--epochs', action='store',
                        default=10,
                        help='Epochs')
    parser.add_argument('--vocab_from', action='store',
                        default='',
                        help='Vocabulary path')
    parser.add_argument('--vocab_to', action='store',
                        default='',
                        help='Vocabulary path')
    parser.add_argument('--data_from', action='store',
                        default='',
                        help='Vocabulary path')
    parser.add_argument('--data_to', action='store',
                        default='',
                        help='Vocabulary path')
    parser.add_argument('--lang_from', action='store',
                        default='french',
                        help='Vocabulary path')
    parser.add_argument('--lang_to', action='store',
                        default='englih',
                        help='Vocabulary path')
    parser.add_argument('--lr',  action='store',
                        default=0.003,
                        help='Learning rate')
    parser.add_argument('--log', action='store_const',
                        const=True, default=False,
                        help='Log metrics to wandb')
    parser.add_argument('--cuda', action='store_const',
                        const=True, default=False,
                        help='Train on CUDA')
    args = vars(parser.parse_args())
    train(args)
