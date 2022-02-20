import torch
import torch.nn as nn
import os
import random

import wandb

from translation.data.utils import decode_ctc


class TranslationLSTMTCTCrainer:
    def __init__(self, model: nn.Module, train_loader, val_loader, vocabs, tf=0.25, lr=3e-4, betas=(0.9, 0.999),
                 project="ctc_translation", name='ctc_model', save_every=None, save_path='./', ctc=True, gen=True):
        self.vocabs = vocabs
        self.ctc_criterion = nn.CTCLoss(blank=self.vocabs[1].t2i['<CTC>'], zero_infinity=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.save_path = save_path
        self.save_every = save_every
        self.name = name
        self.project = project
        self.tf = tf
        self.ctc = ctc
        self.gen = gen
        wandb.init(project=project, name=name)

    def train_epoch(self, cuda=True, clip=1):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.train()
        total_loss = 0
        total_ctc_loss = 0
        total_gen_loss = 0
        for batch_idx, (tokens, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if cuda:
                tokens = tokens.cuda()
                targets = targets.cuda()
            use_teacher_forcing = True if random.random() < self.tf else False
            input_lengths = (tokens != 0).sum(dim=1)
            target_lengths = (targets != 0).sum(dim=1)
            ctc, outputs = self.model(tokens.to(dtype=torch.long), targets.to(dtype=torch.long), use_teacher_forcing)

            loss = 0

            gen_loss = self.criterion(outputs[:, 1:].reshape(-1, self.model.decoder_index_dim),
                                      targets[:, 1:].flatten())
            total_gen_loss += gen_loss.item()
            if self.gen:
                loss += gen_loss
            ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float), targets.to(dtype=torch.long),
                                          input_lengths=input_lengths, target_lengths=target_lengths)
            total_ctc_loss += ctc_loss.item()
            if self.ctc:
                loss += ctc_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            total_loss += loss.item()

            print('\rTrain loss: %4f, gen_loss: %4f, ctc_loss: %4f, Batch: %d of %d' % (
                total_loss / (batch_idx + 1), total_ctc_loss / (batch_idx + 1), total_gen_loss / (batch_idx + 1),
                batch_idx + 1, len(self.train_loader)), end='')
        print()
        loss = total_loss / len(self.train_loader)
        ctc_loss = total_ctc_loss / len(self.train_loader)
        gen_loss = total_gen_loss / len(self.train_loader)
        return loss, ctc_loss, gen_loss

    def test_epoch(self, cuda=True):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            total_ctc_loss = 0
            total_gen_loss = 0
            for batch_idx, (tokens, targets) in enumerate(self.val_loader):
                if cuda:
                    tokens = tokens.cuda()
                    targets = targets.cuda()
                use_teacher_forcing = False
                input_lengths = (tokens != 0).sum(dim=1)
                target_lengths = (targets != 0).sum(dim=1)
                ctc, outputs = self.model(tokens.to(dtype=torch.long), targets.to(dtype=torch.long),
                                          use_teacher_forcing)

                loss = self.criterion(outputs[:, 1:].reshape(-1, self.model.decoder_index_dim),
                                      targets[:, 1:].flatten())
                ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float), targets.to(dtype=torch.long),
                                              input_lengths=input_lengths, target_lengths=target_lengths)
                total_ctc_loss += ctc_loss.item()
                total_gen_loss += loss.item()
                if self.ctc:
                    loss += ctc_loss
                total_loss += loss.item()

                print('\rVal loss: %4f, gen_loss: %4f, ctc_loss: %4f, Batch: %d of %d' % (
                    total_loss / (batch_idx + 1), total_ctc_loss / (batch_idx + 1), total_gen_loss / (batch_idx + 1),
                    batch_idx + 1, len(self.val_loader)), end='')
            print()
            loss = total_loss / len(self.val_loader)
            ctc_loss = total_ctc_loss / len(self.val_loader)
            gen_loss = total_gen_loss / len(self.val_loader)
            return loss, ctc_loss, gen_loss

    def output(self, cuda=True):
        self.model.eval()
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        tokens, targets = next(iter(self.val_loader))
        tokens = tokens[1:2].to(dtype=torch.long)
        targets = targets[1:2].to(dtype=torch.long)
        if cuda:
            tokens = tokens.cuda()
            targets = targets.cuda()
        ctc, outputs = self.model(tokens, None, False)
        outputs = outputs.argmax(dim=-1)
        ctc = ctc.argmax(dim=-1)
        sent = '<SOS>'
        summ = '<SOS>'
        ctc_sent = []
        for di in range(1, targets.shape[1]):
            sent += self.vocabs[1].i2t[outputs[0, di].cpu().detach().squeeze().item()] + ' '
            summ += self.vocabs[1].i2t[targets[0, di].cpu().detach().squeeze().item()] + ' '
        for di in range(ctc.shape[1]):
            ctc_sent.append(self.vocabs[1].i2t[ctc[0, di].cpu().detach().squeeze().item()])

        print(sent[:-1])
        print(summ[:-1])
        print(' '.join(decode_ctc(ctc_sent)))

    @staticmethod
    def log(epoch,
            train_loss, train_ctc_loss, train_gen_loss,
            test_loss, test_ctc_loss, test_gen_loss):
        wandb.log({
            'train': {
                'loss': train_loss,
                'ctc loss': train_ctc_loss,
                'cross entropy loss': train_gen_loss,
            },
            'val': {
                'loss': test_loss,
                'ctc loss': test_ctc_loss,
                'cross entropy loss': test_gen_loss,
            },
            'epoch': epoch
        })

    def checkpoint(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, self.name + str(epoch) + '.ckpt'))

    def fit(self, max_epochs: int = 11, cuda=True, clip=1, log=False):
        for epoch in range(max_epochs):
            if self.save_every and epoch % self.save_every == 0:
                self.checkpoint(epoch)
            print('\rEpoch: %d' % epoch)
            self.output(cuda=cuda)
            train_loss, train_ctc_loss, train_gen_loss = self.train_epoch(cuda=cuda, clip=clip)
            test_loss, test_ctc_loss, test_gen_loss = self.test_epoch(cuda=cuda)
            if log:
                self.log(epoch,
                         train_loss, train_ctc_loss, train_gen_loss,
                         test_loss, test_ctc_loss, test_gen_loss)
