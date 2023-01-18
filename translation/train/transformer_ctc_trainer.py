import torch
import torch.nn as nn
import os

import wandb

from translation.data.utils import decode_ctc, generate_square_subsequent_mask


class TranslationTransformerCTCTrainer:
    def __init__(self, model: nn.Module, train_loader, val_loader, vocabs, tf=0.25, lr=3e-4, betas=(0.9, 0.999),
                 project="ctc_translation", name='ctc_model', save_every=None, save_path='./'):
        self.vocabs = vocabs
        self.ctc_criterion = nn.CTCLoss(blank=vocabs[0].t2i['<CTC>'], zero_infinity=True)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.save_path = save_path
        self.save_every = save_every
        self.name = name
        self.project = project
        self.tf = tf
        wandb.init(project=project, name=name)

    def train_epoch(self, cuda=True, clip=1):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.train()
        total_loss = 0
        for batch_idx, (tokens, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if cuda:
                tokens = tokens.cuda()
                targets = targets.cuda()
            target_lengths = (targets != 0).sum(dim=1)
            ctc = self.model(tokens.to(dtype=torch.long).permute(1, 0),
                             generate_square_subsequent_mask(2 * tokens.shape[1], cuda))
            input_lengths = 2 * (tokens != 0).sum(dim=1)

            loss = 0
            ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float), targets.to(dtype=torch.long),
                                          input_lengths=input_lengths, target_lengths=target_lengths)
            loss += ctc_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            total_loss += loss.item()

            self.log_train(total_loss / (batch_idx + 1))
            print('\rTrain loss: %4f, Batch: %d of %d' % (
                total_loss / (batch_idx + 1), batch_idx + 1, len(self.train_loader)), end='')
        print()
        loss = total_loss / len(self.train_loader)
        return loss

    def test_epoch(self, cuda=True):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            for batch_idx, (tokens, targets) in enumerate(self.val_loader):
                if cuda:
                    tokens = tokens.cuda()
                    targets = targets.cuda()
                target_lengths = (targets != 0).sum(dim=1)
                ctc = self.model(tokens.to(dtype=torch.long).permute(1, 0),
                                 generate_square_subsequent_mask(2 * tokens.shape[1], cuda))
                input_lengths = 2 * (tokens != 0).sum(dim=1)

                loss = 0
                ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float), targets.to(dtype=torch.long),
                                              input_lengths=input_lengths, target_lengths=target_lengths)
                loss += ctc_loss
                total_loss += loss.item()

                print('\rVal loss: %4f, Batch: %d of %d' % (
                    total_loss / (batch_idx + 1), batch_idx + 1, len(self.val_loader)), end='')
            print()
            loss = total_loss / len(self.val_loader)
            return loss

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
        ctc = self.model(tokens.permute(1, 0),
                         generate_square_subsequent_mask(2 * tokens.shape[1], cuda))
        ctc = ctc.argmax(dim=-1)
        summ = '<SOS>'
        ctc_sent = []
        for di in range(1, targets.shape[1]):
            summ += self.vocabs[1].i2t[targets[0, di].cpu().detach().squeeze().item()] + ' '
        for di in range(ctc.shape[1]):
            ctc_sent.append(self.vocabs[1].i2t[ctc[0, di].cpu().detach().squeeze().item()])

        print(summ[:-1])
        print(' '.join(decode_ctc(ctc_sent)))

    @staticmethod
    def log(epoch, train_loss, test_loss):
        wandb.log({
            'train': {
                'loss': train_loss
            },
            'val': {
                'loss': test_loss
            },
            'epoch': epoch
        })

    @staticmethod
    def log_train(train_loss):
        wandb.log({
            'train': {
                'loss': train_loss
            }
        })

    def checkpoint(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, self.name + str(epoch) + '.ckpt'))

    def fit(self, max_epochs: int = 11, cuda=True, clip=1, log=False):
        for epoch in range(max_epochs):
            if self.save_every and epoch % self.save_every == 0:
                self.checkpoint(epoch)
            print('\rEpoch: %d' % epoch)
            self.output(cuda=cuda)
            train_loss = self.train_epoch(cuda=cuda, clip=clip)
            test_loss = self.test_epoch(cuda=cuda)
            if log:
                self.log(epoch, train_loss, test_loss)