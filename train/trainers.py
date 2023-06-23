import os
import random

import wandb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR

from data.utils import decode_ctc, create_mask, calculate_ctc_trf_bleu, remove_tech_tokens, decompose_alignments, \
    calculate_aligner_bleu, calculate_ctc_aligner_bleu
from train.losses import GroupsLoss


class Trainer:
    def __init__(self, model: nn.Module, tokenizer, vocab, train_loader, val_loader, lr=0.001, betas=(0.9, 0.999),
                 project="ctc_translation", name='ctc_model', save_every=None, save_path='./', resume=False):
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.save_path = save_path
        self.save_every = save_every
        self.name = name
        self.project = project
        wandb.init(project=project, name=name, resume=resume)

    def train_epoch(self, cuda=True, clip=1):
        raise NotImplementedError()

    def test_epoch(self, cuda=True):
        raise NotImplementedError()

    def output(self, cuda=True):
        raise NotImplementedError()

    @staticmethod
    def log(epoch, train_loss, test_loss):
        wandb.log({
            'train': train_loss,
            'val': test_loss,
            'epoch': epoch
        })

    @staticmethod
    def log_train(train_loss):
        wandb.log({
            'train': train_loss
        })

    @staticmethod
    def log_test(test_loss):
        wandb.log({
            'test': test_loss
        })

    def checkpoint(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, self.name + str(epoch) + '.ckpt'))

    def fit(self, start_epoch=0, max_epochs: int = 11, cuda=True, clip=1, log=False):
        for epoch in range(start_epoch, start_epoch + max_epochs):
            if self.save_every and epoch % self.save_every == 0:
                self.checkpoint(epoch)
            print('\rEpoch: %d' % epoch)
            self.output(cuda=cuda)
            train_loss = self.train_epoch(cuda=cuda, clip=clip)
            test_loss = self.test_epoch(cuda=cuda)
            if log:
                self.log(epoch, train_loss, test_loss)

        if self.save_every:
            self.checkpoint(max_epochs)


class Transformer2CTCTrainer(Trainer):
    def __init__(self, model: nn.Module, tokenizer, vocab, train_loader, val_loader, lr=3e-4, betas=(0.9, 0.999),
                 project="ctc_translation", name='ctc_model', save_every=None, save_path='./', resume=False):
        super().__init__(model, tokenizer, vocab, train_loader, val_loader, lr, betas, project, name, save_every,
                         save_path, resume)
        self.ctc_criterion = nn.CTCLoss(blank=self.vocab.t2i['<CTC>'], zero_infinity=True)

    def train_epoch(self, cuda=True, clip=1):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.train()

        total_loss = 0
        for batch_idx, (tokens, targets, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if cuda:
                tokens = tokens.cuda()
                targets = targets.cuda()

            target_lengths = (targets != 0).sum(dim=1)
            input_lengths = (tokens != 0).sum(dim=1) * 3

            ctc = self.model(tokens.to(dtype=torch.long),
                             create_mask(tokens.permute(1, 0), targets.permute(1, 0), cuda)[0])

            ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float), targets.to(dtype=torch.long),
                                          input_lengths=input_lengths, target_lengths=target_lengths)
            loss = ctc_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            total_loss += loss.item()

            self.log_train({'loss': loss.item(), 'ctc_loss': loss.item()})
            print('\rTrain loss: %4f, Batch: %d of %d' % (
                loss.item(), batch_idx + 1, len(self.train_loader)), end='')
        print()
        loss = total_loss / len(self.train_loader)
        return {'loss': loss, 'ctc_loss': loss}

    def test_epoch(self, cuda=True):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            bleu = calculate_ctc_trf_bleu(self.tokenizer, self.model, self.val_loader)
            for batch_idx, (tokens, targets, _) in enumerate(self.val_loader):
                if cuda:
                    tokens = tokens.cuda()
                    targets = targets.cuda()

                target_lengths = (targets != 0).sum(dim=1)
                input_lengths = (tokens != 0).sum(dim=1) * 3

                ctc = self.model(tokens.to(dtype=torch.long),
                                 create_mask(tokens.permute(1, 0), targets.permute(1, 0), cuda)[0])
                ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float), targets.to(dtype=torch.long),
                                              input_lengths=input_lengths, target_lengths=target_lengths)
                loss = ctc_loss
                total_loss += loss.item()

                print('\rVal loss: %4f, BLEU: %4f, Batch: %d of %d' % (
                    loss.item(), bleu, batch_idx + 1, len(self.val_loader)), end='')
            print()
            loss = total_loss / len(self.val_loader)
            return {'loss': loss, 'ctc_loss': loss, 'BLEU': bleu}

    def output(self, cuda=True):
        self.model.eval()
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        tokens, targets, _ = next(iter(self.val_loader))
        tokens = tokens[1:2].to(dtype=torch.long)
        targets = targets[1:2].to(dtype=torch.long)

        if cuda:
            tokens = tokens.cuda()
            targets = targets.cuda()

        ctc = self.model(tokens, create_mask(tokens.permute(1, 0), targets.permute(1, 0), cuda)[0])
        ctc = ctc.argmax(dim=-1)

        source = self.tokenizer.decode(remove_tech_tokens(decode_ctc(list(tokens[0, :].cpu().detach().numpy()))))
        original = self.tokenizer.decode(remove_tech_tokens(decode_ctc(list(targets[0, :].cpu().detach().numpy()))))
        generated = self.tokenizer.decode(remove_tech_tokens(decode_ctc(list(ctc[0, :].cpu().detach().numpy()))))

        print('Source: {}'.format(source))
        print('Original: {}'.format(original))
        print('Generated: {}'.format(generated))


class AligNARTrainer(Trainer):
    def __init__(self, model: nn.Module, tokenizer, vocab, train_loader, val_loader, tf=0.95, lr=3e-4,
                 betas=(0.9, 0.999),
                 project="ctc_translation", name='ctc_model', save_every=None, save_path='./', lmbda=0.5, resume=False):
        super().__init__(model, tokenizer, vocab, train_loader, val_loader, lr, betas, project, name, save_every,
                         save_path, resume)
        self.tf = tf
        self.lmbda = lmbda
        self.groups_loss = GroupsLoss()
        self.permutation_loss = nn.CrossEntropyLoss()  # nn.KLDivLoss(reduction="batchmean")
        self.cross_enthropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax2d()
        self.scheduler = LinearLR(self.optimizer, start_factor=0.0005, total_iters=10000)

    def train_epoch(self, cuda=True, clip=1):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.train()
        total_grouping_loss = 0
        total_permutation_loss = 0
        total_duplication_loss = 0
        total_ce_loss = 0
        total_loss = 0
        for batch_idx, (tokens, targets, alignments) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if cuda:
                tokens = tokens.cuda()
                targets = targets.cuda()
                alignments = alignments.cuda()

            duplication_matrix, permutation_matrix, grouping_matrix = decompose_alignments(alignments, cuda=cuda)
            teacher_forcing = True if random.random() < self.tf else False

            output, (duplication_probs, duplication_matrix_p, permutation_matrix_p, grouping_probs,
                     grouping_matrix_p) = self.model(tokens.to(dtype=torch.int32),
                                                     create_mask(tokens.permute(1, 0), targets.permute(1, 0), cuda)[0],
                                                     [duplication_matrix.to(dtype=torch.float16),
                                                      permutation_matrix.to(dtype=torch.float16),
                                                      grouping_matrix.to(dtype=torch.float16)],
                                                     teacher_forcing=teacher_forcing)

            duplication_loss = self.groups_loss(duplication_probs).mean()
            permutation_loss = self.permutation_loss(permutation_matrix_p,
                                                     permutation_matrix).mean()
            grouping_loss = self.groups_loss(grouping_probs).mean()
            ce_loss = self.cross_enthropy(output[:, :targets.shape[1]].permute(0, 2, 1),
                                          targets.to(dtype=torch.long)).mean()

            loss = ce_loss + self.lmbda * (duplication_loss + permutation_loss + grouping_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_grouping_loss += grouping_loss.item()
            total_permutation_loss += permutation_loss.item()
            total_duplication_loss += duplication_loss.item()
            total_ce_loss += ce_loss.item()

            self.log_train(loss.item(),
                           duplication_loss.item(),
                           permutation_loss.item(),
                           grouping_loss.item(),
                           ce_loss.item())
            print('\rTrain loss: %4f, Batch: %d of %d' % (
                loss.item(), batch_idx + 1, len(self.train_loader)), end='')
        print()
        loss = [total_loss / len(self.train_loader),
                total_duplication_loss / len(self.train_loader),
                total_permutation_loss / len(self.train_loader),
                total_grouping_loss / len(self.train_loader),
                total_ce_loss / len(self.train_loader)]
        return loss

    def test_epoch(self, cuda=True):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        with torch.no_grad():
            self.model.eval()
            total_grouping_loss = 0
            total_permutation_loss = 0
            total_duplication_loss = 0
            total_ce_loss = 0
            total_loss = 0
            bleu = calculate_aligner_bleu(self.tokenizer, self.model, self.val_loader)
            for batch_idx, (tokens, targets, alignments) in enumerate(self.val_loader):
                if cuda:
                    tokens = tokens.cuda()
                    targets = targets.cuda()
                    alignments = alignments.cuda()

                duplication_matrix, permutation_matrix, grouping_matrix = decompose_alignments(alignments, cuda=cuda)

                output, (duplication_probs, duplication_matrix_p, permutation_matrix_p, grouping_probs,
                         grouping_matrix_p) = self.model(tokens.to(dtype=torch.int32),
                                                         create_mask(tokens.permute(1, 0), targets.permute(1, 0), cuda)[
                                                             0])

                duplication_loss = self.groups_loss(duplication_probs).mean()
                permutation_loss = self.permutation_loss(permutation_matrix_p,
                                                         permutation_matrix).mean()
                grouping_loss = self.groups_loss(grouping_probs).mean()
                ce_loss = self.cross_enthropy(output[:, :targets.shape[1]].permute(0, 2, 1),
                                              targets.to(dtype=torch.long)).mean()

                loss = ce_loss + self.lmbda * (duplication_loss + permutation_loss + grouping_loss)
                total_loss += loss.item()
                total_grouping_loss += grouping_loss.item()
                total_permutation_loss += permutation_loss.item()
                total_duplication_loss += duplication_loss.item()
                total_ce_loss += ce_loss.item()

                print('\rVal loss: %4f, BLEU: %4f, Batch: %d of %d' % (
                    loss.item(), bleu, batch_idx + 1, len(self.val_loader)), end='')
            print()
            loss = [total_loss / len(self.val_loader),
                    total_duplication_loss / len(self.val_loader),
                    total_permutation_loss / len(self.val_loader),
                    total_grouping_loss / len(self.val_loader),
                    total_ce_loss / len(self.val_loader), bleu]
            return loss

    def output(self, cuda=True):
        self.model.eval()
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        tokens, targets, alignments = next(iter(self.val_loader))
        tokens = tokens[1:2].to(dtype=torch.int32)
        targets = targets[1:2].to(dtype=torch.int32)

        if cuda:
            tokens = tokens.cuda()
            targets = targets.cuda()
        output, _ = self.model(tokens.to(dtype=torch.int32),
                               create_mask(tokens.permute(1, 0), targets.permute(1, 0), cuda)[0])
        output = output.argmax(dim=-1)

        source = self.tokenizer.decode(remove_tech_tokens(list(tokens[0, :].cpu().detach().numpy())))
        original = self.tokenizer.decode(remove_tech_tokens(list(targets[0, :].cpu().detach().numpy())))
        generated = self.tokenizer.decode(remove_tech_tokens(list(output[0, :].cpu().detach().numpy())))

        print('Source: {}'.format(source))
        print('Original: {}'.format(original))
        print('Generated: {}'.format(generated))

    @staticmethod
    def log(epoch, train_loss, test_loss):
        wandb.log({
            'train': {
                'loss': train_loss[0],
                'duplication_loss': train_loss[1],
                'permutation_loss': train_loss[2],
                'grouping_loss': train_loss[3],
                'cross_enthropy_loss': train_loss[4]
            },
            'val': {
                'loss': test_loss[0],
                'duplication_loss': test_loss[1],
                'permutation_loss': test_loss[2],
                'grouping_loss': test_loss[3],
                'cross_enthropy_loss': test_loss[4],
                'bleu': test_loss[5]
            },
            'epoch': epoch
        })

    @staticmethod
    def log_train(train_loss, duplication_loss, permutation_loss, grouping_loss, ce_loss):
        wandb.log({
            'train': {
                'loss': train_loss,
                'duplication_loss': duplication_loss,
                'permutation_loss': permutation_loss,
                'grouping_loss': grouping_loss,
                'cross_enthropy_loss': ce_loss
            }
        })

    @staticmethod
    def log_test(test_loss, duplication_loss, permutation_loss, grouping_loss, ce_loss):
        wandb.log({
            'test': {
                'loss': test_loss,
                'duplication_loss': duplication_loss,
                'permutation_loss': permutation_loss,
                'grouping_loss': grouping_loss,
                'cross_enthropy_loss': ce_loss,
            }
        })


class CTCAligNARTrainer(Trainer):
    def __init__(self, model: nn.Module, tokenizer, vocab, train_loader, val_loader, tf=0.5, lr=5e-4, betas=(0.9, 0.999),
                 project="ctc_translation", name='ctc_model', save_every=None, save_path='./', lmbda=0.75,
                 resume=False):
        super().__init__(model, tokenizer, vocab, train_loader, val_loader, lr, betas, project, name, save_every, save_path, resume)
        self.tf = tf
        self.lmbda = lmbda
        self.groups_loss = GroupsLoss()
        self.permutation_loss = nn.CrossEntropyLoss()  # nn.KLDivLoss(reduction="batchmean")
        self.ctc_criterion = nn.CTCLoss(blank=self.vocab.t2i['<CTC>'], zero_infinity=True)
        self.scheduler = LinearLR(self.optimizer, start_factor=0.0005, total_iters=10000)

    def train_epoch(self, cuda=True, clip=1):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.train()

        total_loss = 0
        total_permutation_loss = 0
        total_duplication_loss = 0
        total_ctc_loss = 0
        for batch_idx, (tokens, targets, alignments) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if cuda:
                tokens = tokens.cuda()
                targets = targets.cuda()
                alignments = alignments.cuda()

            input_lengths = (tokens != 0).sum(dim=1) * 3
            target_lengths = (targets != 0).sum(dim=1)

            duplication_matrix, permutation_matrix, grouping_matrix = decompose_alignments(alignments, cuda=cuda)
            teacher_forcing = True if random.random() < self.tf else False

            ctc, (duplication_probs, duplication_matrix_p, permutation_matrix_p) = self.model(
                tokens.to(dtype=torch.int32),
                create_mask(tokens.permute(1, 0), targets.permute(1, 0), cuda)[0],
                [duplication_matrix.to(dtype=torch.float16),
                 permutation_matrix.to(dtype=torch.float16)],
                teacher_forcing=teacher_forcing)
            loss = 0
            ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float32), targets.to(dtype=torch.int32),
                                          input_lengths=input_lengths, target_lengths=target_lengths)
            duplication_loss = self.groups_loss(duplication_probs).mean()
            permutation_loss = self.permutation_loss(permutation_matrix_p,
                                                     0.9 * permutation_matrix + 0.1 / permutation_matrix.shape[
                                                         -1]).mean()
            if ctc_loss.isnan():
                print(teacher_forcing)
                print(ctc.permute(1, 0, 2))
                print(targets)
                raise

            if permutation_loss.isnan():
                print(teacher_forcing)
                print(permutation_matrix_p)
                print(permutation_matrix)
                raise
            loss += ctc_loss + permutation_loss + duplication_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_ctc_loss += ctc_loss.item()
            total_permutation_loss += permutation_loss.item()
            total_duplication_loss += duplication_loss.item()

            self.log_train(loss.item(),
                           permutation_loss.item(),
                           duplication_loss.item(),
                           ctc_loss.item())
            print('\rTrain loss: %4f, Batch: %d of %d' % (
                loss.item(), batch_idx + 1, len(self.train_loader)), end='')
        print()
        loss = [total_loss / len(self.train_loader),
                total_permutation_loss / len(self.train_loader),
                total_duplication_loss / len(self.train_loader),
                total_ctc_loss / len(self.train_loader)]
        return loss

    def test_epoch(self, cuda=True):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            total_permutation_loss = 0
            total_duplication_loss = 0
            total_ctc_loss = 0
            bleu = calculate_ctc_aligner_bleu(self.tokenizer, self.model, self.val_loader)
            for batch_idx, (tokens, targets, alignments) in enumerate(self.val_loader):
                if cuda:
                    tokens = tokens.cuda()
                    targets = targets.cuda()
                    alignments = alignments.cuda()
                input_lengths = (tokens != 0).sum(dim=1) * 3
                target_lengths = (targets != 0).sum(dim=1)

                duplication_matrix, permutation_matrix, grouping_matrix = decompose_alignments(alignments, cuda=cuda)

                ctc, (duplication_probs, duplication_matrix_p, permutation_matrix_p) = self.model(
                    tokens.to(dtype=torch.long),
                    create_mask(tokens.permute(1, 0), targets.permute(1, 0), cuda)[0])

                loss = 0
                ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float32),
                                              targets.to(dtype=torch.int32),
                                              input_lengths=input_lengths, target_lengths=target_lengths)
                duplication_loss = self.groups_loss(duplication_probs).mean()
                permutation_loss = self.permutation_loss(permutation_matrix_p,
                                                         0.9 * permutation_matrix + 0.1 / permutation_matrix.shape[
                                                             -1]).mean()
                loss += ctc_loss + permutation_loss + duplication_loss

                total_loss += loss.item()
                total_ctc_loss += ctc_loss.item()
                total_permutation_loss += permutation_loss.item()
                total_duplication_loss += duplication_loss.item()

                print('\rVal loss: %4f, BLEU: %4f, Batch: %d of %d' % (
                    loss.item(), bleu, batch_idx + 1, len(self.val_loader)), end='')
            print()
            loss = [total_loss / len(self.val_loader),
                    total_permutation_loss / len(self.val_loader),
                    total_duplication_loss / len(self.val_loader),
                    total_ctc_loss / len(self.val_loader), bleu]
            return loss

    def output(self, cuda=True):
        self.model.eval()
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        tokens, targets, _ = next(iter(self.val_loader))
        tokens = tokens[1:2].to(dtype=torch.long)
        targets = targets[1:2].to(dtype=torch.long)

        if cuda:
            tokens = tokens.cuda()
            targets = targets.cuda()
        ctc, _ = self.model(tokens, create_mask(tokens.permute(1, 0), targets.permute(1, 0), cuda)[0])

        ctc = ctc.argmax(dim=-1)

        source = self.tokenizer.decode(remove_tech_tokens(decode_ctc(list(tokens[0, :].cpu().detach().numpy()))))
        original = self.tokenizer.decode(remove_tech_tokens(decode_ctc(list(targets[0, :].cpu().detach().numpy()))))
        generated = self.tokenizer.decode(remove_tech_tokens(decode_ctc(list(ctc[0, :].cpu().detach().numpy()))))

        print('Source: {}'.format(source))
        print('Original: {}'.format(original))
        print('Generated: {}'.format(generated))

    @staticmethod
    def log(epoch, train_loss, test_loss):
        wandb.log({
            'train': {
                'loss': train_loss[0],
                'permutation_loss': train_loss[1],
                'duplication_loss': train_loss[2],
                'ctc_loss': train_loss[3],
            },
            'val': {
                'loss': test_loss[0],
                'permutation_loss': test_loss[1],
                'duplication_loss': train_loss[2],
                'ctc_loss': test_loss[3],
                'BLEU': test_loss[4]
            },
            'epoch': epoch
        })

    @staticmethod
    def log_train(train_loss, permutation_loss, duplication_loss, ctc_loss):
        wandb.log({
            'train': {
                'loss': train_loss,
                'permutation_loss': permutation_loss,
                'duplication_loss': duplication_loss,
                'ctc_loss': ctc_loss
            }
        })

    @staticmethod
    def log_test(test_loss, permutation_loss, duplication_loss, ctc_loss):
        wandb.log({
            'test': {
                'loss': test_loss,
                'permutation_loss': permutation_loss,
                'duplication_loss': duplication_loss,
                'ctc_loss': ctc_loss
            }
        })
