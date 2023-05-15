import wandb
import torch
import torch.nn as nn

from translation.data.utils import decode_ctc, generate_square_subsequent_mask
from translation.train.losses import GroupsLoss


class Trainer:
    def __init__(self, model: nn.Module, train_loader, val_loader, vocabs, tf=0.25, lr=3e-4, betas=(0.9, 0.999),
                 project="ctc_translation", name='ctc_model', save_every=None, save_path='./'):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabs = vocabs
        self.model = model
        self.save_path = save_path
        self.save_every = save_every
        self.name = name
        self.project = project
        self.tf = tf
        wandb.init(project=project, name=name)

    def train_epoch(self, cuda=True, clip=1):
        pass

    def test_epoch(self, cuda=True):
        pass

    def output(self, cuda=True):
        pass

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

    @staticmethod
    def log_test(test_loss):
        wandb.log({
            'test': {
                'loss': test_loss
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


class TranslationLSTMTCTCrainer(Trainer):
    def __init__(self, model: nn.Module, train_loader, val_loader, vocabs, tf=0.25, lr=3e-4, betas=(0.9, 0.999),
                 project="ctc_translation", name='ctc_model', save_every=None, save_path='./'):
        super().__init__(model, train_loader, val_loader, vocabs, tf, lr, betas, project, name, save_every, save_path)
        self.ctc_criterion = nn.CTCLoss(blank=vocabs[0].t2i['<CTC>'], zero_infinity=True)

    def train_epoch(self, cuda=True, clip=1):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.train()
        total_loss = 0
        total_ctc_loss = 0
        for batch_idx, (tokens, targets, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if cuda:
                tokens = tokens.cuda()
                targets = targets.cuda()
            target_lengths = (targets != 0).sum(dim=1)
            ctc = self.model(tokens.to(dtype=torch.long), targets.to(dtype=torch.long))
            input_lengths = (tokens != 0).sum(dim=1)
            loss = 0
            ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float), targets.to(dtype=torch.long),
                                          input_lengths=input_lengths, target_lengths=target_lengths)
            total_ctc_loss += ctc_loss.item()
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
            total_ctc_loss = 0
            for batch_idx, (tokens, targets, _) in enumerate(self.val_loader):
                if cuda:
                    tokens = tokens.cuda()
                    targets = targets.cuda()
                target_lengths = (targets != 0).sum(dim=1)
                ctc = self.model(tokens.to(dtype=torch.long), targets.to(dtype=torch.long))
                input_lengths = (tokens != 0).sum(dim=1)

                loss = 0
                ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float), targets.to(dtype=torch.long),
                                              input_lengths=input_lengths, target_lengths=target_lengths)
                total_ctc_loss += ctc_loss.item()
                loss += ctc_loss
                total_loss += loss.item()

                self.log_test(total_loss / (batch_idx + 1))
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
        tokens, targets, _ = next(iter(self.val_loader))
        tokens = tokens[1:2].to(dtype=torch.long)
        targets = targets[1:2].to(dtype=torch.long)
        if cuda:
            tokens = tokens.cuda()
            targets = targets.cuda()
        ctc = self.model(tokens, targets)
        ctc = ctc.argmax(dim=-1)
        summ = '<SOS>'
        ctc_sent = []
        for di in range(1, targets.shape[1]):
            summ += self.vocabs[1].i2t[targets[0, di].cpu().detach().squeeze().item()] + ' '
        for di in range(ctc.shape[1]):
            ctc_sent.append(self.vocabs[1].i2t[ctc[0, di].cpu().detach().squeeze().item()])

        print(summ[:-1])
        print(' '.join(decode_ctc(ctc_sent)))


class TranslationTransformerCTCTrainer(Trainer):
    def __init__(self, model: nn.Module, train_loader, val_loader, vocabs, tf=0.25, lr=3e-4, betas=(0.9, 0.999),
                 project="ctc_translation", name='ctc_model', save_every=None, save_path='./'):
        super().__init__(model, train_loader, val_loader, vocabs, tf, lr, betas, project, name, save_every, save_path)
        self.ctc_criterion = nn.CTCLoss(blank=vocabs[0].t2i['<CTC>'], zero_infinity=True)

    def train_epoch(self, cuda=True, clip=1):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.train()
        total_loss = 0
        total_ctc_loss = 0
        for batch_idx, (tokens, targets, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if cuda:
                tokens = tokens.cuda()
                targets = targets.cuda()
            target_lengths = (targets != 0).sum(dim=1)
            ctc = self.model(tokens.to(dtype=torch.long).permute(1, 0),
                             generate_square_subsequent_mask(tokens.shape[1], cuda))
            input_lengths = (tokens != 0).sum(dim=1)
            loss = 0
            ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float), targets.to(dtype=torch.long),
                                          input_lengths=input_lengths, target_lengths=target_lengths)
            total_ctc_loss += ctc_loss.item()
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
            total_ctc_loss = 0
            for batch_idx, (tokens, targets, _) in enumerate(self.val_loader):
                if cuda:
                    tokens = tokens.cuda()
                    targets = targets.cuda()
                target_lengths = (targets != 0).sum(dim=1)
                ctc = self.model(tokens.to(dtype=torch.long).permute(1, 0),
                                 generate_square_subsequent_mask(tokens.shape[1], cuda))
                input_lengths = (tokens != 0).sum(dim=1)

                loss = 0
                ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float), targets.to(dtype=torch.long),
                                              input_lengths=input_lengths, target_lengths=target_lengths)
                total_ctc_loss += ctc_loss.item()
                loss += ctc_loss
                total_loss += loss.item()

                self.log_test(total_loss / (batch_idx + 1))
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
        tokens, targets, _ = next(iter(self.val_loader))
        tokens = tokens[1:2].to(dtype=torch.long)
        targets = targets[1:2].to(dtype=torch.long)
        batch_size = tokens.shape[0]
        if cuda:
            tokens = tokens.cuda()
            targets = targets.cuda()
        ctc = self.model(tokens.permute(1, 0),
                         generate_square_subsequent_mask(tokens.shape[1], cuda))
        ctc = ctc.argmax(dim=-1)
        summ = '<SOS>'
        ctc_sent = []
        for di in range(1, targets.shape[1]):
            summ += self.vocabs[1].i2t[targets[0, di].cpu().detach().squeeze().item()] + ' '
        for di in range(ctc.shape[1]):
            ctc_sent.append(self.vocabs[1].i2t[ctc[0, di].cpu().detach().squeeze().item()])

        print(summ[:-1])
        print(' '.join(decode_ctc(ctc_sent)))


class AligNARTrainer(Trainer):
    def __init__(self, model: nn.Module, train_loader, val_loader, vocabs, tf=0.25, lr=3e-4, betas=(0.9, 0.999),
                 project="ctc_translation", name='ctc_model', save_every=None, save_path='./', lmbda=0.5):
        super().__init__(model, train_loader, val_loader, vocabs, tf, lr, betas, project, name, save_every, save_path)
        self.lmbda = lmbda
        self.groups_loss = GroupsLoss()
        self.permutation_loss = nn.KLDivLoss()
        self.cross_enthropy = nn.CrossEntropyLoss()

    def train_epoch(self, cuda=True, clip=1):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.train()
        total_grouping_loss = 0
        total_permutation_loss = 0
        total_duplication_loss = 0
        total_loss = 0
        for batch_idx, (tokens, targets, alignments) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if cuda:
                tokens = tokens.cuda()
                targets = targets.cuda()
                alignments = alignments.cuda()

            groups = torch.cumsum(alignments.sum(2), 1).unsqueeze(2)
            lengths = torch.cumsum(torch.clamp(alignments.sum(1), 0, 3), 0).unsqueeze(1)

            grouping_matrix = torch.zeros(
                (alignments.shape[0], alignments.shape[1], alignments.shape[1] * 3)) + torch.arange(1, 1 +
                                                                                                    alignments.shape[
                                                                                                        1] * 3).unsqueeze(
                0).unsqueeze(1)
            if cuda:
                grouping_matrix = grouping_matrix.cuda()
            grouping_matrix[:, 0, :] = torch.clamp(groups[:, 0, :] - grouping_matrix[:, 0, :] + 1, 0, 1)
            grouping_matrix[:, 1:, :] = torch.clamp(groups[:, 1:, :] - grouping_matrix[:, 1:, :] + 1, 0,
                                                    1) * torch.clamp(grouping_matrix[:, 1:, :] - groups[:, :-1, :], 0,
                                                                     1)

            duplication_matrix = torch.zeros(
                (alignments.shape[0], alignments.shape[1] * 3, alignments.shape[2])) + torch.arange(1, 1 +
                                                                                                    alignments.shape[
                                                                                                        1] * 3).unsqueeze(
                0).unsqueeze(2)
            if cuda:
                duplication_matrix = duplication_matrix.cuda()
            duplication_matrix[:, :, 0] = torch.clamp(lengths[:, :, 0] - duplication_matrix[:, :, 0] + 1, 0, 1)
            duplication_matrix[:, :, 1:] = torch.clamp(lengths[:, :, 1:] - duplication_matrix[:, :, 1:] + 1, 0,
                                                       1) * torch.clamp(
                duplication_matrix[:, :, 1:] - lengths[:, :, :-1], 0, 1)

            permutation_matrix = grouping_matrix.permute(0, 2, 1) @ alignments @ duplication_matrix.permute(0, 2, 1)

            # print(tokens.shape, alignments.shape, duplication_matrix.shape)

            output, (duplication_probs, duplication_matrix_p, permutation_matrix_p, grouping_probs,
                     grouping_matrix_p) = self.model(tokens.to(dtype=torch.long).permute(1, 0),
                                                     generate_square_subsequent_mask(tokens.shape[1], cuda),
                                                     [duplication_matrix.to(dtype=torch.float),
                                                      permutation_matrix.to(dtype=torch.float),
                                                      grouping_matrix.to(dtype=torch.float)])

            duplication_loss = self.groups_loss(duplication_probs).mean()
            permutation_loss = self.permutation_loss(permutation_matrix_p, permutation_matrix).mean()
            grouping_loss = self.groups_loss(grouping_probs).mean()
            ce_loss = self.cross_enthropy(output.permute(0, 2, 1), targets).mean()

            loss = ce_loss + self.lmbda * (duplication_loss + permutation_loss + grouping_loss)
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
            for batch_idx, (tokens, targets, alignments) in enumerate(self.val_loader):
                if cuda:
                    tokens = tokens.cuda()
                    targets = targets.cuda()
                    alignments = alignments.cuda()

                targets = targets.repeat(1, 3)
                targets[:, targets.shape[1] // 3:] = 0

                groups = torch.cumsum(alignments.sum(2), 1).unsqueeze(2)
                lengths = torch.cumsum(torch.clamp(alignments.sum(1), 0, 3), 0).unsqueeze(1)

                grouping_matrix = torch.zeros(
                    (alignments.shape[0], alignments.shape[1], alignments.shape[1] * 3)) + torch.arange(1, 1 +
                                                                                                        alignments.shape[
                                                                                                            1] * 3).unsqueeze(
                    0).unsqueeze(1)
                if cuda:
                    grouping_matrix = grouping_matrix.cuda()
                grouping_matrix[:, 0, :] = torch.clamp(groups[:, 0, :] - grouping_matrix[:, 0, :] + 1, 0, 1)
                grouping_matrix[:, 1:, :] = torch.clamp(groups[:, 1:, :] - grouping_matrix[:, 1:, :] + 1, 0,
                                                        1) * torch.clamp(grouping_matrix[:, 1:, :] - groups[:, :-1, :],
                                                                         0, 1)

                duplication_matrix = torch.zeros(
                    (alignments.shape[0], alignments.shape[1] * 3, alignments.shape[2])) + torch.arange(1, 1 +
                                                                                                        alignments.shape[
                                                                                                            1] * 3).unsqueeze(
                    0).unsqueeze(2)
                if cuda:
                    duplication_matrix = duplication_matrix.cuda()
                duplication_matrix[:, :, 0] = torch.clamp(lengths[:, :, 0] - duplication_matrix[:, :, 0] + 1, 0, 1)
                duplication_matrix[:, :, 1:] = torch.clamp(lengths[:, :, 1:] - duplication_matrix[:, :, 1:] + 1, 0,
                                                           1) * torch.clamp(
                    duplication_matrix[:, :, 1:] - lengths[:, :, :-1], 0, 1)

                permutation_matrix = grouping_matrix.permute(0, 2, 1) @ alignments @ duplication_matrix.permute(0, 2, 1)

                output, (duplication_probs, duplication_matrix_p, permutation_matrix_p, grouping_probs,
                         grouping_matrix_p) = self.model(tokens.to(dtype=torch.long).permute(1, 0),
                                                         generate_square_subsequent_mask(tokens.shape[1], cuda))

                duplication_loss = self.groups_loss(duplication_probs).mean()
                permutation_loss = self.permutation_loss(permutation_matrix_p, permutation_matrix).mean()
                grouping_loss = self.groups_loss(grouping_probs).mean()
                ce_loss = self.cross_enthropy(output.permute(0, 2, 1), targets).mean()

                loss = ce_loss + self.lmbda * (duplication_loss + permutation_loss + grouping_loss)
                total_loss += loss.item()

                self.log_test(total_loss / (batch_idx + 1))
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
        tokens, targets, alignments = next(iter(self.val_loader))
        tokens = tokens[1:2].to(dtype=torch.long)
        targets = targets[1:2].to(dtype=torch.long)
        if cuda:
            tokens = tokens.cuda()
            targets = targets.cuda()
        output, _ = self.model(tokens.to(dtype=torch.long).permute(1, 0),
                               generate_square_subsequent_mask(tokens.shape[1], cuda))
        output = output.argmax(dim=-1)

        summ = '<SOS>'
        sent = []
        for di in range(1, targets.shape[1]):
            summ += self.vocabs[1].i2t[targets[0, di].cpu().detach().squeeze().item()] + ' '
        for di in range(output.shape[1]):
            sent.append(self.vocabs[1].i2t[output[0, di].cpu().detach().squeeze().item()])

        print(summ[:-1])
        print(' '.join(sent))


class CTCAligNARTrainer(Trainer):
    def __init__(self, model: nn.Module, train_loader, val_loader, vocabs, tf=0.25, lr=3e-4, betas=(0.9, 0.999),
                 project="ctc_translation", name='ctc_model', save_every=None, save_path='./', lmbda=0.5):
        super().__init__(model, train_loader, val_loader, vocabs, tf, lr, betas, project, name, save_every, save_path)
        self.lmbda = lmbda
        self.groups_loss = GroupsLoss()
        self.permutation_loss = nn.KLDivLoss()
        self.ctc_criterion = nn.CTCLoss(blank=vocabs[0].t2i['<CTC>'], zero_infinity=True)

    def train_epoch(self, cuda=True, clip=1):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.train()
        total_loss = 0
        for batch_idx, (tokens, targets, alignments) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if cuda:
                tokens = tokens.cuda()
                targets = targets.cuda()
                alignments = alignments.cuda()

            target_lengths = (targets != 0).sum(dim=1)

            groups = torch.cumsum(alignments.sum(2), 1).unsqueeze(2)
            lengths = torch.cumsum(torch.clamp(alignments.sum(1), 0, 3), 0).unsqueeze(1)

            grouping_matrix = torch.zeros(
                (alignments.shape[0], alignments.shape[1], alignments.shape[1] * 3)) + torch.arange(1, 1 +
                                                                                                    alignments.shape[
                                                                                                        1] * 3).unsqueeze(
                0).unsqueeze(1)
            if cuda:
                grouping_matrix = grouping_matrix.cuda()
            grouping_matrix[:, 0, :] = torch.clamp(groups[:, 0, :] - grouping_matrix[:, 0, :] + 1, 0, 1)
            grouping_matrix[:, 1:, :] = torch.clamp(groups[:, 1:, :] - grouping_matrix[:, 1:, :] + 1, 0,
                                                    1) * torch.clamp(grouping_matrix[:, 1:, :] - groups[:, :-1, :], 0,
                                                                     1)

            duplication_matrix = torch.zeros(
                (alignments.shape[0], alignments.shape[1] * 3, alignments.shape[2])) + torch.arange(1, 1 +
                                                                                                    alignments.shape[
                                                                                                        1] * 3).unsqueeze(
                0).unsqueeze(2)
            if cuda:
                duplication_matrix = duplication_matrix.cuda()
            duplication_matrix[:, :, 0] = torch.clamp(lengths[:, :, 0] - duplication_matrix[:, :, 0] + 1, 0, 1)
            duplication_matrix[:, :, 1:] = torch.clamp(lengths[:, :, 1:] - duplication_matrix[:, :, 1:] + 1, 0,
                                                       1) * torch.clamp(
                duplication_matrix[:, :, 1:] - lengths[:, :, :-1], 0, 1)

            permutation_matrix = grouping_matrix.permute(0, 2, 1) @ alignments @ duplication_matrix.permute(0, 2, 1)

            ctc, (duplication_matrix_p, permutation_matrix_p) = self.model(tokens.to(dtype=torch.long).permute(1, 0),
                                                                           generate_square_subsequent_mask(
                                                                               tokens.shape[1], cuda),
                                                                           [duplication_matrix.to(dtype=torch.float),
                                                                            permutation_matrix.to(dtype=torch.float)])
            input_lengths = (tokens != 0).sum(dim=1)
            loss = 0
            ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float), targets.to(dtype=torch.long),
                                          input_lengths=input_lengths, target_lengths=target_lengths)
            permutation_loss = self.permutation_loss(permutation_matrix_p, permutation_matrix).mean()
            loss += ctc_loss + permutation_loss
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
            total_ctc_loss = 0
            for batch_idx, (tokens, targets, alignments) in enumerate(self.val_loader):
                if cuda:
                    tokens = tokens.cuda()
                    targets = targets.cuda()
                    alignments = alignments.cuda()
                target_lengths = (targets != 0).sum(dim=1)
                targets = targets.repeat(1, 3)
                targets[:, targets.shape[1] // 3:] = 0

                groups = torch.cumsum(alignments.sum(2), 1).unsqueeze(2)
                lengths = torch.cumsum(torch.clamp(alignments.sum(1), 0, 3), 0).unsqueeze(1)

                grouping_matrix = torch.zeros(
                    (alignments.shape[0], alignments.shape[1], alignments.shape[1] * 3)) + torch.arange(1, 1 +
                                                                                                        alignments.shape[
                                                                                                            1] * 3).unsqueeze(
                    0).unsqueeze(1)
                if cuda:
                    grouping_matrix = grouping_matrix.cuda()
                grouping_matrix[:, 0, :] = torch.clamp(groups[:, 0, :] - grouping_matrix[:, 0, :] + 1, 0, 1)
                grouping_matrix[:, 1:, :] = torch.clamp(groups[:, 1:, :] - grouping_matrix[:, 1:, :] + 1, 0,
                                                        1) * torch.clamp(grouping_matrix[:, 1:, :] - groups[:, :-1, :],
                                                                         0, 1)

                duplication_matrix = torch.zeros(
                    (alignments.shape[0], alignments.shape[1] * 3, alignments.shape[2])) + torch.arange(1, 1 +
                                                                                                        alignments.shape[
                                                                                                            1] * 3).unsqueeze(
                    0).unsqueeze(2)
                if cuda:
                    duplication_matrix = duplication_matrix.cuda()
                duplication_matrix[:, :, 0] = torch.clamp(lengths[:, :, 0] - duplication_matrix[:, :, 0] + 1, 0, 1)
                duplication_matrix[:, :, 1:] = torch.clamp(lengths[:, :, 1:] - duplication_matrix[:, :, 1:] + 1, 0,
                                                           1) * torch.clamp(
                    duplication_matrix[:, :, 1:] - lengths[:, :, :-1], 0, 1)

                permutation_matrix = grouping_matrix.permute(0, 2, 1) @ alignments @ duplication_matrix.permute(0, 2, 1)

                ctc, (duplication_matrix_p, permutation_matrix_p) = self.model(
                    tokens.to(dtype=torch.long).permute(1, 0),
                    generate_square_subsequent_mask(tokens.shape[1], cuda))
                input_lengths = (tokens != 0).sum(dim=1)

                loss = 0
                ctc_loss = self.ctc_criterion(ctc.permute(1, 0, 2).to(dtype=torch.float), targets.to(dtype=torch.long),
                                              input_lengths=input_lengths, target_lengths=target_lengths)
                permutation_loss = self.permutation_loss(permutation_matrix_p, permutation_matrix).mean()
                total_ctc_loss += ctc_loss.item()
                loss += ctc_loss + permutation_loss
                total_loss += loss.item()

                self.log_test(total_loss / (batch_idx + 1))
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
        tokens, targets, _ = next(iter(self.val_loader))
        tokens = tokens[1:2].to(dtype=torch.long)
        targets = targets[1:2].to(dtype=torch.long)
        if cuda:
            tokens = tokens.cuda()
            targets = targets.cuda()
        ctc, _ = self.model(tokens.permute(1, 0),
                            generate_square_subsequent_mask(tokens.shape[1], cuda))
        ctc = ctc.argmax(dim=-1)
        summ = '<SOS>'
        ctc_sent = []
        for di in range(1, targets.shape[1]):
            summ += self.vocabs[1].i2t[targets[0, di].cpu().detach().squeeze().item()] + ' '
        for di in range(ctc.shape[1]):
            ctc_sent.append(self.vocabs[1].i2t[ctc[0, di].cpu().detach().squeeze().item()])

        print(summ[:-1])
        print(' '.join(decode_ctc(ctc_sent)))
