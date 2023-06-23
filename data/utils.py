import torch
from sacrebleu.metrics import BLEU

import numpy as np

from scipy.optimize import linear_sum_assignment


def generate_square_subsequent_mask(sz, cuda=True):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(dtype=torch.float16)
    if cuda:
        return mask.cuda()
    return mask.cpu()


def create_mask(src, tgt, cuda=True, pad_idx=0):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).to(dtype=torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)

    if cuda:
        return src_mask.cuda(), tgt_mask.cuda(), src_padding_mask.cuda(), tgt_padding_mask.cuda()
    return src_mask.cpu(), tgt_mask.cpu(), src_padding_mask.cpu(), tgt_padding_mask.cpu()


def decode_ctc(sent):
    if sent is None or len(sent) == 0:
        return sent
    last_token = sent[0]
    real_sent = []
    for token in sent[1:]:
        if last_token != 2 and token != last_token:
            real_sent.append(last_token)
        last_token = token
    if last_token != 2:
        real_sent.append(last_token)
    return real_sent


def cut_on_eos(tokens_iter):
    for token in tokens_iter:
        if token == '<EOS>':
            break
        yield token


def remove_tech_tokens(tokens_iter, tokens_to_remove=(0, 1, 2, 3, 4)):
    return [x - 5 for x in tokens_iter if x not in tokens_to_remove]


def generate_translation(tokenizer, src, trg, model):
    model.eval()

    ctc = model(src, trg)
    output = ctc[:, 1:].argmax(-1)

    source = tokenizer.decode(remove_tech_tokens(list(src[0, :].cpu().detach().numpy())))
    original = tokenizer.decode(remove_tech_tokens(list(trg[0, :].cpu().detach().numpy())))
    generated = tokenizer.decode(remove_tech_tokens(list(output[0, :].cpu().detach().numpy())))

    print('Source: {}'.format(' '.join(source)))
    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()


def generate_trf_translation(tokenizer, src, trg, model):
    model.eval()

    ctc = model(src.cpu(), None)
    output = ctc[:, 1:].argmax(-1)

    source = tokenizer.decode(remove_tech_tokens(list(src[0, :].cpu().detach().numpy())))
    original = tokenizer.decode(remove_tech_tokens(list(trg[0, :].cpu().detach().numpy())))
    generated = tokenizer.decode(remove_tech_tokens(list(output[0, :].cpu().detach().numpy())))

    print('Source: {}'.format(' '.join(source)))
    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()


def get_text(tokenizer, x):
    generated = tokenizer.decode(remove_tech_tokens(x))
    return generated


def calculate_ctc_bleu(tokenizer, model, loader):
    bleu = BLEU()
    model.eval()
    generated = []
    original = []
    for src, trg, _ in loader:
        ctc = model(src.cuda(), trg.cuda())
        output = ctc[:, 1:].argmax(-1)

        original.extend([get_text(tokenizer, decode_ctc(x)) for x in trg.cpu().numpy()])
        generated.extend([get_text(tokenizer, decode_ctc(x)) for x in output.detach().cpu().squeeze().numpy()])
    return bleu.corpus_score(generated, list([original])).score


def calculate_transformer_bleu(tokenizer, model, loader):
    bleu = BLEU()
    model.eval()
    generated = []
    original = []
    for src, trg, _ in loader:
        outputs = model(src.cuda(), None, None, None, None, None, use_teacher_forcing=False)
        output = outputs.argmax(-1)

        original.extend([get_text(tokenizer, x) for x in trg.cpu().numpy()])
        generated.extend([get_text(tokenizer, x) for x in output.detach().cpu().squeeze().numpy()])
    return bleu.corpus_score(generated, list([original])).score


def calculate_aligner_bleu(tokenizer, model, loader):
    bleu = BLEU()
    model.eval()
    generated = []
    original = []
    for src, trg, _ in loader:
        output, _ = model(src.cuda(), None)
        output = output[:, 1:].argmax(-1)

        original.extend([get_text(tokenizer, x) for x in trg.cpu().numpy()])
        generated.extend([get_text(tokenizer, x) for x in output.detach().cpu().squeeze().numpy()])

    return bleu.corpus_score(generated, list([original])).score


def calculate_ctc_aligner_bleu(tokenizer, model, loader):
    bleu = BLEU()
    model.eval()
    generated = []
    original = []
    for src, trg, _ in loader:
        ctc, _ = model(src.cuda(), None)
        output = ctc[:, 1:].argmax(-1)

        original.extend([get_text(tokenizer, decode_ctc(x)) for x in trg.cpu().numpy()])
        generated.extend([get_text(tokenizer, decode_ctc(x)) for x in output.detach().cpu().squeeze().numpy()])

    return bleu.corpus_score(generated, list([original])).score


def calculate_ctc_trf_bleu(tokenizer, model, loader):
    bleu = BLEU()
    model.eval()
    generated = []
    original = []
    for src, trg, _ in loader:
        ctc = model(src.cuda(), None)
        output = ctc[:, 1:].argmax(-1)

        original.extend([get_text(tokenizer, decode_ctc(x)) for x in trg.cpu().numpy()])
        generated.extend([get_text(tokenizer, decode_ctc(x)) for x in output.detach().cpu().squeeze().numpy()])

    return bleu.corpus_score(generated, list([original])).score


def decompose_alignments(alignments, cuda=True):
    groups = torch.cumsum(alignments.sum(2), 1).unsqueeze(2)
    lengths = torch.cumsum(torch.clamp(alignments.sum(1), 0, 6), 1).unsqueeze(1)

    grouping_matrix = torch.zeros((alignments.shape[0], alignments.shape[1], alignments.shape[2] * 3), dtype=torch.float32) + torch.arange(1, 1 + alignments.shape[2] * 3, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    if cuda:
        grouping_matrix = grouping_matrix.cuda()
    grouping_matrix[:, 0, :] = torch.clamp(groups[:, 0, :] - grouping_matrix[:, 0, :] + 1, 0, 1)
    grouping_matrix[:, 1:, :] = torch.clamp(groups[:, 1:, :] - grouping_matrix[:, 1:, :] + 1, 0, 1) * torch.clamp(grouping_matrix[:, 1:, :] - groups[:, :-1, :], 0, 1)

    duplication_matrix = torch.zeros((alignments.shape[0], alignments.shape[2] * 3, alignments.shape[2]), dtype=torch.float32) + torch.arange(1, 1 + alignments.shape[2] * 3, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    if cuda:
        duplication_matrix = duplication_matrix.cuda()
    duplication_matrix[:, :, 0] = torch.clamp(lengths[:, :, 0] - duplication_matrix[:, :, 0] + 1, 0, 1)
    duplication_matrix[:, :, 1:] = torch.clamp(lengths[:, :, 1:] - duplication_matrix[:, :, 1:] + 1, 0, 1) * torch.clamp(duplication_matrix[:, :, 1:] - lengths[:, :, :-1], 0, 1)

    permutation_matrix = (grouping_matrix.permute(0, 2, 1) / (alignments.to(dtype=torch.float32).sum(2).unsqueeze(1) + 1e-16))@alignments.to(dtype=torch.float32)@(duplication_matrix.permute(0, 2, 1) / (alignments.to(dtype=torch.float32).sum(1).unsqueeze(2) + 1e-16))

    permutation_matrix = permutation_matrix.cpu().detach().numpy()

    for i in range(permutation_matrix.shape[0]):
        permutation_matrix[i] += 1e-16
        xs, ys = linear_sum_assignment(-1 * np.log(permutation_matrix[i]))
        permutation_matrix[i] *= 0
        permutation_matrix[i, xs, ys] = 1

    permutation_matrix = torch.Tensor(permutation_matrix).to(device=alignments.device, dtype=torch.float32)

    return duplication_matrix, permutation_matrix, grouping_matrix
