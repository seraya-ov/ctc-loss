import torch
from nltk.translate.bleu_score import corpus_bleu


def generate_square_subsequent_mask(sz, cuda=True):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    if cuda:
        return mask.cuda()
    return mask.cpu()


def decode_ctc(sent):
    if not sent:
        return sent
    last_token = sent[0]
    real_sent = []
    for token in sent[1:]:
        if token != '<CTC>' and token != last_token:
            real_sent.append(last_token)
        last_token = token
    return real_sent


def cut_on_eos(tokens_iter):
    for token in tokens_iter:
        if token == '<EOS>':
            break
        yield token


def remove_tech_tokens(tokens_iter, tokens_to_remove=None):
    if tokens_to_remove is None:
        tokens_to_remove = ['<SOS>', '<UNK>', '<PAD>']
    return [x for x in tokens_iter if x not in tokens_to_remove]


def generate_translation(src, model, vocabs):
    model.eval()

    ctc = model(src)
    output = ctc[:, 1:].argmax(-1)

    original = remove_tech_tokens(cut_on_eos([vocabs[0].i2t[x] for x in list(src[0, :].cpu().detach().numpy())]))
    generated = decode_ctc(
        remove_tech_tokens(cut_on_eos([vocabs[1].i2t[x] for x in list(output[0, :].cpu().detach().squeeze().numpy())])))

    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()


def generate_trf_translation(src, model, vocabs):
    model.eval()

    ctc = model(src.permute(1, 0).cpu(),
                generate_square_subsequent_mask(src.shape[1], False))
    output = ctc[:, 1:].argmax(-1)

    original = remove_tech_tokens(cut_on_eos([vocabs[0].i2t[x] for x in list(src[0, :].cpu().detach().numpy())]))
    generated = remove_tech_tokens(
        cut_on_eos([vocabs[1].i2t[x] for x in list(output[0, :].cpu().detach().squeeze().numpy())]))

    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()


def get_text(x, vocabs):
    generated = remove_tech_tokens(cut_on_eos([vocabs[1].i2t[elem] for elem in list(x)]))
    return generated


def calculate_ctc_bleu(model, loader, vocabs):
    model.eval()
    generated = []
    original = []
    for src, trg in loader:
        ctc = model(src.cuda(), trg.cuda())
        output = ctc[:, 1:].argmax(-1)

        original.extend([decode_ctc(get_text(x, vocabs)) for x in trg.cpu().numpy()])
        generated.extend([decode_ctc(get_text(x, vocabs)) for x in output.detach().cpu().squeeze().numpy()])

    return corpus_bleu([[text] for text in original], generated) * 100


def calculate_ctc_trf_bleu(model, loader, vocabs):
    model.eval()
    generated = []
    original = []
    for src, trg in loader:
        ctc = model(src.permute(1, 0).cuda(),
                    generate_square_subsequent_mask(src.shape[1], True))
        output = ctc[:, 1:].argmax(-1)

        original.extend([decode_ctc(get_text(x, vocabs)) for x in trg.cpu().numpy()])
        generated.extend([decode_ctc(get_text(x, vocabs)) for x in output.detach().cpu().squeeze().numpy()])

    return corpus_bleu([[text] for text in original], generated) * 100
