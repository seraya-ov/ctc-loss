from nltk.translate.bleu_score import corpus_bleu


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

    ctc, output = model(src, src, False)
    output = output[:, 1:].argmax(-1)

    original = remove_tech_tokens(cut_on_eos([vocabs[0].i2t[x] for x in list(src[0, :].cpu().detach().numpy())]))
    generated = remove_tech_tokens(cut_on_eos([vocabs[1].i2t[x] for x in list(output[0, :].cpu().detach().numpy())]))

    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()


def get_text(x, vocabs):
    generated = remove_tech_tokens(cut_on_eos([vocabs[1].i2t[elem] for elem in list(x)]))
    return generated


def calculate_bleu(model, loader, vocabs):
    generated = []
    original = []
    for src, trg in loader:
        ctc, output = model(src, trg, False)
        output = output[:, 1:].argmax(-1)

        original.extend([get_text(x, vocabs) for x in trg.cpu().numpy()])
        generated.extend([get_text(x, vocabs) for x in output.detach().cpu().numpy()])

    return corpus_bleu([[text] for text in original], generated) * 100


def calculate_ctc_bleu(model, loader, vocabs):
    generated = []
    original = []
    for src, trg in loader:
        ctc, output = model(src, trg, False)
        output = output[:, 1:].argmax(-1)

        original.extend([decode_ctc(get_text(x, vocabs)) for x in trg.cpu().numpy()])
        generated.extend([decode_ctc(get_text(x, vocabs)) for x in output.detach().cpu().numpy()])

    return corpus_bleu([[text] for text in original], generated) * 100
