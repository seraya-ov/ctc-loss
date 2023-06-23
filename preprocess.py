import codecs

from tqdm.notebook import tqdm


def read_bpe_data(tokenizer, dataset, save_path, maxlen=60, maxsize=5000000):
    output_file = codecs.open(save_path, 'w', 'utf-8')

    for i, data in tqdm(enumerate(dataset)):
        line1, line2 = data['translation']['ro'], data['translation']['en']
        content1, content2 = line1.strip(), line2.strip()
        tokenized_content1 = tokenizer.encode(content1).tokens
        tokenized_content2 = tokenizer.encode(content2).tokens
        content1, content2 = ' '.join(tokenized_content1), ' '.join(tokenized_content2)
        if '|||' in line1 or '|||' in line2:
            continue
        if maxlen > len(tokenized_content1) > 1 \
                and maxlen > len(tokenized_content2) > 1:
            output_file.write(content1 + ' ||| ' + content2 + '\n')

        if i > maxsize:
            break

    output_file.close()


if __name__ == '__main__':
    pass
