import codecs

from collections import defaultdict
from nltk.tokenize import word_tokenize
from tqdm.notebook import tqdm


def read_data(read_paths, save_path, maxlen=30, langs=('german', 'english'), maxsize=5000000):
    input_file_lang1 = codecs.open(read_paths[0], 'r', 'utf-8')
    input_file_lang2 = codecs.open(read_paths[1], 'r', 'utf-8')
    output_file = codecs.open(save_path, 'w', 'utf-8')

    for i, (line1, line2) in tqdm(enumerate(zip(input_file_lang1, input_file_lang2))):
        content1, content2 = line1.strip().lower(), line2.strip().lower()
        tokenized_content1 = word_tokenize(content1, language=langs[0])
        tokenized_content2 = word_tokenize(content2, language=langs[1])
        if maxlen > len(tokenized_content1) > 1 \
                and maxlen > len(tokenized_content2) > 1:
            output_file.write(content1 + ' ||| ' + content2 + '\n')

        if i > maxsize:
            break

    input_file_lang1.close()
    input_file_lang2.close()
    output_file.close()


def read_vocab_from_corpus(vocab_path, save_path, capacity=50000, lang='english'):
    counts = defaultdict(int)
    input_file = codecs.open(vocab_path, 'r', 'utf-8')
    output_file = codecs.open(save_path, 'w', 'utf-8')

    for line in tqdm(input_file):
        content = line.strip().lower()
        tokenized_content = word_tokenize(content, language=lang)
        for word in tokenized_content:
            counts[word] += 1

    picked_words = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)[:capacity]
    for word in picked_words:
        output_file.write(word + '\n')

    input_file.close()
    output_file.close()


if __name__ == '__main__':
    pass