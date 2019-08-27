import pickle
import random

from config import tran_file, vocab_file, training_files, validation_files
from utils import ensure_folder


def process_data():
    with open(tran_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    samples = []
    for i, line in enumerate(lines):
        tokens = line.split()
        audiopath = 'data/km_kh_male/wavs/{}.wav'.format(tokens[0])
        text = ''.join(tokens[1:])
        for token in text:
            build_vocab(token)
        samples.append('{}|{}\n'.format(audiopath, text))

    valid_ids = random.sample(range(len(samples)), 100)
    train = []
    valid = []
    for id in range(len(samples)):
        sample = samples[id]
        if id in valid_ids:
            valid.append(sample)
        else:
            train.append(sample)

    ensure_folder('filelists')

    # print(samples)
    with open(training_files, 'w', encoding='utf-8') as file:
        file.writelines(train)
    with open(validation_files, 'w', encoding='utf-8') as file:
        file.writelines(valid)

    print('num_train: ' + str(len(train)))
    print('num_valid: ' + str(len(valid)))


def build_vocab(token):
    global char2idx, idx2char
    if not token in char2idx:
        next_index = len(char2idx)
        char2idx[token] = next_index
        idx2char[next_index] = token


if __name__ == "__main__":
    char2idx = {}
    idx2char = {}

    process_data()

    data = dict()
    data['char2idx'] = char2idx
    data['idx2char'] = idx2char

    with open(vocab_file, 'wb') as file:
        pickle.dump(data, file)

    print('vocab_size: ' + str(len(data['char2idx'])))
    print('char2idx: ' + str(char2idx))
