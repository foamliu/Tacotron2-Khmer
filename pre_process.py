import argparse
import os
import pickle

import pinyin
from tqdm import tqdm

from config import wav_folder, tran_file, data_file, char_to_idx, unk_id, speaker_info
from utils import ensure_folder


def get_data(split, gender):
    print('getting {} data...'.format(split))

    speaker_dict = get_speaker_dict()

    with open(tran_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    tran_dict = dict()
    for line in lines:
        tokens = line.split()
        key = tokens[0]
        trn = ''.join(tokens[1:])
        tran_dict[key] = trn

    samples = []

    folder = os.path.join(wav_folder, split)
    ensure_folder(folder)
    dirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    for dir in tqdm(dirs):
        files = [f for f in os.listdir(dir) if f.endswith('.wav')]

        for f in files:
            audiopath = os.path.join(dir, f)
            speaker_id = audiopath.split('/')[4]
            speaker_gender = speaker_dict[speaker_id]

            if speaker_gender != gender:
                continue

            key = f.split('.')[0]
            if key in tran_dict:
                text = tran_dict[key]
                text = pinyin.get(text.strip(), format="numerical", delimiter=" ")
                text = list(text)

                temp = []
                for token in text:
                    if token in char_to_idx:
                        temp.append(char_to_idx[token])
                    else:
                        temp.append(unk_id)

                text = temp

                samples.append({'text': text, 'audiopath': audiopath})

    print('split: {}, num_files: {}'.format(split, len(samples)))
    return samples


def get_speaker_dict():
    with open(speaker_info, 'r') as file:
        lines = file.readlines()

    speaker_dict = {'S' + l.strip().split(' ')[0]: l.strip().split(' ')[1] for l in lines}
    return speaker_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Tacotron2')
    parser.add_argument('--gender', default='M', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    gender = args.gender

    print('gender: ' + gender)

    data = dict()
    data['train'] = get_data('train', gender)
    data['dev'] = get_data('dev', gender)
    data['test'] = get_data('test', gender)

    with open(data_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_dev: ' + str(len(data['dev'])))
    print('num_test: ' + str(len(data['test'])))


if __name__ == "__main__":
    main()
