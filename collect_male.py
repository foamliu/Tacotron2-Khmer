import os
from shutil import copyfile

from config import speaker_info


def get_speaker_dict():
    with open(speaker_info, 'r') as file:
        lines = file.readlines()

    speaker_dict = {'S' + l.strip().split(' ')[0]: l.strip().split(' ')[1] for l in lines}
    return speaker_dict


if __name__ == "__main__":
    speaker_dict = get_speaker_dict()

    for item in speaker_dict.items():
        folder = os.path.join('data/data_aishell/wav/train', item[0])
        if item[1] == 'M' and os.path.isdir(folder):
            file = [f for f in os.listdir(folder) if f.endswith('.wav')][0]
            file = os.path.join(folder, file)
            target = 'audios/{}.wav'.format(item[0])
            print(file)
            copyfile(file, target)
