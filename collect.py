import pickle
import random
from shutil import copyfile

from config import data_file

with open(data_file, 'rb') as file:
    data = pickle.load(file)

split = 'train'
samples = data[split]
samples = random.sample(samples, 20)

for i, sample in enumerate(samples):
    audiopath = sample['audiopath']
    print(audiopath)
    target = 'audios/{}.wav'.format(i)
    copyfile(audiopath, target)
