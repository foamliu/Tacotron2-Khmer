import os

import librosa
from tqdm import tqdm

folder = '../data/km_kh_male/wavs'
files = [f for f in os.listdir(folder) if f.endswith('.wav')]

sampling_rate = 22050

total_duration = 0

for file in tqdm(files):
    fullpath = os.path.join(folder, file)
    y, sr = librosa.core.load(fullpath, sampling_rate)
    yt, index = librosa.effects.trim(y, top_db=60)
    d = librosa.get_duration(yt)
    total_duration += d

print('{:.4f} hours'.format(total_duration / 3600))
