import pickle
import random

import numpy as np
import torch
import torch.utils.data

from config import data_file
from models import layers
from utils import load_wav_to_torch


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, split, hparams):
        with open(data_file, 'rb') as file:
            data = pickle.load(file)

        self.samples = data[split]
        print('loading {} {} samples...'.format(len(self.samples), split))

        self.sampling_rate = hparams.sampling_rate
        # self.max_wav_value = hparams.max_wav_value
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.samples)

    def get_mel_text_pair(self, sample):
        # separate filename and text
        audiopath, text = sample['audiopath'], sample['text']
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        # audio_norm = audio / self.max_wav_value
        audio_norm = audio.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text)
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.samples[index])

    def __len__(self):
        return len(self.samples)


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths


if __name__ == '__main__':
    import config
    from tqdm import tqdm
    from utils import parse_args, sequence_to_text

    args = parse_args()
    collate_fn = TextMelCollate(config.n_frames_per_step)

    train_dataset = TextMelLoader('train', config)
    print('len(train_dataset): ' + str(len(train_dataset)))

    dev_dataset = TextMelLoader('dev', config)
    print('len(dev_dataset): ' + str(len(dev_dataset)))

    text, mel = train_dataset[0]
    print('text: ' + str(text))
    text = sequence_to_text(text.numpy().tolist())
    text = ''.join(text)
    print('text: ' + str(text))
    print('type(mel): ' + str(type(mel)))

    text_lengths = []
    mel_lengths = []

    for data in tqdm(dev_dataset):
        text, mel = data
        text = sequence_to_text(text.numpy().tolist())
        text = ''.join(text)
        mel = mel.numpy()

        # print('text: ' + str(text))
        # print('mel.size: ' + str(mel.size))
        text_lengths.append(len(text))
        mel_lengths.append(mel.size)
        # print('np.mean(mel): ' + str(np.mean(mel)))
        # print('np.max(mel): ' + str(np.max(mel)))
        # print('np.min(mel): ' + str(np.min(mel)))

    print('np.mean(text_lengths): ' + str(np.mean(text_lengths)))
    print('np.mean(mel_lengths): ' + str(np.mean(mel_lengths)))
