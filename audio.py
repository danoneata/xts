from abc import ABCMeta, abstractmethod
import pdb
import sys

import librosa
import numpy as np
import torch

from scipy.signal import lfilter

from hparams import hparams

sys.path.insert(0, "tacotron2")
from tacotron2.audio_processing import griffin_lim
from tacotron2.layers import TacotronSTFT


class AudioProcessing(metaclass=ABCMeta):
    @abstractmethod
    def audio_to_mel(self, audio: np.ndarray) -> torch.tensor:
        """Returns an NumPy array of size seq_len × n_mel_channels"""
        pass

    @abstractmethod
    def mel_to_audio(self, mel: torch.tensor) -> np.ndarray:
        pass

    def load_audio(self, path: str) -> np.ndarray:
        audio, sr = librosa.core.load(path, self.sampling_rate)
        assert self.sampling_rate == sr
        return audio


class Tacotron(AudioProcessing):
    """Preprocesses audio as in the Tacotron2 code."""

    def __init__(
        self,
        sampling_rate=22_050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        mel_fmin=0.0,
        mel_fmax=8000.0,
    ):

        N_MEL_CHANNELS = hparams.n_mel_channels
        self.sampling_rate = sampling_rate
        self.taco_stft = TacotronSTFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            sampling_rate=sampling_rate,
            n_mel_channels=N_MEL_CHANNELS,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
        )

    def audio_to_mel(self, audio):
        audio = torch.tensor(audio)
        audio = audio.unsqueeze(0)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec = self.taco_stft.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0)
        return melspec.T

    def mel_to_audio(self, mel):
        # TODO make it work in batch mode
        mel = mel.unsqueeze(0)
        mel_decompress = self.taco_stft.spectral_de_normalize(mel)
        mel_decompress = mel_decompress.transpose(1, 2).data.cpu()

        spec_from_mel_scaling = 1000

        spec_from_mel = torch.mm(mel_decompress[0], self.taco_stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
        spec_from_mel = spec_from_mel * spec_from_mel_scaling

        GRIFFIN_ITERS = 60
        audio = griffin_lim(
            torch.autograd.Variable(spec_from_mel[:, :, :-1]),
            self.taco_stft.stft_fn,
            GRIFFIN_ITERS,
        )
        audio = audio.squeeze()
        audio = audio.cpu().numpy()

        return audio


class DeepConvTTS(AudioProcessing):
    """Preprocesses audio as in the Deep Convolutional TTS code:
    https://github.com/Kyubyong/dc_tts
    
    """
    def __init__(self, sampling_rate, hop_length=None, win_length=None):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length or int(sampling_rate * 0.0125)
        self.win_length = win_length or int(sampling_rate * 0.05)
        self.n_fft = 1024
        self.preemphasis = 0.97
        self.ref_db = 20
        self.max_db = 100

    def audio_to_mel(self, audio):
        audio = np.append(audio[0], audio[1:] - self.preemphasis * audio[:-1])
        mel = librosa.feature.melspectrogram(
            audio,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=hparams.n_mel_channels,
            win_length=self.win_length,
        )
        mel_abs = np.abs(mel)
        mel_db = librosa.power_to_db(mel_abs, ref=self.max_db)
        mag = np.clip((mel_db - self.ref_db + self.max_db) / self.max_db, 1e-8, 1)
        return torch.tensor(mag.T)

    def mel_to_audio(self, mel):
        mag = mel.T.numpy()
        mel_db = np.clip(mag, 0, 1) * self.max_db - self.max_db + self.ref_db
        mel_abs = librosa.db_to_power(mel_db, ref=self.max_db)
        audio = librosa.feature.inverse.mel_to_audio(
            mel_abs, 
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        audio = lfilter([1], [1, -self.preemphasis], audio)
        audio, _ = librosa.effects.trim(audio)
        return audio


AUDIO_PROCESSING = {
    "deep-conv-tts": lambda sr: DeepConvTTS(sampling_rate=sr),
    # modules for which the sequence length is a multiple of 3 of the video sequence
    # (works for the GRID dataset and for a sampling rate of 16 kHz)
    "tacotron-3": lambda sr: Tacotron(sampling_rate=sr, hop_length=212),
    "deep-conv-tts-3": lambda sr: DeepConvTTS(sampling_rate=sr, hop_length=212),
}
