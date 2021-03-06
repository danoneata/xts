from abc import ABCMeta, abstractmethod
import pdb
import sys

import librosa  # type: ignore
import numpy as np  # type: ignore
import torch

from scipy.signal import lfilter  # type: ignore

sys.path.insert(0, "tacotron2")
from tacotron2.audio_processing import griffin_lim  # type: ignore
from tacotron2.layers import TacotronSTFT  # type: ignore


class AudioProcessing(metaclass=ABCMeta):
    def __init__(self, sampling_rate, n_mel_channels):
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    @abstractmethod
    def audio_to_mel(self, audio: np.ndarray) -> torch.Tensor:
        """Returns an NumPy array of size seq_len × n_mel_channels"""
        pass

    @abstractmethod
    def mel_to_audio(self, mel: torch.Tensor) -> np.ndarray:
        pass

    def load_audio(self, path: str) -> np.ndarray:
        audio, sr = librosa.core.load(path, self.sampling_rate)
        assert self.sampling_rate == sr
        return audio


class Tacotron(AudioProcessing):
    """Preprocesses audio as in the Tacotron2 code."""

    def __init__(
        self,
        sampling_rate,
        n_mel_channels,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        mel_fmin=0.0,
        mel_fmax=8000.0,
    ):
        super(Tacotron, self).__init__(sampling_rate, n_mel_channels)
        self.taco_stft = TacotronSTFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            sampling_rate=sampling_rate,
            n_mel_channels=n_mel_channels,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
        )

    def audio_to_mel(self, audio):
        audio = torch.tensor(audio)
        audio = audio.unsqueeze(0)
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
            spec_from_mel[:, :, :-1],
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
    def __init__(self, sampling_rate, n_mel_channels, hop_length=None, win_length=None):
        super(DeepConvTTS, self).__init__(sampling_rate, n_mel_channels)
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
            n_mels=self.n_mel_channels,
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
    "deep-conv-tts": lambda s, n: DeepConvTTS(s, n),
    # modules for which the sequence length is a multiple of 3 of the video sequence
    # (works for the GRID dataset and for a sampling rate of 16 kHz)
    "tacotron-3": lambda s, n: Tacotron(s, n, hop_length=212),
    "deep-conv-tts-3": lambda s, n: DeepConvTTS(s, n, hop_length=212),
}
