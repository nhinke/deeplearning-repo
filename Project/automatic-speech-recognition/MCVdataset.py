
import os
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Union
from MCVencoding import EncoderDecoder

import torch
import torchaudio

# https://pytorch.org/audio/stable/_modules/torchaudio/datasets/commonvoice.html#COMMONVOICE


def load_commonvoice_item(line: List[str], header: List[str], path: str, folder_audio: str, ext_audio: str) -> Tuple[torch.Tensor, int, Dict[str, str]]:
    # Each line has the following data:,'c':3
    # client_id, path, sentence, up_votes, down_votes, age, gender, accent

    assert header[1] == "path"
    fileid = line[1]
    filename = os.path.join(path, folder_audio, fileid)
    if not filename.endswith(ext_audio):
        filename += ext_audio

    waveform, sample_rate = torchaudio.load(filename)

    metadata = dict(zip(header, line))

    return waveform, sample_rate, metadata


def resample_audio(signal: torch.Tensor, sample_rate: int, new_sample_rate: int) -> Tuple[torch.Tensor, int]:

    if (sample_rate == new_sample_rate):
        new_signal = signal
    else:
        num_channels = signal.shape[0]
        new_signal = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(signal[:1,:])
        if (num_channels > 1):
            new_ch2 = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(signal[1:,:])
            new_signal = torch.cat([new_signal, new_ch2])

    return new_signal, new_sample_rate


def make_mel_spectrogram_db(signal: torch.Tensor, sample_rate: int, n_mels: int=64, n_fft: int=1024, top_db: int=80) -> torch.Tensor:

    spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(torchaudio.transforms.MelSpectrogram(sample_rate,n_fft=n_fft,n_mels=n_mels)(signal))

    return spec


def perform_spec_augmentation(spec_in: torch.Tensor, prob_augment: float=0.5, max_mask_pct: float=0.1, n_freq_masks: int=1, n_time_masks: int=1) -> torch.Tensor:
    
    n_mels = spec_in.shape[1]
    n_time_steps = spec_in.shape[2]

    spec_out = spec_in
    mask_value = spec_in.mean()

    if (torch.rand(1,1).item() < prob_augment):
        for _ in range(n_freq_masks):
            spec_out = torchaudio.transforms.FrequencyMasking(freq_mask_param=max_mask_pct*n_mels)(spec_out,mask_value)

        for _ in range(n_time_masks):
            spec_out = torchaudio.transforms.TimeMasking(time_mask_param=max_mask_pct*n_time_steps)(spec_out,mask_value)
    
    return spec_out


class SpecAugment(torch.nn.Module):

    def __init__(self, freq_mask=10, time_mask=10):
        super(SpecAugment, self).__init__()
        self.augment = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

    def forward(self,spec):
        return self.augment(spec)


class MozillaCommonVoiceDataset(torch.utils.data.Dataset):
    """Create a Dataset for CommonVoice.

    Args:
        root (str or Path): Path to the directory where the dataset is located.
             (Where the ``tsv`` file is present.)
        tsv (str, optional):
            The name of the tsv file used to construct the metadata, such as
            ``"train.tsv"``, ``"test.tsv"``, ``"dev.tsv"``, ``"invalidated.tsv"``,
            ``"validated.tsv"`` and ``"other.tsv"``. (default: ``"train.tsv"``)
    """

    std_sample_rate = 48000
    min_sig_len_ms = 1500
    max_sig_len_ms = 7000
    min_sig_len = std_sample_rate//1000*min_sig_len_ms
    max_sig_len = std_sample_rate//1000*max_sig_len_ms

    _ext_txt = ".txt"
    _ext_audio = ".mp3"
    _folder_audio = "clips"

    def __init__(self, root: Union[str, Path], tsv: str="train.tsv", augment: bool=False, resample_audio: bool=False) -> None:

        self.augment = augment
        self.resample = resample_audio
        # self.spec_aug = SpecAugment()
        self.EncDec = EncoderDecoder()

        # Get string representation of 'root' in case Path object is passed
        self._path = os.fspath(root)
        self._tsv = os.path.join(self._path, tsv)

        with open(self._tsv, "r") as tsv_:
            walker = csv.reader(tsv_, delimiter="\t")
            self._header = next(walker)
            self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, Dict[str, str]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, Dict[str, str]): ``(waveform, sample_rate, dictionary)``,  where dictionary
            is built from the TSV file with the following keys: ``client_id``, ``path``, ``sentence``,
            ``up_votes``, ``down_votes``, ``age``, ``gender`` and ``accent``.
        """

        # load sample from dataset
        line = self._walker[n]
        try:
            waveform, sample_rate, metadata = load_commonvoice_item(line, self._header, self._path, self._folder_audio, self._ext_audio)
        except Exception:
            # print('Warning: failed to load audio file')
            return self.__getitem__(n-1 if n != 0 else n+1)

        # resample audio signals to all have the same sample rate (unnecessary since all CV audio files already have sample rate of 48000)
        if (self.resample):
            waveform, sample_rate = resample_audio(waveform, sample_rate, self.std_sample_rate)
        
        # add sample rate to metadata
        metadata['sample rate'] = sample_rate
        
        # check if signal length between min and max values
        sig_len = waveform.shape[1]
        if (sig_len < self.min_sig_len or sig_len > self.max_sig_len):
            return self.__getitem__(n-1 if n != 0 else n+1)
        metadata['duration_ms'] = sig_len*1000.0/self.std_sample_rate

        # TODO: time shift?
        # if (self.augment):
        #     print('true')

        # transform signal to spectrogram
        mel_spec = make_mel_spectrogram_db(waveform, sample_rate)

        # perform time and frequency masking on mel spectrogram
        if (self.augment):
            # mel_spec = self.spec_aug(mel_spec)
            # mel_spec = perform_spec_augmentation(mel_spec)
            mel_spec = perform_spec_augmentation(mel_spec, prob_augment=0.75, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        metadata['label'] = self.EncDec.integer_encoding(metadata['sentence'])
        # metadata['label2'] = self.EncDec.integer_decoding(metadata['label'])

        return mel_spec, metadata

    def __len__(self) -> int:
        # return 12*10
        return len(self._walker)
        