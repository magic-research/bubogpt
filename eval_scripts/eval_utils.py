import torch
import torchaudio
from PIL import Image
import numpy as np


def load_image(image, image_processor):
    if isinstance(image, str):  # is a image path
        raw_image = Image.open(image).convert('RGB')
        image = image_processor(raw_image).unsqueeze(0)
    elif isinstance(image, Image.Image):
        raw_image = image
        image = image_processor(raw_image).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
    return image


def load_audio(audio, audio_processor):
    if isinstance(audio, str):  # is a audio path
        raw_audio = torchaudio.load(audio)
        audio = audio_processor(raw_audio)
    elif isinstance(audio, tuple):
        sample_rate, raw_waveform = audio
        waveform = raw_waveform / np.iinfo(raw_waveform.dtype).max
        if waveform.ndim == 1:
            waveform = torch.from_numpy(waveform[None, :])
        elif waveform.ndim == 2:
            waveform = torch.from_numpy(waveform).mean(1).unsqueeze(0)
        else:
            raise NotImplementedError  # "No such data!"
        audio = audio_processor((waveform, sample_rate))
    else:
        raise NotImplementedError
    return audio.unsqueeze(0)
