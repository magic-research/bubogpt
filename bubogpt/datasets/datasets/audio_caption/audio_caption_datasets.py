import json
import os 
import torchaudio
import random
import tempfile

from torch.utils.data import Dataset, default_collate
import webdataset as wds
from bubogpt.datasets.datasets.base_dataset import BaseDualDataset


class GenericAudioDataset(BaseDualDataset):
    def __init__(self, audio_processor, text_processor, location):
        super().__init__(x_processor=audio_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode(wds.torch_audio, handler=wds.warn_and_continue),
            wds.to_tuple("flac", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.x_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "audio": sample[0],
            # [clips_per_video, channel, mel_bins, time_steps]
            "text_input": self.text_processor(sample[1]["caption"]),
        }


class AudioCaptionDataset(BaseDualDataset):
    def __init__(self, audio_processor, text_processor, audio_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(audio_processor, text_processor, audio_root, ann_paths)

        self.audio_ids = {}
        n = 0
        for ann in self.annotation:
            audio_id = ann["audio_id"]
            if audio_id not in self.audio_ids.keys():
                self.audio_ids[audio_id] = n
                n += 1

        with open("prompts/alignment_audio.txt") as f:
            self.prompts = f.read().splitlines()
        print(f"==> {self.__class__.__name__} using prompts: ", "\n  " + "\n  ".join(self.prompts))

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        audio_file = ann["audio_id"] + ".wav"
        audio_path = os.path.join(self.x_root, audio_file)
        audio = torchaudio.load(audio_path)
        audio = self.x_processor(audio)
        caption = self.text_processor(ann["caption"])

        return {
            "audio": audio,
            "text_input": caption,
            # "audio_id": self.audio_ids[ann["audio_id"]],
            "prompt": random.choice(self.prompts),
        }
