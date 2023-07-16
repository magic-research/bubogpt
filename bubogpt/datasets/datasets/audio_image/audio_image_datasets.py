import os
import random
import json
import torchaudio
from torch.utils.data import Dataset
from PIL import Image
from bubogpt.datasets.datasets.base_dataset import BaseMultiSourceDataset
import webdataset as wds


class AudioLocalizationDataset(BaseMultiSourceDataset):
    def __init__(self, processors, roots, ann_paths):
        super().__init__(processors, roots, ann_paths)

        with open("prompts/alignment_audio_image_region.txt") as f:
            self.prompts = f.read().splitlines()
        print(f"==> {self.__class__.__name__} using prompts: ", "\n  " + "\n  ".join(self.prompts))

    def __getitem__(self, index):
        ann = self.annotation[index]

        audio_file = ann["audio_id"] + ".wav"
        image_file = ann["image_id"] + ".jpg"
        audio_path = os.path.join(self.roots["audio"], audio_file)
        image_path = os.path.join(self.roots["image"], image_file)

        audio = torchaudio.load(audio_path)
        image = Image.open(image_path).convert("RGB")
        audio = self.processors["audio"](audio)
        image = self.processors["image"](image)
        caption = self.processors["text"](ann["caption"])

        return {
            "audio": audio,
            "vision": image,
            "text_input": caption,
            "prompt": random.choice(self.prompts),
        }


class AudioImageNegDataset(Dataset):
    def __init__(self, processors, roots, ann_paths) -> None:
        super().__init__()

        self.processors = processors
        self.roots = roots
        self.ann_paths = ann_paths

        self.img_annotation = []
        for ann_path in ann_paths['image']:
            self.img_annotation.extend(json.load(open(ann_path, "r"))['annotations'])

        self.aud_annotation = []
        for ann_path in ann_paths['audio']:
            self.aud_annotation.extend(json.load(open(ann_path, "r"))['annotations'])

        with open("prompts/alignment_audio_image_neg.txt") as f:
            self.prompts = f.read().splitlines()
        print(f"==> {self.__class__.__name__} using prompts: ", "\n  " + "\n  ".join(self.prompts))

    def __len__(self):
        return len(self.img_annotation)

    def __getitem__(self, index):

        img_ann = self.img_annotation[index]

        img_file = '{}.jpg'.format(img_ann["image_id"])
        image_path = os.path.join(self.roots['image'], img_file)
        image = Image.open(image_path).convert("RGB")
        image = self.processors['image'](image)

        aud_index = random.randint(0, len(self.aud_annotation)-1)
        aud_ann = self.aud_annotation[aud_index]

        audio_file = aud_ann["audio_id"] + ".wav"
        audio_path = os.path.join(self.roots['audio'], audio_file)
        audio = torchaudio.load(audio_path)
        audio = self.processors['audio'](audio)
        prompt = random.choice(self.prompts)
        if "related" in prompt:
            prefix = "They seem unrelated. "
        else:
            prefix = "They seem unrelated. " if random.random() < 0.5 else ""
        caption = self.processors['text'](prefix + img_ann["caption"] + aud_ann["caption"])

        return {
            'audio': audio,
            'vision': image,
            'text_input': caption,
            'prompt': prompt,
        }
