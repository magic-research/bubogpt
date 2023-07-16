import os
import logging
import warnings

from bubogpt.common.registry import registry
from bubogpt.datasets.builders.audio_base_dataset_builder import AudioBaseDatasetBuilder
from bubogpt.datasets.datasets.audio_caption import GenericAudioDataset, AudioCaptionDataset


class GenericAudioBuilder(AudioBaseDatasetBuilder):
    train_dataset_cls = GenericAudioDataset

    def _download_ann(self):
        pass

    def _download_aud(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            audio_processor=self.audio_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("bbc")
class BBCBuilder(GenericAudioBuilder):
    DATASET_CONFIG_DICT = {"default": "configs/datasets/bbc/defaults.yaml"}


@registry.register_builder("audioset")
class AudioSetBuilder(GenericAudioBuilder):
    DATASET_CONFIG_DICT = {"default": "configs/datasets/audioset/defaults.yaml"}


@registry.register_builder("soundbible")
class SoundBibleBuilder(GenericAudioBuilder):
    DATASET_CONFIG_DICT = {"default": "configs/datasets/soundbible/defaults.yaml"}


@registry.register_builder("freesound")
class FreeSoundBuilder(GenericAudioBuilder):
    DATASET_CONFIG_DICT = {"default": "configs/datasets/freesound/defaults.yaml"}


@registry.register_builder("clotho_align")
class ClothoAlignBuilderAudio(GenericAudioBuilder):
    train_dataset_cls = AudioCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/clotho/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            audio_processor=self.audio_processors["train"],
            text_processor=self.text_processors["train"],
            audio_root=os.path.join(storage_path, 'all'),
            ann_paths=[os.path.join(storage_path, 'audio_cap.json')],
        )

        return datasets
