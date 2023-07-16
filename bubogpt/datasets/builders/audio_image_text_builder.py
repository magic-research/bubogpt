import logging
import os
import warnings

from bubogpt.common.registry import registry
from bubogpt.datasets.builders.multimodal_base_dataset_builder import MultimodalBaseDatasetBuilder
from bubogpt.datasets.datasets.audio_image.audio_image_datasets import AudioLocalizationDataset, AudioImageNegDataset


@registry.register_builder("vggss_align")
class VGGSSBuilderAudioImage(MultimodalBaseDatasetBuilder):
    train_dataset_cls = AudioLocalizationDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vggss/align.yaml",
        "3k": "configs/datasets/vggss/align3k.yaml",
        "5k": "configs/datasets/vggss/align5k.yaml",
        "31k": "configs/datasets/vggss/align31k.yaml",
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
        print("Building datasets with: ", self.get_ann_files())

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            processors={**{
                modal: self.processors[modal]["train"] for modal in self.data_type
            }, **{
                "text": self.processors["text"]["train"]
            }},
            roots={
                modal: os.path.join(storage_path, f"{modal}s") for modal in self.data_type
            },
            # ann_paths=[os.path.join(storage_path, 'vggsound_balanced.json')],
            ann_paths=self.get_ann_files(),
        )

        return datasets

    def get_ann_files(self):
        ann_files = self.config.build_info.get("ann_files", ["vggsound_balanced.json"])
        return [os.path.join(self.config.build_info.storage, fname) for fname in ann_files]


@registry.register_builder("aud_img_neg")
class NegBuilderAudioImage(MultimodalBaseDatasetBuilder):
    train_dataset_cls = AudioImageNegDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/aud_img_neg/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        # storage_path = build_info.storage
        storage_path = {
            "image": build_info.image.storage,
            "audio": build_info.audio.storage,
        }
        ann_files = {
            "image": build_info.image.ann_files,
            "audio": build_info.audio.ann_files,
        }
        ann_paths = {
            modal: [os.path.join(storage_path[modal], fname) for fname in ann_files[modal]] for modal in self.data_type
        }

        datasets = dict()

        for path in storage_path.values():
            if not os.path.exists(path):
                warnings.warn("storage path {} does not exist.".format(path))
        print("Building datasets with: ", ann_paths)

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            processors={**{
                modal: self.processors[modal]["train"] for modal in self.data_type
            }, **{
                "text": self.processors["text"]["train"]
            }},
            roots={
                modal: os.path.join(storage_path[modal], f"{modal}") for modal in self.data_type
            },
            ann_paths=ann_paths,
        )

        return datasets
