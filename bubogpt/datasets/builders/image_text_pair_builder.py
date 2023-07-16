import os
import logging
import warnings

from bubogpt.common.registry import registry
from bubogpt.datasets.builders.image_base_dataset_builder import ImageBaseDatasetBuilder
from bubogpt.datasets.datasets.image_caption.laion_dataset import LaionDataset
from bubogpt.datasets.datasets.image_caption.cc_sbu_dataset import CCSBUDataset, \
    CCSBUAlignDatasetImageImageCaptionDataset, CCDataset
from bubogpt.datasets.datasets.image_caption.llava_dataset import LlavaInstruct150Dataset

@registry.register_builder("cc_sbu")
class CCSBUBuilderImage(ImageBaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vision_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("laion")
class LaionBuilderImage(ImageBaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vision_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilderImage(ImageBaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDatasetImageImageCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
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
            vision_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets


@registry.register_builder("cc12m")
class CC12MBuilder(ImageBaseDatasetBuilder):
    train_dataset_cls = CCDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc12m/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets
    

@registry.register_builder("llava_instruct150")
class LlavaInstruct150Builder(ImageBaseDatasetBuilder):
    train_dataset_cls = LlavaInstruct150Dataset

    DATASET_CONFIG_DICT = {"default": None}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    
    def build(self):
        self.build_processors()

        datasets = dict()
        split = "train"
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root="/path/to/dataset/COCO_2014",
            ann_paths=[os.path.join("/path/to/dataset/llava/annotations", subset + '.json')
                       for subset in ["complex_reasoning_77k", "conversation_58k", "detail_23k"]],
        )
        return datasets


# from bubogpt.datasets.builders.image_text_pair_builder import LlavaInstruct150Builder

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from itertools import islice

    data_cfg = OmegaConf.create({
        "vis_processor": {"train": {"name": "imagebind_vision_train", "image_size": 224}},
        "text_processor": {"train": {"name": "imagebind_caption"}},
        "data_type": "image",
        })

    builder = LlavaInstruct150Builder(data_cfg)

    datasets = builder.build_datasets()

    datasets["train"].check_existence()

    for sample in islice(datasets["train"], 10):
        print(sample["vision"].shape, sample["prompt"], sample["text_input"])
