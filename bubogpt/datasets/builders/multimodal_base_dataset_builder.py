import logging

import torch.distributed as dist

import bubogpt.common.utils as utils
from bubogpt.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from bubogpt.common.registry import registry
from bubogpt.datasets.builders import load_dataset_config
from bubogpt.processors.base_processor import BaseProcessor


class MultimodalBaseDatasetBuilder():
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            # help to create datasets from default config.
            self.config = load_dataset_config(self.default_config_path())
        elif isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

        self.data_type = self.config.data_type.split("_")
        # It will be a list like ["audio", "image"], etc.

        # Add "text" manually here.

        self.processors = {modal: {"train": BaseProcessor(), "eval": BaseProcessor()}
                           for modal in [*self.data_type, "text"]}

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        if is_main_process():
            self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build_processors(self):
        for modal in [*self.data_type, "text"]:
            proc_cfg = self.config.get("{}_processor".format(modal))
            if proc_cfg is not None:
                train_cfg = proc_cfg.get("train")
                eval_cfg = proc_cfg.get("eval")
                self.processors[modal]["train"] = self._build_proc_from_cfg(train_cfg)
                self.processors[modal]["eval"] = self._build_proc_from_cfg(eval_cfg)


    @staticmethod
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )

    @classmethod
    def default_config_path(cls, type="default"):
        return utils.get_abs_path(cls.DATASET_CONFIG_DICT[type])

    def _download_data(self):
        pass
