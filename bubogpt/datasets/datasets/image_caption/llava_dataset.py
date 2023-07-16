import os
import json
import random
from PIL import Image
import webdataset as wds
from bubogpt.datasets.datasets.base_dataset import BaseDualDataset
from bubogpt.datasets.datasets.image_caption.image_caption_datasets import ImageCaptionDataset


class LlavaInstruct150Dataset(BaseDualDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(x_processor=vis_processor, text_processor=text_processor)
        self.vis_root = vis_root
        self.ann_paths = ann_paths

        self.data_list = data_list = []
        # for split in ["complex_reasoning_77k", "conversation_58k", "detail_23k"]:
        #     with open(os.path.join(vis_root, f'annotations/{split}.json'), 'r') as f:
        #         data_list.extend(json.load(f))
        for ann_path in ann_paths:
            with open(ann_path) as f:
                data_list.extend(json.load(f))

        self.annotation = []
        for item in data_list:
            image_id = item['id']
            conversations = item['conversations']
            for conv_id in range(len(conversations) //2 ):
                question = conversations[2*conv_id]['value']
                answer = conversations[2 * conv_id+1]['value']
                self.annotation.append({'image_id':image_id, 'question':question, 'answer':answer})
        
        # llava prompts
        self.prompts = [
            "<Vision><ModalityHere></Vision> <question>",
            "<Vision><ModalityHere></Vision> Quesion: <question>",
            "<Vision><ModalityHere></Vision> <question> A detail answer to the question is",
            "<Vision><ModalityHere></Vision> Quesion: <question> detail answer:",
            "<Vision><ModalityHere></Vision> Based on the image, respond to this question with a detail answer: <question> Answer:",
            "<Vision><ModalityHere></Vision> Use the provided image to answer the question: <question>",
            "<Vision><ModalityHere></Vision> What is the answer to the following question? <question>",
        ]
        print(f"==> {self.__class__.__name__} using prompts: ", "\n  " + "\n  ".join(self.prompts))
        # self.prompt_template = '###Human: {} ###Assistant: '

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, "train2014/COCO_train2014_{:0>12}.jpg".format(ann["image_id"]))
        image = Image.open(image_path).convert("RGB")
        image = self.x_processor(image)

        question = ann['question']
        question = question.replace('<image>\n', '').replace('\n<image>', '')
        # prompt = self.prompt_template.format(random.choice(self.prompts))
        prompt = random.choice(self.prompts)
        prompt = prompt.replace('<question>', question)

        return {
            "vision": image,
            "prompt": prompt,
            "text_input": ann["answer"],
        }

    def check_existence(self):
        from tqdm import tqdm
        for i in tqdm(range(len(self.data_list))):
            image_id = self.data_list[i]["id"]
            image_path = os.path.join(self.vis_root, "train2014/COCO_train2014_{:0>12}.jpg".format(image_id))
            if not os.path.exists(image_path):
                print(f'Image does not exist: {image_path}')
        print("Checking sucessful!")
