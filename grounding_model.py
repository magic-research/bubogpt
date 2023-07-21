import PIL
import numpy as np
import torch
import torch.nn as nn
import torchvision
from yacs.config import CfgNode as CN
from PIL import ImageDraw
from segment_anything import build_sam, SamPredictor
from segment_anything.utils.amg import remove_small_regions
from PIL import ImageDraw, ImageFont

from constants.constant import DARKER_COLOR_MAP, LIGHTER_COLOR_MAP, COLORS
import groundingdino.datasets.transforms as T
# from groundingdino import build_groundingdino
 
from groundingdino.models.GroundingDINO.groundingdino import build_groundingdino
from groundingdino.util.inference import annotate, load_image, predict
from groundingdino.util.utils import clean_state_dict


def load_groundingdino_model(model_config_path, model_checkpoint_path):
    import gc
    args = CN.load_cfg(open(model_config_path, "r"))
    model = build_groundingdino(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print('loading GroundingDINO:', load_res)
    gc.collect()
    _ = model.eval()
    return model


class GroundingModule(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
        groundingdino_checkpoint = "./checkpoints/groundingdino_swint_ogc.pth"
        groundingdino_config_file = "./eval_configs/GroundingDINO_SwinT_OGC.yaml"

        self.grounding_model = load_groundingdino_model(groundingdino_config_file,
                                                        groundingdino_checkpoint).to(device)
        self.grounding_model.eval()

        sam = build_sam(checkpoint=sam_checkpoint).to(device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)

    @torch.no_grad()
    def prompt2mask(self, original_image, prompt, state, box_threshold=0.35, text_threshold=0.25, num_boxes=10):
        def image_transform_grounding(init_image):
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image, _ = transform(init_image, None)  # 3, h, w
            return init_image, image

        image_np = np.array(original_image, dtype=np.uint8)
        prompt = prompt.lower()
        prompt = prompt.strip()
        if not prompt.endswith("."):
            prompt = prompt + "."
        _, image_tensor = image_transform_grounding(original_image)
        print('==> Box grounding with "{}"...'.format(prompt))
        with torch.cuda.amp.autocast(enabled=True):
            boxes, logits, phrases = predict(self.grounding_model,
                                             image_tensor, prompt, box_threshold, text_threshold, device=self.device)
        print(phrases)
        # from PIL import Image, ImageDraw, ImageFont
        H, W = original_image.size[1], original_image.size[0]

        draw_img = original_image.copy()
        draw = ImageDraw.Draw(draw_img)
        color_boxes = []
        color_masks = []
        local_results = [original_image.copy() for _ in range(len(state['entity']))]

        local2entity = {}
        for obj_ind, (box, label) in enumerate(zip(boxes, phrases)):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            # random color
            for i, s in enumerate(state['entity']):
                # print(label.lower(), i[0].lower(), label.lower() == i[0].lower())
                if label.lower() == s[0].lower():
                    local2entity[obj_ind] = i
                    break

            if obj_ind not in local2entity:
                print('Color not found', label)
                color = "grey"  # In grey mode.
                # tuple(np.random.randint(0, 255, size=3).tolist())
            else:
                for i, s in enumerate(state['entity']):
                    # print(label.lower(), i[0].lower(), label.lower() == i[0].lower())
                    if label.lower() == s[0].lower():
                        local2entity[obj_ind] = i
                        break

                if obj_ind not in local2entity:
                    print('Color not found', label)
                    color = tuple(np.random.randint(0, 255, size=3).tolist())
                else:
                    color = state['entity'][local2entity[obj_ind]][3]
            color_boxes.append(color)
            print(color_boxes)
            # draw
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=10)
            # font = ImageFont.load_default()
            font = ImageFont.truetype('InputSans-Regular.ttf', int(H / 512.0 * 30))
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (x0, y0, w + x0, y0 + h)
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), str(label), fill="white", font=font)

            if obj_ind in local2entity:
                local_draw = ImageDraw.Draw(local_results[local2entity[obj_ind]])
                local_draw.rectangle([x0, y0, x1, y1], outline=color, width=10)
                local_draw.rectangle(bbox, fill=color)
                local_draw.text((x0, y0), str(label), fill="white", font=font)

        if boxes.size(0) > 0:
            print('==> Mask grounding...')
            boxes = boxes * torch.Tensor([W, H, W, H])
            boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
            boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]

            self.sam_predictor.set_image(image_np)

            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes, image_np.shape[:2])
            with torch.cuda.amp.autocast(enabled=True):
                masks, _, _ = self.sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes.to(self.device),
                    multimask_output=False,
                )

            # remove small disconnected regions and holes
            fine_masks = []
            for mask in masks.to('cpu').numpy():  # masks: [num_masks, 1, h, w]
                fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])
            masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
            masks = torch.from_numpy(masks)

            num_obj = min(len(logits), num_boxes)
            mask_map = None

            full_img = None
            for obj_ind in range(num_obj):
                # box = boxes[obj_ind]

                m = masks[obj_ind][0]

                if full_img is None:
                    full_img = np.zeros((m.shape[0], m.shape[1], 3))
                    mask_map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
                local_image = np.zeros((m.shape[0], m.shape[1], 3))

                mask_map[m != 0] = obj_ind + 1
                # color_mask = np.random.random((1, 3)).tolist()[0]
                color_mask = np.array(color_boxes[obj_ind]) / 255.0
                full_img[m != 0] = color_mask
                local_image[m != 0] = color_mask
                # if local_results[local2entity[obj_ind]] is not None:
                #     local_image[m == 0] = np.asarray(local_results[local2entity[obj_ind]])[m == 0]
                local_image = (local_image * 255).astype(np.uint8)
                local_image = PIL.Image.fromarray(local_image)
                if local_results[local2entity[obj_ind]] is not None:
                    local_results[local2entity[obj_ind]] = PIL.Image.blend(local_results[local2entity[obj_ind]],
                                                                           local_image, 0.5)
            full_img = (full_img * 255).astype(np.uint8)
            full_img = PIL.Image.fromarray(full_img)
            draw_img = PIL.Image.blend(draw_img, full_img, 0.5)

        return draw_img, local_results

    # def draw_text(self, entity_state, entity, text):
    #     local_img = entity_state['grounding']['local'][entity]['image'].copy()
    #     H, W = local_img.width, local_img.height
    #     font = ImageFont.truetype('InputSans-Regular.ttf', int(min(H, W) / 512.0 * 30))
    #
    #     for x0, y0 in entity_state['grounding']['local'][entity]['text_positions']:
    #         color = entity_state['grounding']['local'][entity]['color']
    #         local_draw = ImageDraw.Draw(local_img)
    #         if hasattr(font, "getbbox"):
    #             bbox = local_draw.textbbox((x0, y0), str(text), font)
    #         else:
    #             w, h = local_draw.textsize(str(text), font)
    #             bbox = (x0, y0, w + x0, y0 + h)
    #
    #         local_draw.rectangle(bbox, fill=DARKER_COLOR_MAP[color])
    #         local_draw.text((x0, y0), str(text), fill="white", font=font)
    #     return local_img

    def draw(self, original_image, entity_state, item=None):
        original_image = original_image.copy()
        W, H = original_image.width, original_image.height
        font = ImageFont.truetype('InputSans-Regular.ttf', int(min(H, W) / 512.0 * 30))
        local_image = np.zeros((H, W, 3))
        local_mask = np.zeros((H, W), dtype=bool)

        def draw_item(img, item):
            nonlocal local_image, local_mask
            entity = entity_state['match_state'][item]
            ei = entity_state['grounding']['local'][entity]
            color = ei['color']
            local_draw = ImageDraw.Draw(img)
            for x0, y0, x1, y1 in ei['entity_positions']:
                local_draw.rectangle([x0, y0, x1, y1], outline=DARKER_COLOR_MAP[color],
                                     width=int(min(H, W) / 512.0 * 10))
            for x0, y0 in ei['text_positions']:
                if hasattr(font, "getbbox"):
                    bbox = local_draw.textbbox((x0, y0), str(item), font)
                else:
                    w, h = local_draw.textsize(str(item), font)
                    bbox = (x0, y0, w + x0, y0 + h)

                local_draw.rectangle(bbox, fill=DARKER_COLOR_MAP[color])
                local_draw.text((x0, y0), str(item), fill="white", font=font)
            for m in ei['masks']:
                local_image[m != 0] = np.array(LIGHTER_COLOR_MAP[color]) / 255.0
                local_mask = np.logical_or(local_mask, m)
            # local_image = (local_image * 255).astype(np.uint8)
            # local_image = PIL.Image.fromarray(local_image)
            # img = PIL.Image.blend(img, local_image, 0.5)
            return img

        if item is None:
            for item in entity_state['match_state'].keys():
                original_image = draw_item(original_image, item)
        else:
            original_image = draw_item(original_image, item)
        local_image[local_mask == 0] = (np.array(original_image) / 255.0)[local_mask == 0]
        local_image = (local_image * 255).astype(np.uint8)
        local_image = PIL.Image.fromarray(local_image)

        original_image = PIL.Image.blend(original_image, local_image, 0.5)
        return original_image

    @torch.no_grad()
    def prompt2mask2(self, original_image, prompt, state, box_threshold=0.25,
                     text_threshold=0.2, iou_threshold=0.5, num_boxes=10):
        def image_transform_grounding(init_image):
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image, _ = transform(init_image, None)  # 3, h, w
            return init_image, image

        image_np = np.array(original_image, dtype=np.uint8)
        prompt = prompt.lower()
        prompt = prompt.strip()
        if not prompt.endswith("."):
            prompt = prompt + "."
        _, image_tensor = image_transform_grounding(original_image)
        print('==> Box grounding with "{}"...'.format(prompt))
        with torch.cuda.amp.autocast(enabled=True):
            boxes, logits, phrases = predict(self.grounding_model,
                                             image_tensor, prompt, box_threshold, text_threshold, device=self.device)
        print('==> Box grounding results {}...'.format(phrases))

        # boxes_filt = boxes.cpu()
        # # use NMS to handle overlapped boxes
        # print(f"==> Before NMS: {boxes_filt.shape[0]} boxes")
        # nms_idx = torchvision.ops.nms(boxes_filt, logits, iou_threshold).numpy().tolist()
        # boxes_filt = boxes_filt[nms_idx]
        # phrases = [phrases[idx] for idx in nms_idx]
        # print(f"==> After NMS: {boxes_filt.shape[0]} boxes")
        # boxes = boxes_filt

        # from PIL import Image, ImageDraw, ImageFont
        H, W = original_image.size[1], original_image.size[0]

        draw_img = original_image.copy()
        draw = ImageDraw.Draw(draw_img)
        color_boxes = []
        color_masks = []

        entity_dict = {}
        for obj_ind, (box, label) in enumerate(zip(boxes, phrases)):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            if label not in entity_dict:
                entity_dict[label] = {
                    'color': COLORS[len(entity_dict) % (len(COLORS) - 1)],
                    # 'image': original_image.copy(),
                    'text_positions': [],
                    'entity_positions': [],
                    'masks': []
                }
            color = entity_dict[label]['color']

            color_boxes.append(DARKER_COLOR_MAP[color])
            color_masks.append(LIGHTER_COLOR_MAP[color])

            # draw
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            draw.rectangle([x0, y0, x1, y1], outline=DARKER_COLOR_MAP[color], width=10)
            font = ImageFont.truetype('InputSans-Regular.ttf', int(min(H, W) / 512.0 * 30))
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (x0, y0, w + x0, y0 + h)

            draw.rectangle(bbox, fill=DARKER_COLOR_MAP[color])
            draw.text((x0, y0), str(label), fill="white", font=font)

            # local_img = entity_dict[label]['image']
            # local_draw = ImageDraw.Draw(local_img)
            # local_draw.rectangle([x0, y0, x1, y1], outline=DARKER_COLOR_MAP[color], width=10)
            entity_dict[label]['text_positions'].append((x0, y0))
            entity_dict[label]['entity_positions'].append((x0, y0, x1, y1))
            # local_draw.rectangle(bbox, fill=DARKER_COLOR_MAP[color])
            # local_draw.text((x0, y0), str(label), fill="white", font=font)

        if boxes.size(0) > 0:
            print('==> Mask grounding...')
            boxes = boxes * torch.Tensor([W, H, W, H])
            boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
            boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]

            self.sam_predictor.set_image(image_np)

            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes,
                                                                               image_np.shape[:2]).to(self.device)
            with torch.cuda.amp.autocast(enabled=True):
                masks, _, _ = self.sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes.to(self.device),
                    multimask_output=False,
                )

            # remove small disconnected regions and holes
            fine_masks = []
            for mask in masks.to('cpu').numpy():  # masks: [num_masks, 1, h, w]
                fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])
            masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
            masks = torch.from_numpy(masks)

            mask_map = None

            full_img = None
            for obj_ind, (box, label) in enumerate(zip(boxes, phrases)):

                m = masks[obj_ind][0]

                if full_img is None:
                    full_img = np.zeros((m.shape[0], m.shape[1], 3))
                    mask_map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
                # local_image = np.zeros((m.shape[0], m.shape[1], 3))

                mask_map[m != 0] = obj_ind + 1
                color_mask = np.array(color_masks[obj_ind]) / 255.0

                full_img[m != 0] = color_mask

                entity_dict[label]['masks'].append(m)
                # local_image[m != 0] = color_mask
                # local_image[m == 0] = (np.array(entity_dict[label]['image']) / 255.0)[m == 0]
                #
                # local_image = (local_image * 255).astype(np.uint8)
                # local_image = PIL.Image.fromarray(local_image)
                # entity_dict[label]['image'] = PIL.Image.blend(entity_dict[label]['image'], local_image, 0.5)

            full_img = (full_img * 255).astype(np.uint8)
            full_img = PIL.Image.fromarray(full_img)
            draw_img = PIL.Image.blend(draw_img, full_img, 0.5)
        print('==> Entity list: {}'.format(list(entity_dict.keys())))
        return draw_img, entity_dict
