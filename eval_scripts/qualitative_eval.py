import argparse
import os
import random
# import sys
# import os
#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from constants.constant import LIGHTER_COLOR_MAP_HEX
# NOTE: Must import LlamaTokenizer before `bubogpt.common.config`
# otherwise, it will cause seg fault when `llama_tokenizer.decode` is called

from grounding_model import GroundingModule
from match import MatchModule
from bubogpt.common.config import Config
from bubogpt.common.dist_utils import get_rank
from bubogpt.common.registry import registry
from eval_scripts.conversation import Chat, CONV_X, DummyChat
# NOTE&TODO: put this before bubogpt import will cause circular import
# possibly because `imagebind` imports `bubogpt` and `bubogpt` also imports `imagebind`
from imagebind.models.image_bind import ModalityType
# from ner import NERModule
from tagging_model import TaggingModule



def parse_args():
    parser = argparse.ArgumentParser(description="Qualitative")
    parser.add_argument("--cfg-path", help="path to configuration file.")
    parser.add_argument("--dummy", action="store_true", help="Debug Mode")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    parser.add_argument("--ground-all", action="store_true")
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()

assert args.dummy or (args.cfg_path is not None), "Invalid Config! Set --dummy or configurate the cfg_path!"

if not args.dummy:
    cfg = Config(args)

    # Create processors
    vis_processor_cfg = cfg.datasets_cfg.default.vis_processor.eval
    aud_processor_cfg = cfg.datasets_cfg.default.audio_processor.eval
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    aud_processor = registry.get_processor_class(aud_processor_cfg.name).from_config(aud_processor_cfg)
    processors = {ModalityType.VISION: vis_processor, ModalityType.AUDIO: aud_processor}

    # Create model
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    chat = Chat(model, processors, device='cuda:{}'.format(args.gpu_id))
else:
    model = None
    chat = DummyChat()

match = MatchModule(model='gpt-4')
tagging_module = TaggingModule(device='cuda:{}'.format(args.gpu_id))
grounding_dino = GroundingModule(device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, emb_list):
    if chat_state is not None:
        chat_state.messages = []
    if emb_list is not None:
        emb_list = []
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=False), \
           gr.update(value=None, interactive=True), \
           gr.update(placeholder='Please upload your image/audio first', interactive=False), \
           gr.update(value=None), \
           gr.update(value="Upload & Start Chat", interactive=True), \
           chat_state, emb_list, gr.update(value={})


def upload_x(gr_img, gr_aud, chat_state):
    if gr_img is None and gr_aud is None:
        return None, None, None, gr.update(interactive=True), chat_state, None, {}
    chat_state = CONV_X.copy()
    emb_list = []
    if gr_img is not None:
        chat.upload_img(gr_img, chat_state, emb_list)
        state = {
            'tags': tagging_module(gr_img)
        }
        # print(state)
    else:
        state = {}
    if gr_aud is not None:
        chat.upload_aud(gr_aud, chat_state, emb_list)
    return gr.update(interactive=False), gr.update(interactive=False), \
           gr.update(interactive=True, placeholder='Type and press Enter'), \
           gr.update(value="Start Chatting", interactive=False), \
           chat_state, emb_list, state


def gradio_ask(user_message, chatbot, chat_state, text_output, last_answer):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state, \
               gr.update(value=None, color_map=None, show_legend=False), gr.update(value=None)
    if last_answer is not None:
        chatbot[-1][1] = last_answer
    chat.ask(user_message, chat_state)
    if text_output is not None:
        os.makedirs('results', exist_ok=True)
        # print("****** Text output is:", text_output)
        chatbot[-1][1] = ''.join(map(lambda x: x[0], text_output))
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state, gr.update(value=None, color_map=None, show_legend=False), gr.update(value=None)


def gradio_answer(image, chatbot, chat_state, emb_list, num_beams, temperature, entity_state):
    llm_message = chat.answer(conversation=chat_state,
                              emb_list=emb_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    if image is not None:
        # new_entity_state = entity_state.value()
        # new_entity_state.update({"answer": llm_message})
        entity_state["answer"] = llm_message
        rich_text, match_state, color_map = match(llm_message, entity_state)
        print("Original Color Map: ", color_map)
        color_map = {key: LIGHTER_COLOR_MAP_HEX[color_map[key]] for key in color_map}
        print("Modified Color Map: ", color_map)
        chatbot[-1][1] = "The answer can be found in the textbox below and I'm trying my best to highlight the " \
                         "corresponding region on the image."
        # new_entity_state.update({"match_state": match_state})
        entity_state['match_state'] = match_state  # item_id -> local_id
        new_grounded_image = grounding_dino.draw(image, entity_state)
        show_legend = bool(match_state)
        print('gradio_answer ==> current state: ', entity_state)

        # if args.ground_all:
        #     ground_img, local_results = grounding_dino.prompt2mask(image,
        #                                                            '.'.join(map(lambda x: x, state['entity'])),
        #                                                            state=state)
        # else:
        #     ground_img = None
        return chatbot, chat_state, emb_list, \
            gr.update(value=rich_text, color_map=color_map, show_legend=show_legend), \
            gr.update(value=entity_state), \
            gr.update(value=llm_message), gr.update(value=new_grounded_image)
    else:
        chatbot[-1][1] = llm_message
        return chatbot, chat_state, emb_list, \
            gr.update(value=None), \
            entity_state, \
            gr.update(value=None), gr.update(value=None)

def grounding_fn(image, chatbot, entity_state):
    # print("Grounding fn: ", entity_state)
    if image and entity_state:
        ground_img, local_results = grounding_dino.prompt2mask2(
            image, ','.join(map(lambda x: x, entity_state['tags'])), state=entity_state
        )
        entity_state['grounding'] = {
            'full': ground_img,
            'local': local_results
        }
        print('grounding_fn ==> current state: ', entity_state)
        return chatbot, gr.update(value=ground_img, interactive=False), entity_state
    return chatbot, gr.update(value=None, interactive=False), entity_state


def select_fn(image, ground_img, entity_state, evt: gr.SelectData):
    if image is None:
        return gr.update(value=None, interactive=False)
    item, label = evt.value[0], evt.value[1]

    if label is None:
        return ground_img
    print('select_fn ==> current state: ', entity_state)
    if 'grounding' not in entity_state:
        ground_img, local_results = grounding_dino.prompt2mask2(image,
                                                                ','.join(map(lambda x: x[0], entity_state['tags'])),
                                                                state=entity_state)
        entity_state['grounding'] = {
            'full': ground_img,
            'local': local_results
        }
    # local_img = entity_state['grounding']['local'][entity]['image']
    # print("DEBUG INFO: ", entity_state)
    local_img = grounding_dino.draw(image, entity_state, item.lower())
    return gr.update(value=local_img, interactive=False)


title = """<h1 align="center">Demo of BuboGPT</h1>"""
description = """<h3>This is the demo of BuboGPT. Upload and start chatting!</h3>"""
# article = """<p><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://raw.githubusercontent.com/Vision-CAIR/MiniGPT-4/main/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>
# """

# TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    # gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            grounded_image = gr.Image(type="pil", interactive=False)
            audio = gr.Audio()
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")

            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers",
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State()
            last_answer = gr.State()
            entity_state = gr.State(value={})
            emb_list = gr.State()
            chatbot = gr.Chatbot(label='BuboGPT')
            text_output = gr.HighlightedText(value=None, label="Response", show_legend=False)
            text_input = gr.Textbox(label='User', placeholder='Please upload your image/audio first', interactive=False)

    upload_button.click(
        upload_x, [image, audio, chat_state],
        [image, audio, text_input, upload_button, chat_state, emb_list, entity_state]).then(
        grounding_fn,
        [image, chatbot, entity_state],
        [chatbot, grounded_image, entity_state]
    )

    text_input.submit(gradio_ask,
                      [text_input, chatbot, chat_state, text_output, last_answer],
                      [text_input, chatbot, chat_state, text_output, last_answer]
                      ).then(
        gradio_answer,
        [image, chatbot, chat_state, emb_list, num_beams, temperature, entity_state],
        [chatbot, chat_state, emb_list, text_output, entity_state, last_answer, grounded_image]
    )

    clear.click(gradio_reset,
                [chat_state, emb_list],
                [chatbot, image, grounded_image, audio, text_input, text_output,
                 upload_button, chat_state, emb_list, entity_state],
                queue=False)

    text_output.select(
        select_fn,
        [image, grounded_image, entity_state],
        [grounded_image]
    )

demo.launch(share=True, enable_queue=True)
