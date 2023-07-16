import os
import torch
import torch.nn as nn
import openai
import random
import contextlib

from constants.constant import COLORS


@contextlib.contextmanager
def auto_proxy():
    use_proxy = "OPENAI_PROXY" in os.environ
    if use_proxy:
        os.environ['http_proxy'] = os.environ["OPENAI_PROXY"]
        os.environ['https_proxy'] = os.environ["OPENAI_PROXY"]

    yield

    if use_proxy:
        os.unsetenv('http_proxy')
        os.unsetenv('https_proxy')


class MatchModule(nn.Module):
    def __init__(self, device='cpu', model="gpt-3.5-turbo"):
        super().__init__()
        self.device = device
        self.model = model
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("Please specify your openai API key with the environment variable OPENAI_API_KEY")
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.examples = [
            (
                "<List>['dog', 'sheepdog', 'grass', 'chase sheepdog', 'field', 'field park', 'grassy', 'corgi', 'brown dog', 'brown', 'park']</List>"
                "<Text>A brown dog running in the grassy field</Text>",
                'brown dog - brown dog\n'
                'grassy field - field'
            ),
            (
                "<List>['man', 'ride', 'bicycle', 'red', 'passenger train', 'track']</List>"
                "<Text>A man riding a bicycle next to a red passenger train on the tracks.</Text>",
                "man - man\n"
                "bicycle - bicycle\n"
                "red passenger train - passenger train\n"
                "tracks - track"
            ),
            (
                "<List>['horse', 'herd', 'dust', 'grassy', 'field']</List>"
                "<Text>The image shows a large herd of wild horses running across a wide, open field . "
                "There are many horses running in different directions, with some running towards the camera "
                "and others running towards the edge of the field. "
                "The horses are brown and white, with some having manes and tails</Text>",
                "herd - herd\n"
                "wild horses - horse\n"
                "field - field"
            ),
            (
                "<List>['man', 'plate platter', 'sandwich', 'tablening table', 'saucer', 'coffee coffee cup', 'coffee', 'bean chip fry', 'chip fry', 'coffee cup', 'bean', 'food', 'table', 'restaurant']</List>"
                "<Text>The image shows a man sitting at a table , surrounded by a large amount of food and drinks . There is a chicken sandwich on the table, as well as a bowl of soup, potato wedges, and several fried potatoes. The man is holding a spoon, which he is expected to use to eat one of the wedges or possibly a piece of the chicken sandwich. The other items on the table, such as drinks and a bowl of soup, appear to be for those accompanying the man at the table. The scene takes place in a dining establishment , likely a restaurant , based on the presence of a spoon and food items on the table, along with a tablecloth and table setting. Additionally, the presence of several chairs and the overall setup suggest this is a formal, sit-down setting rather than a fast food or take-out restaurant. The amount of food on the table suggests that this is a hearty, satisfying meal, providing a range of flavors and textures that satisfy the palate.</Text>",
                "man - man\n"
                "table - table\n"
                "food - food\n"
                "chicken sandwich - sandwich\n"
                "restaurant - restaurant\n"
                "fried potatoes - chip fry\n"
                "drinks - coffee"
            ),
            (
                "<List>['bacon', 'silverware utensil', 'fork', 'coffee', 'table dinning table', 'plate platter', 'beverage', 'napkin', 'bread french toast pan', 'pine cone', 'coffee cup cup mug', 'fruit', 'breakfast food fruit', 'bacon', 'gravy', 'bread pancake']</List>"
                "<Text>The image presents a delicious breakfast setting on a wooden dining table. The main course is a white plate with French toast and bacon . Adding to the meal are a bottle of maple syrup and a cup of coffee , both placed next to the plate. The table is set with a fork , a knife, and a spoon, all arranged neatly around the plate. There are also a few apples scattered across the table, possibly serving as a healthy addition to the meal. Overall, the scene is inviting and warmly lit, making the breakfast look especially appetizing.</Text>",
                "wooden dinning table - table dinning table\n"
                "fork - fork\n"
                "coffee - coffee\n"
                "apples - fruit\n"
                "white plate - plate platter\n"
                "french toast - bread french toast pan\n"
                "bacon - bacon"
            ),
            (
                "<List>['woman', 'canopy', 'man', 'dog pet', 'dog', 'canopy', 'bicycle', 'person', 'leash', "
                "'dog pet', 'leash', 'stall', 'person woman', 'dog pet', 'city street road', 'street scene']</List>"
                "<Text>The image captures a lively street scene with several people walking and riding bikes. "
                "There are two bicycles in the picture, one located in the middle of the scene and the other towards "
                "the right side. Among the people, some are walking close to the bicycles, while others are scattered"
                "throughout the scene. In addition to the bicycles and people, there are four dogs in the picture, "
                "adding to the liveliness of the scene. The dogs are walking around the street, mingling with the "
                "pedestrians and bikers. The street is bustling with activity, as people, bikes, and dogs all "
                "share the space and enjoy the day.</Text>",
                "street scene - street scene\n"
                "the street - city street road\n"
                "bicycles - bicycle\n"
                "four dogs - dog\n"
                "people - person"
            )
        ]
        self.system_prompt = "You are a helpful assistant. Now I will give you a list of entities and give you a " \
                             "paragraph or sentence. " \
                             "you need to first extract the entity given in the text and then" \
                             "find the corresponding entity having similar or identical meanings in the given list. " \
                             "Find all the pairs." \
                             "Are you clear? let us think step by step. " \
                             "The extracted entities must come from the given text and the corresponding entity must " \
                             "come from the given list. " \
                             "If multiple entities can be linked to the same span of text or vice versa, " \
                             "just keep one and do not merge them." \
                             "Here is an example: <List>['dog', 'sheepdog', 'grass', 'chase sheepdog', 'field', " \
                             "'field park', 'grassy', 'corgi', 'brown dog', 'brown', 'park']</List> " \
                             "<Text>A brown dog running in the grassy field</Text>" \
                             "The answer is: brown dog — brown dog \n grassy field — field"

    @torch.no_grad()
    def forward(self, text, entity_state):
        entity_list = list(entity_state['grounding']['local'].keys())
        message = [
            {"role": "system", "content": self.system_prompt},
        ]
        for q, a in self.examples:
            message.append({"role": "user", "content": q})
            message.append({"role": "system", "content": a})
        message.append({
            "role": "user",
            "content": '<List>{}<List><Text>{}</Text>'.format(entity_state['grounding']['local'].keys(), text)
        })

        print('==> Sending request to ChatGPT...')
        with auto_proxy():
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=message
            )
        ans = resp['choices'][0]['message']['content']
        print("===> In the matching module.")
        print('==> Response from ChatGPT received: {}.'.format(ans))
        # print(resp)
        items = ans.split('\n')
        res = []
        match_state = {}
        for i in items:
            if ' - ' not in i:
                continue
            name, ref = i.split(' - ', maxsplit=1)
            name, ref = name.lower(), ref.lower()
            # NOTE: ref may not be contained in the original text, double check later.
            if ref in entity_list:
                color_name = entity_state['grounding']['local'][ref]['color']
            else:
                print('pair {} - {} not found'.format(name, ref))
                # color_name = "grey"
                continue
            match_state[name] = ref
            entity_idx = text.lower().find(name)
            if entity_idx == -1:
                entity_idx = text.lower().find(name.lower())
                ref = name
            if entity_idx == -1:
                continue

            res.append((name, ref, entity_idx, color_name))
        res = sorted(res, key=lambda x: x[2])
        # TODO: Bug to fix
        highlight_output = []
        prev = 0
        color_map = {}

        for i, r in enumerate(res):
            if r[2] < prev:
                continue
            # to avoid one-vs-many alignments
            if r[2] != prev:
                highlight_output.append((text[prev:r[2]], None))
            highlight_output.append((text[r[2]:r[2] + len(r[0])], f'{i + 1}'))
            color_map[f'{i + 1}'] = r[-1]
            prev = r[2] + len(r[0])
        if prev != len(text) - 1:
            highlight_output.append((text[prev:], None))
        print("=======> Highlight Output: ", highlight_output)
        return highlight_output, match_state, color_map


if __name__ == '__main__':
    ner = MatchModule(model='gpt-4')
    print(
        ner('The image shows a resort with a large swimming pool surrounded by lounge chairs and umbrellas. There are several buildings in the background with white walls and blue roofs. There are sand dunes and palm trees in the background indicating that the resort is located in a desert area. The sky is clear and blue with a few fluffy clouds in the distance.'))
