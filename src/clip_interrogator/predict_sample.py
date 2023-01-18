import os
import argparse
from PIL import Image
from clip_interrogator import ClipInterrogator, Config

args = argparse.ArgumentParser()

def parse_args():
    global args
    args.add_argument("--source_dir", type = str, default = ".", required=True)
    args = args.parse_args()

models = [
    "ViT-L-14/openai",
    "ViT-H-14/laion2b_s32b_b79k",
    "xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k",
    "deepdanbooru"
]

class Interrogator:
    def __init__(self):
        self.base_size = 512
        self.ci = ClipInterrogator(Config(
            clip_model_name="ViT-H-14/laion2b_s32b_b79k",
            blip_image_eval_size=512,
        ))

    def interrogate(self, pil_image: Image):
        prompt = self.ci.interrogate(pil_image)

        return prompt

def create_source_list(source_dir):
    source_dir = os.path.abspath(source_dir)
    source_list = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                source_list.append(os.path.join(root, file))
    return source_list


def main():
    parse_args()
    interrogator = Interrogator()
    source_list = create_source_list(args.source_dir)

    for source in source_list:
        image = Image.open(source)

        print("")
        print(source)
        prompt = interrogator.interrogate(image)

        print(prompt)
        print("")

if __name__ == "__main__":
    main()
