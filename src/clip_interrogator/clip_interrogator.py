import hashlib
import math
import numpy as np
import open_clip
import os
import pickle
import time
import torch
import hashlib
import requests
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from typing import List
from .blip import blip_decoder, BLIP_Decoder

@dataclass 
class Config:
    # models can optionally be passed in directly
    blip_model: BLIP_Decoder = None
    clip_model = None
    clip_preprocess = None

    # blip settings
    blip_image_eval_size: int = 384
    blip_max_length: int = 32
    blip_model_url: str = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
    blip_num_beams: int = 8
    blip_offload: bool = False

    # clip settings
    clip_model_name: str = 'ViT-L-14/openai'
    clip_model_path: str = None

    # interrogator settings
    cache_path: str = 'cache'
    chunk_size: int = 2048
    data_path: str = os.path.join(os.path.dirname(__file__), 'data')
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    flavor_intermediate_count: int = 2048
    quiet: bool = False # when quiet progress bars are not shown


class ClipInterrogator():
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        self.load_blip_model()
        self.load_clip_model()


    def load_blip_model(self):
        if self.config.blip_model is None:
            self.cache_model_path = os.path.join(self.config.cache_path, 'model_large_caption.pth')
            if os.path.exists(self.cache_model_path):
                model_path = self.cache_model_path
                with open(self.cache_model_path, 'rb') as f:
                    model_hash = hashlib.md5(f.read()).hexdigest()
                    if not model_hash == 'b78e0b7488c83ba75d58f93f79e885b6':
                        self.download_blip_model()
                        model_path = self.cache_model_path
            else:
                self.download_blip_model()
#                model_path = self.config.blip_model_url
                model_path = self.cache_model_path
            blip_model = blip_decoder(
                pretrained=model_path, 
                image_size=self.config.blip_image_eval_size, 
                vit='large'
            )
            blip_model.eval()
            blip_model = blip_model.to(self.config.device)
            self.blip_model = blip_model
        else:
            self.blip_model = self.config.blip_model

        return


    def download_blip_model(self):
        # download new model
        url = self.config.blip_model_url
        file_size = int(requests.head(url).headers["content-length"])
        r = requests.get(url, allow_redirects=True, stream=True)
        pbar = tqdm(total=file_size, unit="B", unit_scale=True)

        with open(self.cache_model_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
                pbar.update(len(chunk))
            pbar.close()

        return

    def load_clip_model(self):
        if self.config.clip_model is None:

            clip_model_name, clip_model_pretrained_name = self.config.clip_model_name.split('/', 2)
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                clip_model_name, 
                pretrained=clip_model_pretrained_name, 
                precision='fp16' if self.config.device == 'cuda' else 'fp32',
                device=self.config.device,
                jit=False,
                cache_dir=self.config.clip_model_path
            )
            self.clip_model.to(self.config.device).eval()
        else:
            self.clip_model = self.config.clip_model
            self.clip_preprocess = self.config.clip_preprocess
        self.tokenize = open_clip.get_tokenizer(clip_model_name)

        return


    def prepare_labels(self):
        sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribble', 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central', 'PornPics', 'sex.com']
        trending_list = [site for site in sites]
        trending_list.extend(["trending on "+site for site in sites])
        trending_list.extend(["featured on "+site for site in sites])
        trending_list.extend([site+" contest winner" for site in sites])

        raw_artists = _load_list(self.config.data_path, 'artists.txt')
        artists = [f"by {a}" for a in raw_artists]
        artists.extend([f"inspired by {a}" for a in raw_artists])

        flavors = _load_list(self.config.data_path, 'flavors.txt')

        self.artists = LabelTable(artists, "artists", self.clip_model, self.tokenize, self.config)
        self.flavors = LabelTable(flavors, "flavors", self.clip_model, self.tokenize, self.config)
        self.mediums = LabelTable(_load_list(self.config.data_path, 'mediums.txt'), "mediums", self.clip_model, self.tokenize, self.config)
        self.movements = LabelTable(_load_list(self.config.data_path, 'movements.txt'), "movements", self.clip_model, self.tokenize, self.config)
        self.trendings = LabelTable(trending_list, "trendings", self.clip_model, self.tokenize, self.config)

        return

    def generate_caption(self, pil_image: Image) -> str:
        if self.config.blip_offload:
            self.blip_model = self.blip_model.to(self.device)
        size = self.config.blip_image_eval_size
        gpu_image = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            caption = self.blip_model.generate(
                gpu_image, 
                sample=False, 
                num_beams=self.config.blip_num_beams, 
                max_length=self.config.blip_max_length, 
                min_length=5
            )
        if self.config.blip_offload:
            self.blip_model = self.blip_model.to("cpu")
        return caption[0]

    def image_to_features(self, image: Image) -> torch.Tensor:
        try:
            images = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = self.clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features
        except Exception as e:
            raise e

    def interragate_score_list(self, image: Image, options: list = None) -> str:
        try:
            image_features = self.image_to_features(image)
            if options:
                result = {}
                with torch.no_grad(), torch.cuda.amp.autocast():
                    for option in options:
                        text_tokens = self.tokenize([option]).to(self.device)
                        text_features = self.clip_model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        similarity = text_features @ image_features.T
                        result[option] = similarity[0][0].item()
            else:
                raise Exception("No options provided")
            torch.cuda.empty_cache()
            return result
        except Exception as e:
            torch.cuda.empty_cache()
            raise e



    def interragate_score(self, image: Image, text: str) -> str:
        try:
            image_features = self.image_to_features(image)
            sim = self.similarity(image_features, text)
            torch.cuda.empty_cache()
            return sim

        except Exception as e:
            torch.cuda.empty_cache()
            raise e

    def interrogate_one(self, image: Image, path: str = None, options: list = None) -> str:
        try:
            image_features = self.image_to_features(image)
            if path:
                seed = _load_list(os.path.dirname(path), os.path.basename(path))
                seed_labels = LabelTable(seed, os.path.basename(path), self.clip_model, self.tokenize, self.config)
            elif options:
                seed_labels = LabelTable(options, "list", self.clip_model, self.tokenize, self.config)
            else:
                raise Exception("No seed or list provided.")
            top = seed_labels.rank(image_features, 1)[0]
            torch.cuda.empty_cache()
            return _truncate_to_fit(top, self.tokenize)
        except Exception as e:
            torch.cuda.empty_cache()
            raise e

    def interrogate_flavors(self, image: Image, path: str = None, options: list = None, max_flavors: int = 32) -> str:
        try:
            image_features = self.image_to_features(image)
            if path:
                seed_labels = _load_list(os.path.dirname(path), os.path.basename(path))
                self.flavors_reduced = LabelTable(seed_labels, os.path.basename(path), self.clip_model, self.tokenize, self.config)
            elif options:
                self.flavors_reduced = LabelTable(options, "list", self.clip_model, self.tokenize, self.config)
            else:
                seed_labels = _load_list(os.path.join(self.config.data_path, 'flavors_reduced.txt'))
                self.flavors_reduced = LabelTable(seed_labels, "flavors_reduced", self.clip_model, self.tokenize, self.config)
            tops = self.flavors_reduced.rank(image_features, max_flavors)
            torch.cuda.empty_cache()
            return ", ".join(tops)
        except Exception as e:
            torch.cuda.empty_cache()
            raise e

    def interrogate_classic(self, image: Image, max_flavors: int=3) -> str:
        caption = self.generate_caption(image)
        image_features = self.image_to_features(image)

        medium = self.mediums.rank(image_features, 1)[0]
        artist = self.artists.rank(image_features, 1)[0]
        trending = self.trendings.rank(image_features, 1)[0]
        movement = self.movements.rank(image_features, 1)[0]
        flaves = ", ".join(self.flavors.rank(image_features, max_flavors))

        if caption.startswith(medium):
            prompt = f"{caption} {artist}, {trending}, {movement}, {flaves}"
        else:
            prompt = f"{caption}, {medium} {artist}, {trending}, {movement}, {flaves}"

        return _truncate_to_fit(prompt, self.tokenize)

    def interrogate_fast(self, image: Image, max_flavors: int = 32) -> str:
        caption = self.generate_caption(image)
        image_features = self.image_to_features(image)
        merged = _merge_tables([self.artists, self.flavors, self.mediums, self.movements, self.trendings], self.config)
        tops = merged.rank(image_features, max_flavors)
        torch.cuda.empty_cache()
        return _truncate_to_fit(caption + ", " + ", ".join(tops), self.tokenize)


    def interrogate(self, image: Image, max_flavors: int=32) -> str:
        caption = self.generate_caption(image)
        image_features = self.image_to_features(image)

        flaves = self.flavors.rank(image_features, self.config.flavor_intermediate_count)
        best_medium = self.mediums.rank(image_features, 1)[0]
        best_artist = self.artists.rank(image_features, 1)[0]
        best_trending = self.trendings.rank(image_features, 1)[0]
        best_movement = self.movements.rank(image_features, 1)[0]

        best_prompt = caption
        best_sim = self.similarity(image_features, best_prompt)

        def check(addition: str) -> bool:
            nonlocal best_prompt, best_sim
            prompt = best_prompt + ", " + addition
            sim = self.similarity(image_features, prompt)
            if sim > best_sim:
                best_sim = sim
                best_prompt = prompt
                return True
            return False

        def check_multi_batch(opts: List[str]):
            nonlocal best_prompt, best_sim
            prompts = []
            for i in range(2**len(opts)):
                prompt = best_prompt
                for bit in range(len(opts)):
                    if i & (1 << bit):
                        prompt += ", " + opts[bit]
                prompts.append(prompt)

            t = LabelTable(prompts, None, self.clip_model, self.tokenize, self.config)
            best_prompt = t.rank(image_features, 1)[0]
            best_sim = self.similarity(image_features, best_prompt)

        check_multi_batch([best_medium, best_artist, best_trending, best_movement])

        extended_flavors = set(flaves)
        for _ in tqdm(range(max_flavors), desc="Flavor chain", disable=self.config.quiet):
            best = self.rank_top(image_features, [f"{best_prompt}, {f}" for f in extended_flavors])
            flave = best[len(best_prompt)+2:]
            if not check(flave):
                break
            if _prompt_at_max_len(best_prompt, self.tokenize):
                break
            extended_flavors.remove(flave)

        torch.cuda.empty_cache()
        return best_prompt

    def rank_top(self, image_features: torch.Tensor, text_array: List[str]) -> str:
        text_tokens = self.tokenize([text for text in text_array]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return text_array[similarity.argmax().item()]

    def similarity(self, image_features: torch.Tensor, text: str) -> float:
        text_tokens = self.tokenize([text]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return similarity[0][0].item()


class LabelTable():
    def __init__(self, labels:List[str], desc:str, clip_model, tokenize, config: Config):
        self.chunk_size = config.chunk_size
        self.config = config
        self.device = config.device
        self.embeds = []
        self.labels = labels
        self.tokenize = tokenize

        hash = hashlib.sha256(",".join(labels).encode()).hexdigest()

        cache_filepath = None
        if config.cache_path is not None and desc is not None:
            os.makedirs(config.cache_path, exist_ok=True)
            sanitized_name = config.clip_model_name.replace('/', '_').replace('@', '_')
            cache_filepath = os.path.join(config.cache_path, f"{sanitized_name}_{desc}.pkl")
            if desc is not None and os.path.exists(cache_filepath):
                with open(cache_filepath, 'rb') as f:
                    data = pickle.load(f)
                    if data.get('hash') == hash:
                        self.labels = data['labels']
                        self.embeds = data['embeds']

        if len(self.labels) != len(self.embeds):
            self.embeds = []
            chunks = np.array_split(self.labels, max(1, len(self.labels)/config.chunk_size))
            for chunk in tqdm(chunks, desc=f"Preprocessing {desc}" if desc else None, disable=self.config.quiet):
                text_tokens = self.tokenize(chunk).to(self.device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    text_features = clip_model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.half().cpu().numpy()
                for i in range(text_features.shape[0]):
                    self.embeds.append(text_features[i])

            if cache_filepath is not None:
                with open(cache_filepath, 'wb') as f:
                    pickle.dump({
                        "labels": self.labels, 
                        "embeds": self.embeds, 
                        "hash": hash, 
                        "model": config.clip_model_name
                    }, f)

        if self.device == 'cpu' or self.device == torch.device('cpu'):
            self.embeds = [e.astype(np.float32) for e in self.embeds]
    
    def _rank(self, image_features: torch.Tensor, text_embeds: torch.Tensor, top_count: int=1) -> str:
        top_count = min(top_count, len(text_embeds))
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).to(self.device)
        with torch.cuda.amp.autocast():
            similarity = image_features @ text_embeds.T
        _, top_labels = similarity.float().cpu().topk(top_count, dim=-1)
        return [top_labels[0][i].numpy() for i in range(top_count)]

    def rank(self, image_features: torch.Tensor, top_count: int=1) -> List[str]:
        if len(self.labels) <= self.chunk_size:
            tops = self._rank(image_features, self.embeds, top_count=top_count)
            return [self.labels[i] for i in tops]

        num_chunks = int(math.ceil(len(self.labels)/self.chunk_size))
        keep_per_chunk = int(self.chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks), disable=self.config.quiet):
            start = chunk_idx*self.chunk_size
            stop = min(start+self.chunk_size, len(self.embeds))
            tops = self._rank(image_features, self.embeds[start:stop], top_count=keep_per_chunk)
            top_labels.extend([self.labels[start+i] for i in tops])
            top_embeds.extend([self.embeds[start+i] for i in tops])

        tops = self._rank(image_features, top_embeds, top_count=top_count)
        return [top_labels[i] for i in tops]


def _load_list(data_path: str, filename: str) -> List[str]:
    with open(os.path.join(data_path, filename), 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items

def _merge_tables(tables: List[LabelTable], config: Config) -> LabelTable:
    m = LabelTable([], None, None, None, config)
    for table in tables:
        m.labels.extend(table.labels)
        m.embeds.extend(table.embeds)
    return m

def _prompt_at_max_len(text: str, tokenize) -> bool:
    tokens = tokenize([text])
    return tokens[0][-1] != 0

def _truncate_to_fit(text: str, tokenize) -> str:
    parts = text.split(', ')
    new_text = parts[0]
    for part in parts[1:]:
        if _prompt_at_max_len(new_text + part, tokenize):
            break
        new_text += ', ' + part
    return new_text
