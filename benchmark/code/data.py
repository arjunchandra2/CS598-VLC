import os
import logging
import json
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import ast

class Crepe(Dataset):
    def __init__(self, transforms, tokenize, category):
        input_filename = f"data/crepe/{category[:4]}_hard_negatives/{category[:4]}_vg_hard_negs_{category[5:]}_all.csv"
        df = pd.read_csv(input_filename)
        self.hard_negs = [list(ast.literal_eval(ls_str)) for ls_str in df["hard_negs"]]
        self.images = df["image_id"].tolist()
        self.captions = df["caption"].tolist()
        self.transforms = transforms
        self.tokenize = (lambda x: x) if tokenize is None else tokenize

    def __len__(self):
        return len(self.captions)

    def get_image_by_id(self, image_id): 
        vg_image_paths = ['data/crepe/VG_100K', 'data/crepe/VG_100K_2']
        for p in vg_image_paths:
            path = os.path.join(p, f"{image_id}.jpg")
            if os.path.exists(path):
                return Image.open(path).convert("RGB")
        raise FileNotFoundError(f'The image with id {image_id} is not found.')
        return None

    def __getitem__(self, idx):
        image = self.transforms(self.get_image_by_id(self.images[idx]))
        caption_pos = self.tokenize([self.captions[idx]])
        caption_neg = self.tokenize(self.hard_negs[idx])
        return image, caption_pos, caption_neg

class SugarCrepe(Dataset):
    def __init__(self, transforms, tokenize, category):
        self.dataset = list(json.load(open(f"data/sugarcrepe/captions/{category}.json", 'r', encoding='utf-8')).values())
        self.transforms = transforms
        self.tokenize = (lambda x: x) if tokenize is None else tokenize

    def __len__(self):
        return len(self.dataset)

    def get_image(self, file_name): 
        image_path = os.path.join("data/sugarcrepe/val2017", file_name)
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transforms(self.get_image(sample["filename"]))
        caption_pos = self.tokenize([sample["caption"]])
        caption_neg = self.tokenize([sample["negative_caption"]])
        return image, caption_pos, caption_neg


DATASETS = {
    "crepe": Crepe,
    "sugarcrepe": SugarCrepe,
}

def get_dataset(name, prep_image, prep_text, **kwargs):
    assert name in DATASETS
    return DATASETS[name](prep_image, prep_text, **kwargs)