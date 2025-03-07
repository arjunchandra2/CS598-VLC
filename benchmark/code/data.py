import os
import logging
import json
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import ast
import clip
from libs.ARO.dataset_zoo import VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order

class Crepe(Dataset):
    def __init__(self, transform, tokenize, category):
        if category[:4] == "prod":
            input_filename = f"data/crepe/prod_hard_negatives/prod_vg_hard_negs_{category[5:]}_all.csv"
        elif category[:4] == "syst":
            input_filename = f"data/crepe/syst_hard_negatives/syst_vg_hard_negs_{category[5:]}_compounds_in_cc12m.csv"
        df = pd.read_csv(input_filename)
        self.hard_negs = [list(ast.literal_eval(ls_str)) for ls_str in df["hard_negs"]]
        self.images = df["image_id"].tolist()
        self.captions = df["caption"].tolist()
        self.transform = transform
        self.tokenize = tokenize

    def __len__(self):
        return len(self.captions)

    def update_prep(self, transform, tokenize):
        self.transform = transform
        self.tokenize = tokenize

    def get_image_by_id(self, image_id): 
        vg_image_paths = ['data/crepe/VG_100K', 'data/crepe/VG_100K_2']
        for p in vg_image_paths:
            path = os.path.join(p, f"{image_id}.jpg")
            if os.path.exists(path):
                return Image.open(path).convert("RGB")
        raise FileNotFoundError(f'The image with id {image_id} is not found.')
        return None

    def __getitem__(self, idx):
        image = self.transform(self.get_image_by_id(self.images[idx]))
        caption_pos = self.tokenize([self.captions[idx]])
        caption_neg = self.tokenize(self.hard_negs[idx])
        return image, caption_pos, caption_neg

class SugarCrepe(Dataset):
    def __init__(self, transform, tokenize, category):
        self.dataset = list(json.load(open(f"data/sugarcrepe/captions/{category}.json", 'r', encoding='utf-8')).values())
        self.transform = transform
        self.tokenize = tokenize

    def __len__(self):
        return len(self.dataset)

    def update_prep(self, transform, tokenize):
        self.transform = transform
        self.tokenize = tokenize

    def get_image(self, file_name): 
        image_path = os.path.join("data/sugarcrepe/val2017", file_name)
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(self.get_image(sample["filename"]))
        caption_pos = self.tokenize([sample["caption"]])
        caption_neg = self.tokenize([sample["negative_caption"]])
        return image, caption_pos, caption_neg

class ColorSwap(Dataset):
    def __init__(self, transform, tokenize):
        self.dataset = json.load(open(f"data/colorswap/test.json", 'r'))+json.load(open(f"data/colorswap/train.json", 'r'))
        self.transform = transform
        self.tokenize = tokenize
        
    def __len__(self):
        return len(self.dataset)

    def update_prep(self, transform, tokenize):
        self.transform = transform
        self.tokenize = tokenize

    def get_image(self, file_name): 
        image_path = os.path.join("data/colorswap", file_name)
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(self.get_image(sample["image_1"]))
        caption_pos = self.tokenize([sample["caption_1"]])
        caption_neg = self.tokenize([sample["caption_2"]])
        return image, caption_pos, caption_neg

class ARO(Dataset):
    def __init__(self, transform, tokenize, category):
        root_dir = "./data/aro/"
        if category == "relation":
            self.dataset = VG_Relation(image_preprocess=transform, download=False, root_dir=root_dir)
        elif category == "attribution":
            self.dataset = VG_Attribution(image_preprocess=transform, download=False, root_dir=root_dir)
        elif category == "coco":
            self.dataset = COCO_Order(image_preprocess=transform, download=False, root_dir=root_dir) 
        elif category == "flickr":
            self.dataset = Flickr30k_Order(image_preprocess=transform, root_dir=root_dir, split="test")
        else:
            raise ValueError
        self.tokenize = tokenize
        self.true_caption = "second" if category in ["relation", "attribution"] else "first"

    def update_prep(self, transform, tokenize):
        self.dataset.image_preprocess = transform
        self.tokenize = tokenize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image_options"][0]
        if self.true_caption == "second":
            caption_pos = self.tokenize([sample["caption_options"][1]])
            caption_neg = self.tokenize([sample["caption_options"][0]])
        else:
            caption_pos = self.tokenize([sample["caption_options"][0]])
            caption_neg = self.tokenize(sample["caption_options"][1:])
        return image, caption_pos, caption_neg

class Winnoground():
    def __init__(self, transform, tokenize):
        with open(f"data/winnoground/examples.jsonl", 'r') as json_file:
            json_list = list(json_file)
        self.dataset = list()
        for json_str in json_list:
            line = json.loads(json_str)
            self.dataset.append(line)
        self.transform = transform
        self.tokenize = tokenize

    def __len__(self):
        return 2*len(self.dataset)

    def update_prep(self, transform, tokenize):
        self.transform = transform
        self.tokenize = tokenize

    def get_image(self, file_name): 
        image_path = os.path.join("data/winnoground/images", file_name+".png")
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, idx):
        parity = idx // len(self.dataset)
        idx = idx % len(self.dataset)
        sample = self.dataset[idx]
        image = self.transform(self.get_image(sample["image_0" if parity else "image_1"]))
        caption_pos = self.tokenize([sample["caption_0" if parity else "caption_1"]])
        caption_neg = self.tokenize([sample["caption_1" if parity else "caption_0"]])
        return image, caption_pos, caption_neg

DATASETS = {
    "crepe": Crepe,
    "sugarcrepe": SugarCrepe,
    "colorswap": ColorSwap,
    "aro": ARO,
    "winnoground": Winnoground,
}

CATEGORIES = {
    "crepe": ["prod_atom", "prod_negate", "prod_swap"],#, "syst_seen", "syst_unseen"],
    "sugarcrepe": ["add_att", "add_obj", "replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"],
    "colorswap": [None],
    "aro": ["relation", "attribution", "coco", "flickr"],
    "winnoground": [None],
}

def get_dataset(name, prep_image, prep_text, **kwargs):
    assert name in DATASETS
    return DATASETS[name](prep_image, prep_text, **kwargs)