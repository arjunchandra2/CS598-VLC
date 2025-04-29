import logging

import datasets
import pandas as pd
from open_clip import tokenize
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class HFDataset(Dataset):
    def __init__(self, data_location, transforms, image_col, text_col):
        logging.debug(f"Loading HF dataset from {data_location}.")
        self.dataset = datasets.load_dataset(data_location, split="train")
        self.image_col = image_col
        self.text_col = text_col
        self.transforms = transforms
        logging.debug("Done loading data.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        images = self.transforms(self.dataset[idx][self.image_col])
        texts = tokenize([self.dataset[idx][self.text_col]])[0]
        return images, texts

#MODIFIED TO RETURN TWO CAPTIONS PER IMAGE
class CSVDataset(Dataset):
    def __init__(self, data_location, transforms, image_col="image_path", pos_col="positive_caption", neg_col="negative_caption", sep=","):
        logging.debug(f"Loading csv data from {data_location}.")
        df = pd.read_csv(data_location, sep=sep)

        self.images = df[image_col].tolist()
        self.positive_captions = df[pos_col].tolist()
        self.negative_captions = df[neg_col].tolist()
        self.transforms = transforms
        logging.debug("Done loading data.")

    def __len__(self):
        return len(self.positive_captions)

    def __getitem__(self, idx):
        image = self.transforms(Image.open(str(self.images[idx])).convert("RGB"))
        pos_text = tokenize([str(self.positive_captions[idx])])[0]
        neg_text = tokenize([str(self.negative_captions[idx])])[0]
        return image, pos_text, neg_text



def get_dataloader(args, preprocess, split="train"):
    if args.datatype == "hf":
        dataset = HFDataset(
            data_location=args.train_dataset if split == "train" else args.eval_dataset,
            transforms=preprocess,
            image_col=args.image_col,
            text_col=args.text_col,
        )
    elif args.datatype == "csv":
        dataset = CSVDataset(
            data_location=args.train_dataset if split == "train" else args.eval_dataset,
            transforms=preprocess,
            image_col=args.image_col,
            pos_col=args.pos_col,
            neg_col=args.neg_col,
            sep=args.csv_separator
        )
    num_samples = len(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return dataloader
