import os
import math
import torch
import pandas as pd
from dotmap import DotMap
from torch.utils.data import DataLoader
from torch.nn import CosineSimilarity
from tqdm import tqdm
from itertools import product
from .data import get_dataset, CATEGORIES
from .models import get_model

BSIZE = 64
N_BATCHES = 256
DEVICE = torch.device("cuda")
        
def eval(model, dataloader, device, n_batches):
    n_batches = min(n_batches, len(dataloader)) if n_batches > 0 else (dataloader)
    acc = torch.zeros(n_batches)
    acc_uni = torch.zeros(n_batches)
    metric = CosineSimilarity(dim=-1)
    for i, batch in enumerate(dataloader):
        if i == n_batches:
            break
        with torch.no_grad():
            images, captions_pos, captions_neg = batch
            captions = torch.cat([captions_pos, captions_neg], dim=1)

            bsize, n_captions = captions.shape[:2]
            captions = captions.reshape(bsize*n_captions, -1)
            images, captions = images.to(device), captions.to(device)
            
            image_features, text_features = model(images, captions)

            image_features = image_features[:,None,:]
            text_features = text_features.reshape(bsize, n_captions, -1)
            pos_features = text_features[:,:1]
            neg_features = text_features[:,1:]

            metric_pos = metric(image_features, pos_features)
            metric_neg = metric(image_features, neg_features)
            acc[i] = torch.mean((metric_pos>metric_neg).float())

            image_features = torch.mean(image_features, dim=0, keepdims=True)
            metric_pos = metric(image_features, pos_features)
            metric_neg = metric(image_features, neg_features)
            acc_uni[i] = torch.mean((metric_pos>metric_neg).float())

    return torch.mean(acc).item(), torch.mean(acc_uni).item()

if __name__ == "__main__":

    # load models
    models = [
        (x, *get_model(x, DEVICE))
        for x in ["DAC-SAM", "ViT"]
    ]

    # load data with dummy image and text preprocessing methods
    prep_image, prep_text = models[0][2:]
    data = list()
    for x in ["winnoground"]:#"crepe", "sugarcrepe", "colorswap", "aro"
        for y in CATEGORIES[x]:
            if y is None:
                data.append((x, y, get_dataset(x, prep_image, prep_text)))
            else:
                data.append((x, y, get_dataset(x, prep_image, prep_text, category=y)))

    # loop over models and data
    metrics = list()
    iterator = tqdm(product(models, data), total=len(models)*len(data))
    for (model_name, model, prep_image, prep_image_val, prep_text), (data_name, category, dataset) in iterator:
        
        # update data preprocessing for model
        dataset.update_prep(prep_image, prep_text)

        # get dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=BSIZE,
            shuffle=False,
            pin_memory=True,
        )

        # run evaluation
        acc_multi, acc_uni = eval(model, dataloader, DEVICE, N_BATCHES)

        # save results
        metrics.append({
            "model": model_name, 
            "data": data_name, 
            "n_samples": BSIZE*min(N_BATCHES, len(dataloader)) if N_BATCHES > 0 else (dataloader),
            "category": category,
            "acc_multi": acc_multi,
            "acc_uni": acc_uni,
        })
    
    # save metrics in csv
    metrics = pd.DataFrame(metrics)
    metrics.to_csv("./results/benchmark_w.csv")