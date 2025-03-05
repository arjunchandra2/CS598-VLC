import os
import math
import torch
from dotmap import DotMap
from torch.utils.data import DataLoader
from torch.nn import CosineSimilarity
from tqdm import tqdm
from .data import get_dataset
from .models import get_model

BSIZE = 64
DEVICE = torch.device("cuda")
        
def eval(model, dataloader, device, test_run=False):
    n_batches = len(dataloader) if not test_run else min(10, len(dataloader))
    acc = torch.zeros(n_batches)
    acc_uni = torch.zeros(n_batches)
    metric = CosineSimilarity(dim=-1)
    for i, batch in tqdm(enumerate(dataloader), total=n_batches):
        if i == n_batches:
            break
        with torch.no_grad():
            images, captions_pos, captions_neg = batch
            captions = torch.cat([captions_pos, captions_neg], dim=1)

            bsize, n_captions = captions.shape[:2]
            captions = captions.reshape(bsize*n_captions, -1)
            images, captions = images.to(device), captions.to(device)
            
            image_features, text_features, logit_scale = model(images, captions)

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

    model, prep_image, prep_text = get_model("DAC-SAM", DEVICE)
    model.eval()

    dataset = get_dataset("sugarcrepe", prep_image, prep_text, category="replace_rel")
    dataloader = DataLoader(
        dataset,
        batch_size=BSIZE,
        shuffle=False,
        pin_memory=True,
    )

    acc_multi, acc_uni = eval(model, dataloader, DEVICE, test_run=True)

    print(acc_multi, acc_uni)