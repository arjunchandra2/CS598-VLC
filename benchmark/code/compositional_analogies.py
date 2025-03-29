import os
import math
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import pairwise_distance
from torchvision import transforms
from tqdm import tqdm
import itertools
from PIL import Image

from .models import get_model

device = torch.device("cuda")

def get_image_path(prep, object_s):
    data_path = "/projectnb/cs598/students/ac25/What's Up/data/controlled_images/"

    if prep == 'left' or prep == 'right':
        prep += '_of'
        
    return data_path + object_s + '_' + prep + '_table.jpeg'


def comp_analogies_image(model_forward, preprocess_val, tokenize, verbose):
    prepositions = ['on', 'under', 'left', 'right']
    objects = ['mug', 'scarf', 'dragonfruit', 'phone', 'cap', 'orange', 'banana', 'lemon', 'plate', 'scissors', 'kettle', 'cup', 'spatula']
    
    correct_analogies = 0
    total_analogies = 0
    total_l2_distance = 0.0 
    num_degenerate = 0
    
    dummy_text = tokenize(["dummy"]).to(device)  # Dummy text input for model_forward
    
    total_iterations = len(list(itertools.product(itertools.combinations(prepositions, 2), itertools.combinations(objects, 2))))

    for prep_pair, object_pair in tqdm(itertools.product(itertools.combinations(prepositions, 2), itertools.combinations(objects, 2)), 
                                        total=total_iterations):
        
         # image file paths
        prep0_object0_path = get_image_path(prep_pair[0], object_pair[0])
        prep1_object0_path = get_image_path(prep_pair[1], object_pair[0])
        prep0_object1_path = get_image_path(prep_pair[0], object_pair[1])
        prep1_object1_path = get_image_path(prep_pair[1], object_pair[1])

        # Load and preprocess images
        prep0_object0 = preprocess_val(Image.open(prep0_object0_path).convert("RGB")).unsqueeze(0)
        prep1_object0 = preprocess_val(Image.open(prep1_object0_path).convert("RGB")).unsqueeze(0)
        prep0_object1 = preprocess_val(Image.open(prep0_object1_path).convert("RGB")).unsqueeze(0)
        prep1_object1 = preprocess_val(Image.open(prep1_object1_path).convert("RGB")).unsqueeze(0)

        if verbose:
            print(f"Testing analogy: {object_pair[0]} {prep_pair[0]} table - {object_pair[0]} {prep_pair[1]} table + {object_pair[1]} {prep_pair[1]} table = {object_pair[1]} {prep_pair[0]} table")

        # Collect negative examples
        negative_images = []
        negative_preps = []
        for neg_prep in prepositions:
            if neg_prep == prep_pair[0]:
                continue
            neg_path = get_image_path(neg_prep, object_pair[1])
            neg_image = preprocess_val(Image.open(neg_path).convert("RGB")).unsqueeze(0)
            negative_images.append(neg_image)
            negative_preps.append(neg_prep)

        # Batch all images together 
        all_images = torch.cat([prep0_object0, prep1_object0, prep0_object1, prep1_object1] + negative_images).to(device)

        # Forward pass
        image_features, _ = model_forward(all_images, dummy_text)

        # Extract embeddings
        prep0_object0_embedding = image_features[0]
        prep1_object0_embedding = image_features[1]
        prep0_object1_embedding = image_features[2]
        prep1_object1_embedding = image_features[3]
        negative_embeddings = image_features[4:]  # Remaining embeddings are negatives

        # Compute the analogy: prep0_object0 - prep1_object0 + prep1_object1
        analogy = prep0_object0_embedding - prep1_object0_embedding + prep1_object1_embedding

        # Compute L2 distance to the correct answer
        l2_distance = torch.norm(analogy - prep0_object1_embedding, p=2).item()
        if verbose:
            print(f"Analogy L2 distance to {object_pair[1]} {prep_pair[0]} table: {l2_distance:.4f}")

        # Compare with negative examples
        correct = True
        for neg_prep, neg_embedding in zip(negative_preps, negative_embeddings):
            neg_l2_distance = torch.norm(analogy - neg_embedding, p=2).item()
            if verbose:
                print(f"L2 distance to negative {object_pair[1]} {neg_prep} table: {neg_l2_distance:.4f}")
            if l2_distance > neg_l2_distance:
                correct = False
                if neg_prep == prep_pair[1]:
                    num_degenerate += 1

        if correct:
            correct_analogies += 1
        total_analogies += 1

        total_l2_distance += l2_distance
        
    accuracy = (correct_analogies / total_analogies) * 100 
    avg_l2_distance = total_l2_distance / total_analogies
    print(f"Image Accuracy: {accuracy:.2f}%")
    print(f"Average L2 distance of analogies: {avg_l2_distance:.4f}")
    print(f"Number of degenerate analogies: {num_degenerate}")


def comp_analogies_text(model_forward, preprocess_val, tokenize, verbose):

    prepositions = ['on', 'under', 'left', 'right']
    objects = ['mug', 'scarf', 'dragonfruit', 'phone', 'cap', 'orange', 'banana', 'lemon', 'plate', 'scissors', 'kettle', 'cup', 'spatula']

    correct_analogies = 0
    total_analogies = 0
    total_l2_distance = 0.0 

    total_iterations = len(list(itertools.product(itertools.combinations(prepositions, 2), itertools.combinations(objects, 2))))
    for prep_pair, object_pair in tqdm(itertools.product(itertools.combinations(prepositions, 2), itertools.combinations(objects, 2)), 
                                        total=total_iterations):
        # Define analogy pairs 
        prep0_object0 = object_pair[0] + ' ' + prep_pair[0] + ' table' 
        prep1_object0 = object_pair[0] + ' ' + prep_pair[1] + ' table'
        prep0_object1 = object_pair[1] + ' ' +  prep_pair[0] + ' table' 
        prep1_object1 = object_pair[1] + ' ' + prep_pair[1] + ' table' 

        # Negative examples
        negatives = [object_pair[1] + ' ' + prep + ' table' for prep in prepositions if prep != prep_pair[0]]

        if verbose:
            print(f"Testing analogy: {prep0_object0} - {prep1_object0} + {prep1_object1} = {prep0_object1}")

        # Use random image (we only care about text embeddings)
        dummy_image = torch.zeros(3, 224, 224).to(device)  
        dummy_image = transforms.ToPILImage()(dummy_image)  
        dummy_image = preprocess_val(dummy_image).unsqueeze(0).to(device)  

        captions = [prep0_object0, prep1_object0, prep0_object1, prep1_object1] + negatives
        tokenized_captions = tokenize(captions).to(device)  

        # Forward pass (text embeddings are all 512 dims?)
        image_features, text_features = model_forward(dummy_image, tokenized_captions)

        # Get the embeddings for the analogy components
        prep0_object0_embedding = text_features[0]
        prep1_object0_embedding = text_features[1]
        prep0_object1_embedding = text_features[2]
        prep1_object1_embedding = text_features[3]
        negative_embeddings = text_features[4:]

        # Compute the analogy: prep0_object0 - prep1_object0 + prep1_object1
        analogy = prep0_object0_embedding - prep1_object0_embedding + prep1_object1_embedding
        
        # Calculate the L2 distance between analogy and prep0_object1 embedding
        l2_distance = pairwise_distance(analogy.unsqueeze(0), prep0_object1_embedding.unsqueeze(0))
        if verbose:
            print(f"Analogy L2 distance to {prep0_object1}: {l2_distance.item():.4f}")

        # Now calculate the L2 distance to the negative examples and check if the analogy holds 
        correct = True
        for neg_emb in negative_embeddings:
            neg_l2_distance = pairwise_distance(analogy.unsqueeze(0), neg_emb.unsqueeze(0))
            if verbose:
                print(f"Analogy L2 distance to negative example: {neg_l2_distance.item():.4f}")
            if l2_distance.item() > neg_l2_distance.item():
                correct = False

        if correct:
            correct_analogies += 1
        total_analogies += 1

        total_l2_distance += l2_distance.item()
    
    accuracy = (correct_analogies / total_analogies) * 100 
    avg_l2_distance = total_l2_distance / total_analogies
    print(f"Text Accuracy: {accuracy:.2f}%")
    print(f"Average L2 distance of analogies: {avg_l2_distance:.4f}")


if __name__ == "__main__":
    #to use shared env run: conda activate /projectnb/cs598/projects/comp_reason/.conda/envs/dac

    # load models
    models = [
        (x, *get_model(x, device))
        for x in ["DAC-SAM", "ViT", "DAC-SAM-base", "DAC-SAM-base-2"]
    ]

    #Run experiments
    for (model_name, model_forward, preprocess_train, preprocess_val, tokenize) in models: 
        print(f"Running compositional analogies for {model_name}:")
        comp_analogies_text(model_forward, preprocess_val, tokenize, verbose=False)
        comp_analogies_image(model_forward, preprocess_val, tokenize, verbose=False)
        

