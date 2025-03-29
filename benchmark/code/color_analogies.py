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

def color_analogies_image(model_forward, preprocess_val, tokenize, verbose):
    colors = ['red', 'yellow', 'green', 'blue']
    objects = ['chair', 'mug', 'apple', 'hat', 'shoe', 'yarn', 'scarf', 'kettle', 'book', 'box', 'helmet', 'glove', 'lemon']
    
    correct_analogies = 0
    total_analogies = 0
    total_l2_distance = 0.0 
    num_degenerate = 0
    
    data_path = "/projectnb/cs598/students/ac25/color_analogies_data"
    dummy_text = tokenize(["dummy"]).to(device)  # Dummy text input for model_forward
    
    total_iterations = len(list(itertools.product(itertools.combinations(colors, 2), itertools.combinations(objects, 2))))

    for color_pair, object_pair in tqdm(itertools.product(itertools.combinations(colors, 2), itertools.combinations(objects, 2)), 
                                        total=total_iterations):
        # image file paths
        color0_object0_path = os.path.join(data_path, f"{color_pair[0]}_{object_pair[0]}.png")
        color1_object0_path = os.path.join(data_path, f"{color_pair[1]}_{object_pair[0]}.png")
        color0_object1_path = os.path.join(data_path, f"{color_pair[0]}_{object_pair[1]}.png")
        color1_object1_path = os.path.join(data_path, f"{color_pair[1]}_{object_pair[1]}.png")

        # Load and preprocess images
        color0_object0 = preprocess_val(Image.open(color0_object0_path).convert("RGB")).unsqueeze(0)
        color1_object0 = preprocess_val(Image.open(color1_object0_path).convert("RGB")).unsqueeze(0)
        color0_object1 = preprocess_val(Image.open(color0_object1_path).convert("RGB")).unsqueeze(0)
        color1_object1 = preprocess_val(Image.open(color1_object1_path).convert("RGB")).unsqueeze(0)

        if verbose:
            print(f"Testing analogy: {color_pair[0]} {object_pair[0]} - {color_pair[1]} {object_pair[0]} + {color_pair[1]} {object_pair[1]} = {color_pair[0]} {object_pair[1]}")

        # Collect negative examples
        negative_images = []
        negative_colors = []
        for neg_color in colors:
            if neg_color == color_pair[0]:
                continue
            neg_path = os.path.join(data_path, f"{neg_color}_{object_pair[1]}.png")
            neg_image = preprocess_val(Image.open(neg_path).convert("RGB")).unsqueeze(0)
            negative_images.append(neg_image)
            negative_colors.append(neg_color)

        # Batch all images together 
        all_images = torch.cat([color0_object0, color1_object0, color0_object1, color1_object1] + negative_images).to(device)

        # Forward pass
        image_features, _ = model_forward(all_images, dummy_text)

        # Extract embeddings
        color0_object0_embedding = image_features[0]
        color1_object0_embedding = image_features[1]
        color0_object1_embedding = image_features[2]
        color1_object1_embedding = image_features[3]
        negative_embeddings = image_features[4:]  # Remaining embeddings are negatives

        # Compute the analogy: color0_object0 - color1_object0 + color1_object1
        analogy = color0_object0_embedding - color1_object0_embedding + color1_object1_embedding

        # Compute L2 distance to the correct answer
        l2_distance = torch.norm(analogy - color0_object1_embedding, p=2).item()
        if verbose:
            print(f"Analogy L2 distance to {color_pair[0]} {object_pair[1]}: {l2_distance:.4f}")

        # Compare with negative examples
        correct = True
        for neg_color, neg_embedding in zip(negative_colors, negative_embeddings):
            neg_l2_distance = torch.norm(analogy - neg_embedding, p=2).item()
            if verbose:
                print(f"L2 distance to negative {neg_color} {object_pair[1]}: {neg_l2_distance:.4f}")
            if l2_distance > neg_l2_distance:
                correct = False
                if neg_color == color_pair[1]:
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


def color_analogies_text(model_forward, preprocess_val, tokenize, verbose):

    colors = ['red', 'yellow', 'green', 'blue']
    objects = ['chair', 'mug', 'apple', 'hat', 'shoe', 'yarn', 'scarf', 'kettle', 'book', 'box', 'helmet', 'glove', 'lemon']

    correct_analogies = 0
    total_analogies = 0
    total_l2_distance = 0.0 

    total_iterations = len(list(itertools.product(itertools.combinations(colors, 2), itertools.combinations(objects, 2))))
    for color_pair, object_pair in tqdm(itertools.product(itertools.combinations(colors, 2), itertools.combinations(objects, 2)), 
                                        total=total_iterations):
        # Define analogy pairs 
        color0_object0 = color_pair[0] + ' ' + object_pair[0]
        color1_object0 = color_pair[1] + ' ' + object_pair[0]
        color0_object1 = color_pair[0] + ' ' + object_pair[1]
        color1_object1 = color_pair[1] + ' ' + object_pair[1]

        # Negative examples
        negatives = [color + ' ' + object_pair[1] for color in colors if color != color_pair[0]]

        if verbose:
            print(f"Testing analogy: {color0_object0} - {color1_object0} + {color1_object1} = {color0_object1}")

        # Use random image (we only care about text embeddings)
        dummy_image = torch.zeros(3, 224, 224).to(device)  
        dummy_image = transforms.ToPILImage()(dummy_image)  
        dummy_image = preprocess_val(dummy_image).unsqueeze(0).to(device)  

        captions = [color0_object0, color1_object0, color0_object1, color1_object1] + negatives
        tokenized_captions = tokenize(captions).to(device)  

        # Forward pass (text embeddings are all 512 dims?)
        image_features, text_features = model_forward(dummy_image, tokenized_captions)

        # Get the embeddings for the analogy components
        color0_object0_embedding = text_features[0]
        color1_object0_embedding = text_features[1]
        color0_object1_embedding = text_features[2]
        color1_object1_embedding = text_features[3]
        negative_embeddings = text_features[4:]

        # Compute the analogy: color0_object0 - color1_object0 + color1_object1
        analogy = color0_object0_embedding - color1_object0_embedding + color1_object1_embedding
        
        # Calculate the L2 distance between analogy and color0_object1 embedding
        l2_distance = pairwise_distance(analogy.unsqueeze(0), color0_object1_embedding.unsqueeze(0))
        if verbose:
            print(f"Analogy L2 distance to {color0_object1}: {l2_distance.item():.4f}")

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
        print(f"Running color analogies for {model_name}:")
        color_analogies_text(model_forward, preprocess_val, tokenize, verbose=False)
        color_analogies_image(model_forward, preprocess_val, tokenize, verbose=True)
        

