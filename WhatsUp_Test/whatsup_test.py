# Uses dataset_zoo from WhatsUp's code : https://github.com/amitakamath/whatsup_vlms

import clip
from dataset_zoo import Controlled_Images
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer

model, image_preprocess = clip.load("ViT-B/32", device="cuda")

root_dir="data"
controlled_a = Controlled_Images(image_preprocess=None, subset="A", download=False)  # Remove the download flag if you already have the data
controlled_b = Controlled_Images(image_preprocess=None, subset="B", download=False)

# Do anything with the dataset. Each item will look like this : 
# item = {"image_options": [image], "caption_options": [true_caption, false_caption_1, ...]}
    
def separate_by_direction(dataset, is_dataset_a=True):

    # Seperate the dataset by the four directions

    left_of = []
    right_of = []
    on_or_in_front_of = []
    under_or_behind = []

    for item in dataset:
        # First caption is the true one
        true_caption = item['caption_options'][0].lower()

        if "left of" in true_caption:
            left_of.append(item)
        elif "right of" in true_caption:
            right_of.append(item)
        elif ("on" in true_caption and is_dataset_a) or ("in front of" in true_caption and not is_dataset_a):
            on_or_in_front_of.append(item)
        elif ("under" in true_caption and is_dataset_a) or ("behind" in true_caption and not is_dataset_a):
            under_or_behind.append(item)

    return left_of, right_of, on_or_in_front_of, under_or_behind

# Separate both datasets
left_a, right_a, on_a, under_a = separate_by_direction(controlled_a, is_dataset_a=True)
left_b, right_b, in_front_b, behind_b = separate_by_direction(controlled_b, is_dataset_a=False)

print(f"Dataset A - Left: {len(left_a)}, Right: {len(right_a)}, On: {len(on_a)}, Under: {len(under_a)}")
print(f"Dataset B - Left: {len(left_b)}, Right: {len(right_b)}, In Front: {len(in_front_b)}, Behind: {len(behind_b)}")

def evaluate_clip(model, dataset, device):

    # Evaluate CLIP model across the four different directions

    correct = 0
    total = 0

    for item in dataset:
        images = item['image_options']
        captions = item['caption_options']

        # Preprocess images and encode using CLIP
        image_tensors = []

        for img in images:
            # If img is a tensor, convert to PIL image first
            if isinstance(img, torch.Tensor):
                # Ensure the image is in the correct format (H, W, C) and convert it to uint8
                img = img.permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) -> (H, W, C)
                img = np.uint8(img * 255)  # Scale the tensor to [0, 255] and convert to uint8
                img = Image.fromarray(img)  # Convert the numpy array to a PIL image

            image_tensors.append(image_preprocess(img).to(device))

        image_tensors = torch.stack(image_tensors)

        text_tokens = clip.tokenize(captions).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensors)
            text_features = model.encode_text(text_tokens)

            # Normalize embeddings for cosine similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = text_features @ image_features.T

        # Check if the true caption has the highest similarity
        predicted_index = similarity.argmax(dim=0)
        if predicted_index[0] == 0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0

def evaluate_clip_overall(model, dataset, device):

    # Evaluate CLIP model across the entire test set

    correct = 0
    total = 0

    for item in dataset:
        images = item['image_options']
        captions = item['caption_options']

        # Preprocess images and encode using CLIP
        image_tensors = []

        for img in images:
            # If img is a tensor, convert to PIL image first
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) -> (H, W, C)
                img = np.uint8(img * 255)  # Scale the tensor to [0, 255] and convert to uint8
                img = Image.fromarray(img)  # Convert the numpy array to a PIL image

            image_tensors.append(image_preprocess(img).to(device))

        image_tensors = torch.stack(image_tensors)

        text_tokens = clip.tokenize(captions).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensors)
            text_features = model.encode_text(text_tokens)

            # Normalize embeddings for cosine similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = text_features @ image_features.T

        # Check if the true caption has the highest similarity
        predicted_index = similarity.argmax(dim=0)
        if predicted_index[0] == 0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0

def evaluate_clip_pairwise(model, dataset, directions, device):

    # Evaluate CLIP model pairwise (WIP)

    correct = 0
    total = 0

    # Get the indices for the two directions
    direction_1, direction_2 = directions

    for item in dataset:
        images = item['image_options']
        captions = item['caption_options']

        # Only consider captions for the two directions of interest
        relevant_captions = [caption for caption in captions if direction_1 in caption or direction_2 in caption]
        if len(relevant_captions) < 2:
            continue  # Skip if we don't have both captions for the two directions

        # Preprocess images and encode using CLIP
        image_tensors = []

        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) -> (H, W, C)
                img = np.uint8(img * 255)  # Scale the tensor to [0, 255] and convert to uint8
                img = Image.fromarray(img)  # Convert the numpy array to a PIL image

            image_tensors.append(image_preprocess(img).to(device))

        image_tensors = torch.stack(image_tensors)

        # Prepare text tokens
        text_tokens = clip.tokenize(relevant_captions).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensors)
            text_features = model.encode_text(text_tokens)

            # Normalize embeddings for cosine similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = text_features @ image_features.T

        # Get the index of the predicted caption
        predicted_index = similarity.argmax(dim=0)

        # Determine if the prediction was correct (matching the first or second direction)
        true_index = 0 if direction_1 in relevant_captions[0] else 1
        if predicted_index[0] == true_index:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0


# Benchmarks

# Evaluate all four directions for both datasets
def run_benchmark():
    print("Evaluating Dataset A...")
    for name, data in [("Left", left_a), ("Right", right_a), ("On", on_a), ("Under", under_a)]:
        acc = evaluate_clip(model, data, device="cuda")
        print(f"{name}: {acc:.4f}")

    print("\nEvaluating Dataset B...")
    for name, data in [("Left", left_b), ("Right", right_b), ("In Front Of", in_front_b), ("Behind", behind_b)]:
        acc = evaluate_clip(model, data, device="cuda")
        print(f"{name}: {acc:.4f}")


# Overall accuracy for Dataset A and B
def run_overall_benchmark():
    print("Evaluating Dataset A...")
    acc_a = evaluate_clip_overall(model, controlled_a, device="cuda")
    print(f"Overall Accuracy for Dataset A: {acc_a:.4f}")

    print("\nEvaluating Dataset B...")
    acc_b = evaluate_clip_overall(model, controlled_b, device="cuda")
    print(f"Overall Accuracy for Dataset B: {acc_b:.4f}")


def run_pairwise_benchmark():
    # Left vs Right pairwise accuracy
    print("Evaluating Pairwise Accuracy: Left vs Right...")
    pairwise_acc_left_right = evaluate_clip_pairwise(model, controlled_a, directions=("left of", "right of"), device="cuda")
    print(f"Pairwise Accuracy (Left vs Right) for Dataset A: {pairwise_acc_left_right:.4f}")

    # On vs Under pairwise accuracy
    print("\nEvaluating Pairwise Accuracy: On vs Under...")
    pairwise_acc_on_under = evaluate_clip_pairwise(model, controlled_a, directions=("on", "under"), device="cuda")
    print(f"Pairwise Accuracy (On vs Under) for Dataset A: {pairwise_acc_on_under:.4f}")

    # Left vs Right pairwise accuracy for Dataset B
    print("\nEvaluating Pairwise Accuracy: Left vs Right for Dataset B...")
    pairwise_acc_left_right_b = evaluate_clip_pairwise(model, controlled_b, directions=("left of", "right of"), device="cuda")
    print(f"Pairwise Accuracy (Left vs Right) for Dataset B: {pairwise_acc_left_right_b:.4f}")

    # On vs Under pairwise accuracy for Dataset B
    print("\nEvaluating Pairwise Accuracy: On vs Under for Dataset B...")
    pairwise_acc_on_under_b = evaluate_clip_pairwise(model, controlled_b, directions=("on", "under"), device="cuda")
    print(f"Pairwise Accuracy (On vs Under) for Dataset B: {pairwise_acc_on_under_b:.4f}")


def main():
    # Run overall benchmark first
    run_overall_benchmark()

    # Run benchmark with directions afterward
    run_benchmark()

    # Run the pairwise benchmark
    run_pairwise_benchmark()

if __name__ == "__main__":
    main()