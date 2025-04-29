import json
import csv

# Paths to your JSON files
positive_json_path = "/projectnb/cs598/projects/comp_reason/CS598-VLC/finetune/data/captions_val.json"
negative_json_path = "/projectnb/cs598/projects/comp_reason/CS598-VLC/finetune/data/negative_captions_val.json"

# Prefix to prepend to each image path
prefix = "/projectnb/cs598/projects/comp_reason/CS598-VLC/finetune/data/openimages/val"

# Load the JSON files
with open(positive_json_path, 'r') as f:
    positive_captions = json.load(f)

with open(negative_json_path, 'r') as f:
    negative_captions = json.load(f)

# Output CSV path
output_csv_path = "/projectnb/cs598/projects/comp_reason/CS598-VLC/finetune/data/val.csv"

# Write merged data to CSV
with open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "positive_caption", "negative_caption"])

    # Use intersection of keys to ensure both captions exist
    common_keys = set(positive_captions) & set(negative_captions)
    for key in sorted(common_keys):
        full_path = prefix + "/" + key
        writer.writerow([full_path, positive_captions[key], negative_captions[key]])

print(f"CSV saved to {output_csv_path}")
