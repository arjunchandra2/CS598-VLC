import os
os.environ['TRANSFORMERS_CACHE'] = '/projectnb/cs598/projects/comp_reason/CS598-VLC/finetune/hf_cache'
os.environ['HF_HOME'] = '/projectnb/cs598/projects/comp_reason/CS598-VLC/finetune/hf_cache'
import json
import torch
import argparse
from filelock import FileLock
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_captions(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def save_captions(path, new_data, lock):
    with lock:
        existing = load_captions(path)
        existing.update(new_data)
        with open(path, 'w') as f:
            json.dump(existing, f, indent=4)

def make_prompt(caption):
    return f"""You are given the description of an image. You should provide a mostly similar description, changing the original one slightly but introducing enough significant 
differences such that the two descriptions could not possibly be for the same image. Keep the description length the same. Finally, only a few things 
(such as counting, objects, attributes, and relationships) can be modified to change the image structure significantly. Provide just the updated description. 

Examples:

Input: A dog to the left of the cat.
Output: A dog to the right of the cat.

Input: A person wearing a red helmet drives a motorbike on a dirt road.
Output: A person in a blue helmet rides a motorbike on a gravel path.

Now, do the same for the following caption. Only answer with one output version:

Input: {caption}
Output:"""

def process_batch(model, tokenizer, prompts, image_ids):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Remove prompt text from output
    cleaned = []
    for prompt, output in zip(prompts, decoded):
        trimmed = output[len(prompt):].strip() if output.startswith(prompt) else output.strip()
        cleaned.append(trimmed)

    return dict(zip(image_ids, cleaned))

def main():
    #to use shared env run: conda activate /projectnb/cs598/projects/comp_reason/.conda/envs/dac
    #then run: 
    #python negative_captions.py --rank 0 

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--input_json', type=str, default='/projectnb/cs598/projects/comp_reason/CS598-VLC/finetune/data/captions_val.json')
    parser.add_argument('--output_json', type=str, default='/projectnb/cs598/projects/comp_reason/CS598-VLC/finetune/data/negative_captions_val.json')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    # Device setup
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    devices = cuda_visible.split(",")
    total_workers = len(devices)
    if args.rank >= len(devices):
        raise ValueError(f"Rank {args.rank} out of available devices {devices}")
    device = f"cuda:{devices[args.rank]}"
    print(f"Running on {device}")

    # Load model and tokenizer
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )

    # Load positive captions and any already generated negative captions 
    all_captions = load_captions(args.input_json)
    with FileLock(args.output_json + ".lock"):
        already_done = load_captions(args.output_json)

    # Partition work across number of gpus
    items = [(k, v) for i, (k, v) in enumerate(all_captions.items()) if (i % total_workers) == args.rank and k not in already_done]

    print(f"{len(items)} captions assigned to rank {args.rank}")

    # Process in batches
    lock = FileLock(args.output_json + ".lock")
    for i in range(0, len(items), args.batch_size):
        batch = items[i:i + args.batch_size]
        image_ids, captions = zip(*batch)
        prompts = [make_prompt(c) for c in captions]
        try:
            neg_captions = process_batch(model, tokenizer, prompts, image_ids)
            save_captions(args.output_json, neg_captions, lock)
            print(f"[Rank {args.rank}] Processed batch {i // args.batch_size + 1} / {len(items) // args.batch_size + 1}")
        except Exception as e:
            print(f"[Rank {args.rank}] Error in batch {i // args.batch_size + 1}: {e}")

if __name__ == "__main__":
    main()