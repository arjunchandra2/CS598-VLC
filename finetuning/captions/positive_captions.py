import os
os.environ['TRANSFORMERS_CACHE'] = '/projectnb/cs598/projects/comp_reason/CS598-VLC/finetune/hf_cache'
os.environ['HF_HOME'] = '/projectnb/cs598/projects/comp_reason/CS598-VLC/finetune/hf_cache'
import json
import torch
import argparse
import hashlib
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from filelock import FileLock
from multiprocessing import Process, Queue, cpu_count

def load_captions(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def update_captions(file_path, new_captions, lock):
    with lock:
        captions = load_captions(file_path)
        captions.update(new_captions)
        with open(file_path, "w") as f:
            json.dump(captions, f, indent=4)

def image_loader(data_folder, rank, total_workers, loader_id, num_loaders, global_captions, queue, batch_size):
    batch = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                image_id = os.path.basename(full_path)
                h = int(hashlib.md5(full_path.encode("utf-8")).hexdigest(), 16)

                # Partition images first across GPUs (total_workers) then across loader processes (num_loaders)
                if (h % total_workers != rank) or ((h // total_workers) % num_loaders != loader_id):
                    continue
                if image_id in global_captions:
                    print(f"Skipping {image_id}: aleady captioned.")
                    continue
                try:
                    img = Image.open(full_path).convert("RGB")
                    batch.append((image_id, img))
                    if len(batch) == batch_size:
                        queue.put(batch)
                        batch = []
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")
    if batch:
        queue.put(batch)
    queue.put(None)  # Signal completion


def gpu_worker(queue, device, processor, model, captions_file, lock):
    while True:
        batch = queue.get()
        if batch is None:
            break
        image_ids, images = zip(*batch)
        texts = [""] * len(images)
        inputs = processor(images=images, text=texts, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        new_captions = {img_id: caption.strip() for img_id, caption in zip(image_ids, captions)}
        update_captions(captions_file, new_captions, lock)
        print(f"Processed batch of {len(images)} images on GPU {device}")

def main():
    parser = argparse.ArgumentParser(description="Efficient multiprocessing image captioning.")
    parser.add_argument('--rank', type=int, required=True, help='Process rank (0-indexed)')
    parser.add_argument('--data_folder', type=str, default='./', help='Image directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_loaders', type=int, default=4, help='Number of loader processes')
    args = parser.parse_args()

    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible is None:
        raise ValueError("CUDA_VISIBLE_DEVICES is not set.")
    devices = cuda_visible.split(',')
    total_workers = len(devices)
    if args.rank >= total_workers:
        raise ValueError("Rank exceeds available GPUs.")

    device = f"cuda:{devices[args.rank]}"
    print(f"GPU worker on {device}")

    captions_file = "./data/captions_test.json"
    lock = FileLock("captions.json.lock")
    with lock:
        global_captions = load_captions(captions_file)

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b", torch_dtype=torch.float16).to(device)

    queue = Queue(maxsize=8)  # Adjust based on memory constraints
    loaders = [
        Process(target=image_loader, args=(
            args.data_folder, args.rank, total_workers, loader_id, args.num_loaders,
            global_captions, queue, args.batch_size))
        for loader_id in range(args.num_loaders)
    ]

    for loader in loaders:
        loader.start()

    gpu_worker(queue, device, processor, model, captions_file, lock)

    for loader in loaders:
        loader.join()

    print("All processing complete.")

if __name__ == '__main__':
    main()