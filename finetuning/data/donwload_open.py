import os
import io
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure the S3 client for public (unsigned) access
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

bucket = 'open-images-dataset'
prefix = 'test/'
local_base_dir = os.path.join('data', 'openimages', 'test')
os.makedirs(local_base_dir, exist_ok=True)

def download_and_resize(key):
    relative_path = key[len(prefix):]  # remove the prefix from key
    local_path = os.path.join(local_base_dir, relative_path)
    
    if os.path.exists(local_path):
        print(f"Skipping {key}: already downloaded.")
        return

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read()
        
        with Image.open(io.BytesIO(data)) as img:
            resized_img = img.resize((256, 256), Image.ANTIALIAS)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            resized_img.save(local_path)
            print(f"Downloaded and resized: {key} -> {local_path}")
    except Exception as e:
        print(f"Error processing {key}: {e}")

def process_bucket_concurrently(max_workers=32):
    continuation_token = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            params = {'Bucket': bucket, 'Prefix': prefix}
            if continuation_token:
                params['ContinuationToken'] = continuation_token

            print("Listing objects...")
            response = s3.list_objects_v2(**params)
            if 'Contents' not in response:
                break

            futures = []
            for obj in response['Contents']:
                key = obj['Key']
                if key.lower().endswith(('.jpg', '.jpeg')):
                    futures.append(executor.submit(download_and_resize, key))

            # Wait for the current batch to finish
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in concurrent task: {e}")

            if response.get('IsTruncated'):
                continuation_token = response.get('NextContinuationToken')
            else:
                break

if __name__ == '__main__':
    process_bucket_concurrently(max_workers=12)