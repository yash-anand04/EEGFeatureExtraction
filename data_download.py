import boto3
import os
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

def download_chbmit_s3():
    bucket_name = 'physionet-open'
    prefix = 'chbmit/1.0.0/'
    local_root = './chbmit_data'

    # Configure S3 client for anonymous (unsigned) access
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    print("Listing files on S3...")
    paginator = s3.get_paginator('list_objects_v2')
    
    # Collect all files that belong to patients 01-23
    files_to_download = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Filter for chb01 through chb23
            parts = key.replace(prefix, "").split('/')
            if parts[0].startswith('chb'):
                try:
                    patient_num = int(parts[0].replace('chb', ''))
                    if 1 <= patient_num <= 23:
                        files_to_download.append((key, obj['Size']))
                except ValueError:
                    continue

    print(f"Found {len(files_to_download)} files. Starting download...")

    for key, size in tqdm(files_to_download, desc="Total Progress"):
        # Create local path
        relative_path = key.replace(prefix, "")
        local_path = os.path.join(local_root, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if not os.path.exists(local_path):
            s3.download_file(bucket_name, key, local_path)

if __name__ == "__main__":
    download_chbmit_s3()