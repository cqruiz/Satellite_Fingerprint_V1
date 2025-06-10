import os
import requests
import json
import time
from tqdm import tqdm

def download_zenodo_dataset(record_id, download_dir="./downloads"):
    """
    Download all files from a Zenodo record.
    
    Args:
        record_id (str): The Zenodo record ID
        download_dir (str): Directory to save downloaded files
    """
    # Create the download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Get record metadata
    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url)
    
    if response.status_code != 200:
        print(f"Error fetching record metadata. Status code: {response.status_code}")
        return
    
    record_data = response.json()
    
    # Extract files info
    files = record_data.get("files", [])
    
    if not files:
        print("No files found in this Zenodo record")
        return
    
    # Download each file
    print(f"Found {len(files)} files to download")
    
    for file_info in files:
        file_name = file_info.get("key")
        file_size = file_info.get("size")
        download_url = file_info.get("links", {}).get("self")
        
        if not download_url:
            print(f"Could not find download URL for {file_name}")
            continue
        
        output_path = os.path.join(download_dir, file_name)
        
        # Check if file already exists
        if os.path.exists(output_path) and os.path.getsize(output_path) == file_size:
            print(f"File {file_name} already exists and has the correct size. Skipping.")
            continue
        
        print(f"Downloading {file_name} ({file_size / (1024 * 1024):.2f} MB)...")
        
        # Download file with progress bar
        response = requests.get(download_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"Downloaded {file_name} successfully")
        
        # Add a small delay between downloads to avoid overloading the server
        time.sleep(1)
    
    print(f"All files downloaded to {download_dir}")

if __name__ == "__main__":
    # Replace with your Zenodo record ID
    record_id = "8220494"
    download_dir = "./zenodo_dataset"
    
    download_zenodo_dataset(record_id, download_dir)