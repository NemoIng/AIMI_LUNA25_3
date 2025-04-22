import os
import requests
ACCESS_TOKEN = "uEWyMCfIH9IfXZM8vggtpKiwvAvhqgzToWwjpNuj3xj7hhIQ3YyIx8tQb5Qe"
record_id = "14223624" #LUNA25 record id

# Specify the output folder where files will be saved
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Define the specific filenames to download
target_filenames = [
"luna25_nodule_blocks.zip.002"]

# Get the metadata of the Zenodo record
r = requests.get(f"https://zenodo.org/api/records/{record_id}", params={'access_token': ACCESS_TOKEN})
if r.status_code != 200:
    print("Error retrieving record:", r.status_code, r.text)
    exit()

# Extract download URLs and filenames
all_files_data = r.json()['files']
download_urls = [f['links']['self'] for f in all_files_data]
filenames = [f['key'] for f in all_files_data]

# Filter the download URLs and filenames to only include the target files
filtered_download_urls = []
filtered_filenames = []

for index, filename in enumerate(filenames):
    if filename in target_filenames:
        filtered_download_urls.append(download_urls[index])
        filtered_filenames.append(filename)

print(f"Total files to download: {len(filtered_download_urls)}")

# Download the specified files
for index, (filename, url) in enumerate(zip(filtered_filenames, filtered_download_urls)):
    file_path = os.path.join(output_folder, filename)

    print(f"Downloading file {index + 1}/{len(filtered_download_urls)}: {filename} -> {file_path}")

    with requests.get(url, params={'access_token': ACCESS_TOKEN}, stream=True) as r:
        r.raise_for_status()  # Raise an error for failed requests
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):  # Download in chunks
                f.write(chunk)

    print(f"Completed: {filename}")

print("All downloads completed successfully!")
