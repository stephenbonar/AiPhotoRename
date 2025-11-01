# aiphotorename.py
#
# Copyright (C) 2025 Stephen Bonar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http ://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import piexif
import torch
from datetime import datetime
from pillow_heif import register_heif_opener
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# def get_date_taken_from_heic(filepath):
#     try:
#         heif_file = pyheif.read(filepath)
#         exif_data = None
#         for metadata in heif_file.metadata or []:
#             if metadata['type'] == 'Exif':
#                 exif_data = metadata['data']
#                 break
#         if exif_data:
#             exif_dict = piexif.load(exif_data)
#             date_str = exif_dict['Exif'].get(piexif.ExifIFD.DateTimeOriginal)
#             if date_str:
#                 # date_str is bytes, e.g. b'2023:07:15 14:23:01'
#                 date_str = date_str.decode()
#                 return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
#     except Exception as e:
#         print(f"Error reading HEIC metadata: {e}")
#     # Fallback: use file's modification time
#     ts = os.path.getmtime(filepath)
#     return datetime.fromtimestamp(ts)

# def rename_heic_by_date(filepath):
#     date_taken = get_date_taken_from_heic(filepath)
#     if not date_taken:
#         print(f"Could not determine date for {filepath}")
#         return
#     dir_name, orig_filename = os.path.split(filepath)
#     ext = os.path.splitext(orig_filename)[1].lower()
#     new_filename = date_taken.strftime("IMG_%Y%m%d_%H%M%S") + ext
#     new_filepath = os.path.join(dir_name, new_filename)
#     # Avoid overwriting existing files
#     counter = 1
#     while os.path.exists(new_filepath):
#         new_filename = date_taken.strftime("IMG_%Y%m%d_%H%M%S") + f"_{counter}" + ext
#         new_filepath = os.path.join(dir_name, new_filename)
#         counter += 1
#     os.rename(filepath, new_filepath)
#     print(f"Renamed '{filepath}' to '{new_filepath}'")

def rename_heic(path):
    try:
        # Register HEIF opener so we can read HEIF files.
        register_heif_opener()

        # Open the HEIC image so we can extract the EXIF metadata.
        img = Image.open(path)

        # Try to get the original date taken so its available for renaming.
        exif_data = img.info.get("exif")
        date_time_original = None
        if exif_data:
            exif_dict = piexif.load(exif_data)
            date_bytes = exif_dict['Exif'].get(piexif.ExifIFD.DateTimeOriginal)
            if date_bytes:
                date_string = date_bytes.decode()
                print(f"Date taken: {date_string}")
                date_format = "%Y:%m:%d %H:%M:%S"
                date_time_original = datetime.strptime(date_string, date_format)

        directory, filename = os.path.split(path)
        name, extension = os.path.splitext(filename)

        if date_time_original:
            year = date_time_original.year
            month = date_time_original.month
            day = date_time_original.day
            hour = date_time_original.hour
            minute = date_time_original.minute
            second = date_time_original.second
            new_name = f"{name}_{year}{month}{day}_{hour}{minute}{second}"
            new_filename = new_name + extension
            print(new_filename)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # NOTE: On first run, take out the local_files_only=True to download the model.
        # This will download to ~/.cache/huggingface
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True, use_fast=True)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True).to(device)

        img.save("/tmp/ai.jpg", format="JPEG")

        # Load and process image
        image = Image.open("/tmp/ai.jpg").convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Generate caption
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        print("Caption:", caption)
        
        #new_filepath = os.path.join(directory, new_filename)
        #os.rename(path, new_filepath)
        #print(f"Renamed '{path}' to '{new_filepath}'")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Print usage if file names were not specified at the command line.
    if len(sys.argv) < 2:
        print("Usage: python aiphotorename.py <heic_file1> [<heic_file2> ...]")
        sys.exit(1)

    # Every arg after sys.argv[0] should be a HEIC file.
    for heic_path in sys.argv[1:]:
        if heic_path.lower().endswith('.heic') and os.path.isfile(heic_path):
            rename_heic(heic_path)
        else:
            print(f"Skipping non-HEIC file: {heic_path}")
