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


def get_date_time_part(exif_data):
    date_time_original = None
    if exif_data:
        exif_dict = piexif.load(exif_data)
        date_bytes = exif_dict['Exif'].get(piexif.ExifIFD.DateTimeOriginal)
        if date_bytes:
            date_string = date_bytes.decode()
            date_format = "%Y:%m:%d %H:%M:%S"
            date_time_original = datetime.strptime(date_string, date_format)
    
    if date_time_original:
        year = date_time_original.year
        month = date_time_original.month
        day = date_time_original.day
        hour = date_time_original.hour
        minute = date_time_original.minute
        second = date_time_original.second
        part = f"{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}{second:02d}"
        return part

    return None


def get_caption_part():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NOTE: On first run, take out the local_files_only=True to download the model.
    # This will download to ~/.cache/huggingface
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True).to(device)

    # Load and process image
    temp_image = Image.open("/tmp/ai.jpg").convert("RGB")
    inputs = processor(images=temp_image, return_tensors="pt").to(device)

    # Generate caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Replace all non-alpha characters with underscores
    caption = ''.join([c if c.isalpha() else '_' for c in caption])

    return caption


def rename_heic(path):
    try:
        # Split out the path components so we can build a new filename.
        directory, filename = os.path.split(path)
        name, extension = os.path.splitext(filename)

        # Register HEIF opener so we can read HEIF files.
        register_heif_opener()

        # Open the image so we can work with it.
        input_image = Image.open(path)

        # Get the datetime part of the filename so it can be used in the rename.
        exif_data = input_image.info.get("exif")
        date_time_part = get_date_time_part(exif_data)

        # Make a temporary jpeg version of the image for AI processing as not
        # all formats such as HEIC are supported.
        input_image.save("/tmp/ai.jpg", format="JPEG")

        # Get the caption part of the filename so it can be used in the rename.
        caption_part = get_caption_part()

        new_filename = f"{name}_{date_time_part}_{caption_part}{extension}"

        # Debug.
        print(f"Original: {filename}, New: {new_filename}")

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
