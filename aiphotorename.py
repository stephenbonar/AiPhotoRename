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
import argparse
from datetime import datetime
from pillow_heif import register_heif_opener
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Constants.
TEMP_DIR = "/tmp"
DATE_FORMAT = "%Y:%m:%d %H:%M:%S"
MODEL_NAME = "Salesforce/blip-image-captioning-base"
TOKENS_TO_SKIP = {
    'a', 'an', 'the', 'in', 'on', 'at', 'of', 'and', 'or', 'is', 
    'are', 'was', 'were', 'with', 'to', 'for', 'around', 'that'
}
PROGRAM_NAME = "AI Photo Renamer"
PROGRAM_VERSION = "v1.0.0"
PROGRAM_COPYRIGHT = "Copyright (C) 2025 Stephen Bonar"

# Global debug variables.
minlength = 255
maxlength = 0


def generate_date_time_part(exif_data):
    """
    Extracts the original date and time from EXIF data and returns a string
    in the format 'YYYYMMDD' for use in filenames.

    Args:
        exif_data (bytes or None): The EXIF data from an image file.

    Returns:
        str or None: The formatted date string if available, otherwise None.
    """
    
    date_time_original = None
    if exif_data:
        exif_dict = piexif.load(exif_data)
        date_bytes = exif_dict['Exif'].get(piexif.ExifIFD.DateTimeOriginal)
        if date_bytes:
            date_string = date_bytes.decode()
            date_time_original = datetime.strptime(date_string, DATE_FORMAT)

    if date_time_original:
        year = date_time_original.year
        month = date_time_original.month
        day = date_time_original.day
        part = f"{year}{month:02d}{day:02d}"
        return part

    return None


def generate_caption_part(offline):
    """
    Generates a CamelCase caption string suitable for use in a filename
    by running an AI image captioning model on a temporary JPEG image.

    Returns:
        str: The processed caption string with stopwords and duplicates removed.
    """

    # Establishes the device to run the model on, either the GPU or CPU. GPU is
    # preferred if available as GPUs are much faster for AI and learning. 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NOTE: On first run, take out the local_files_only=True to download the
    # model. This will download to ~/.cache/huggingface
    #
    # Loads a pre-trained BLIP image captioning processor from the Hugging Face
    # model hub.
    processor = BlipProcessor.from_pretrained(
        MODEL_NAME, local_files_only=offline, use_fast=True
    )

    # Loads a pre-trained BLIP image captioning model from the Hugging Face
    # model hub and moves it to the selected device.
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_NAME, local_files_only=offline
    ).to(device)

    # Load the temporary copy of the image as this is the version of the
    # image that was converted from the original into jpeg for AI processing.
    temp_image = Image.open("/tmp/ai.jpg").convert("RGB")

    # Prepare the image for the model by preprocessing it and converting it
    # into PyTorch tensors, then moving the tensors to the selected device.
    tensors = processor(images=temp_image, return_tensors="pt").to(device)

    # Obtain the batch of token IDs from the model by unpacking the tensors and
    # passing them as key-value pairs to the model's generate method so we can
    # generate a caption from the token ids.
    token_id_batch = model.generate(**tensors)

    # Obtains the full caption string by decoding the first set of token IDs in
    # the batch (there is only 1 batch) and skipping any special tokens like
    # <pad> or <end>. This gives us a human-readable caption. That said, we will
    # want to clean up the caption in the following lines to make it more
    # suitable for a filename.
    caption = processor.decode(token_id_batch[0], skip_special_tokens=True)

    # We will build the caption part of the filename by processing each token.
    caption_part = ''

    # Split the caption into individual tokens so we can process each one. 
    caption_tokens = caption.split()
    
    # We need to keep track of which tokens have already been added to avoid
    # duplicates.
    tokens_added = set()

    for token in caption_tokens:
        # We want to skip tokens like duplicates, particles, prepositions, and 
        # conjunctions as they make the filename too long.
        skip_token = token in TOKENS_TO_SKIP or token in tokens_added

        if not skip_token and token.isalpha():
            # Capitalize the first letter of the token so the caption part of
            # the filename is CamelCase.
            caption_part += token.capitalize()

            tokens_added.add(token)

    return caption_part


def generate_ai_filename(path, filename_stem, filename_extension, offline):
    """
    Generates a new filename for an image by combining the original filename
    stem, the date extracted from EXIF data (if available), and an 
    AI-generated caption.

    Args:
        path (str): The file path to the image.
        filename_stem (str): The base name of the file without extension.
        filename_extension (str): The file extension, including the dot.

    Returns:
        str or None: The newly generated filename, or None if an error occurs.
    """

    new_filename = None

    try:
        # Open the image so we can work with it.
        input_image = Image.open(path)

        # Convert images with unsupported modes to RGB before saving as JPEG.
        # This is mostly to fix .png images.
        if input_image.mode in ("RGBA", "P"):
            input_image = input_image.convert("RGB")

        # Get the datetime part of the filename so it can be used in the rename.
        exif_data = input_image.info.get("exif")
        date_time_part = generate_date_time_part(exif_data)

        # Make a temporary jpeg version of the image for AI processing as not
        # all formats such as HEIC are supported.
        input_image.save(f"{TEMP_DIR}/ai.jpg", format="JPEG")

        # Get the caption part of the filename so it can be used in the rename.
        caption_part = generate_caption_part(offline)

        # Build the new filename using the available parts.
        new_filename = f"{filename_stem}"
        if date_time_part:
            new_filename += f"_{date_time_part}_"
        else:
            new_filename += "_"
        new_filename += f"{caption_part}{filename_extension}"
    except Exception as e:
        print(f"Error generating new filename for {path}: {e}")

    return new_filename


def rename_photo(original_path, directory, new_filename):
    """
    Renames a photo file to a new filename within the specified directory.

    Args:
        original_path (str): The current file path of the image.
        directory (str): The directory where the file should be renamed.
        new_filename (str): The new filename to assign to the image.

    Returns:
        None
    """

    try:
        new_path = os.path.join(directory, new_filename)
        if os.path.exists(new_path):
            print(f"File already exists: {new_path}")
        else:
            os.rename(original_path, new_path)
    except Exception as e:
        print(f"Error renaming file: {e}")


if __name__ == "__main__":
    # Register HEIF opener so we can read HEIF files.
    register_heif_opener()

    # Ensure configured TEMP_DIR exists
    if not os.path.exists(TEMP_DIR):
        print(f"Error: TEMP_DIR '{TEMP_DIR}' does not exist.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description=PROGRAM_NAME
    )
    parser.add_argument(
        "-d", "--dry-run",
        action="store_true",
        help="Only print the new filename, do not rename files"
    )
    parser.add_argument(
        "-c", "--confirm",
        action="store_true",
        help="Prompt for confirmation before renaming each file"
    )
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Print version information and exit"
    )
    parser.add_argument(
        "-i", "--init",
        action="store_true",
        help="Download model files if not present (run in online mode)"
    )
    parser.add_argument(
        "image_files",
        nargs="*",
        help="Image files to process"
    )
    args = parser.parse_args()

    if args.version:
        print(f"{PROGRAM_NAME} {PROGRAM_VERSION}")
        print(PROGRAM_COPYRIGHT)
        sys.exit(0)

    if not args.image_files:
        parser.error("the following arguments are required: image_files")

    # Set offline mode based on --init flag
    offline = not args.init

    # Process each file, checking if Pillow can open it.
    for image_path in args.image_files:
        if os.path.isfile(image_path):
            try:
                # Try opening with Pillow to check if it's a supported image.
                with Image.open(image_path) as img:
                    # Split the path components so we can build a new filename.
                    directory, filename = os.path.split(image_path)
                    filename_stem, filename_ext= os.path.splitext(filename)

                    new_filename = generate_ai_filename(
                        image_path, filename_stem, filename_ext, offline
                    )

                    if new_filename:
                        print(f"Renaming {filename} to {new_filename}", end="")
                        if not args.dry_run:
                            if args.confirm:
                                prompt = " Proceed? [y/n]: "
                                response = input(prompt).strip().lower()
                                if response != "y":
                                    print(" Skipped.")
                                    continue
                                else:
                                    print()
                            else:
                                print()
                            rename_photo(image_path, directory, new_filename)
                        else:
                            print(" (dry-run)")
            except Exception as e:
                print(f"Skipping file: {image_path}, error: {e}")
        else:
            print(f"Skipping non-existent image file: {image_path}")
