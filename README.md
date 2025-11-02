# AI Photo Renamer

A command line Python script that renames image files to append the date taken and a brief AI generated caption.

# Installation

1. Download the latest release of the script and extract it from the zip file.
2. Install the Python dependencies for the script using pip. 
Use a Python virtual environment in the extracted script's directory if you'd like, installing the dependencies in the virtual environment and activating it:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install pillow pillow-heif piexif transformers torch
```

NOTE: if you decide to go the virtual environment route, do not forget to activate the environment each time you run the script.

3. Initialize the script by running the following command:

```bash
python3 aiphotorename.py --init
```

This downloads the AI models locally and caches them in your user profile.
The models get cached in ~/.cache/huggingface.
You only need to use the --init parameter once.
The next time you run the script, it will use the AI models locally.

To generate the AI caption, the script saves a temporary copy of each image as a jpeg file.
As of this writing, the script has only be tested on Linux.
To run this on Windows, you will likely need to modify the script to use an appropriate Windows temp directory (i.e. C:\Windows\Temp).
Set the TEMP_DIR variable to the correct directory for your platform.
In the future, I may update the code to choose the correct directory based on platform. 

# Usage

To rename a file, run the following command, replacing IMG_0000.JPG with the filename of your photo:

```bash
python3 aiphotorename.py IMG_0000.JPG
```

This would rename the file using the following naming convention:

```bash
IMG_0000_YYYYMMDD_CAPTION
```

Where:

- YYYMMDD is date taken
- CAPTION is a brief AI generated caption in CamelCase.

You may specify multiple photo filenames separated by spaces.
In bash, using wildcards are the best way to specify multiple files as arguments.
For instance, if you have photos in a directory, you can run:

```bash
python3 aiphotorename.py /home/user/photos/*
```

The script ignores any files that are not photos.

To preview the changes it would make and ensure it is selecting the right photos, you can run it with the --dry-run parameter:

```bash
python3 aiphotorename.py --dry-run /home/user/photos/*
```

Additionally, you can have the script prompt for confirmation on each rename:

```bash
python3 aiphotorename.py --confirm /home/user/photos/*
```

To see all available options, run:

```bash
python3 aiphotorename.py --help
```

# Warning

While this is the first release of the script, it should be considered experimental.
It has gone through some limited testing and I have successfully used it for my own purposes.
Use at your own risk!