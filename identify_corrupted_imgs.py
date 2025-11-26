import os

from PIL import Image
from tqdm import tqdm

def is_image_corrupted(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that it is, in fact an image
        return False
    except (IOError, SyntaxError):
        return True
    
def find_corrupted_images(directory):
    corrupted_images = []
    for root, _, files in os.walk(directory):
        for file in tqdm(files, desc=f"Checking images in {root}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_path = os.path.join(root, file)
                if is_image_corrupted(image_path):
                    corrupted_images.append(image_path)
    return corrupted_images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Identify corrupted images in a directory.")
    parser.add_argument("directory", type=str, help="Path to the directory to scan for corrupted images.")
    args = parser.parse_args()

    corrupted_images = find_corrupted_images(args.directory)
    
    if corrupted_images:
        print("Corrupted images found:")
        with open("corrupted_images.txt", "w") as f:
            for img in corrupted_images:
                print(img)
                f.write(f"{img}\n")
    else:
        print("No corrupted images found.")