import os
import time
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

# 🖼️ Folder containing 10 test JPEGs (change this to your own path)
image_folder = Path("sample_images")  # Put your 10 images here

# 💡 Transforms for "full preprocessing"
transform_full = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Just resize and convert to tensor (no normalize, for partial prep)
transform_partial = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_size(path):
    return os.path.getsize(path) / 1024  # KB

def benchmark_strategy(description, loader_fn):
    total_time = 0
    total_size = 0
    shapes = []
    for img_path in image_folder.glob("*.jpg"):
        start = time.time()
        data, size = loader_fn(img_path)
        total_time += time.time() - start
        total_size += size
        shapes.append(tuple(data.shape))
    print(f"\n📦 {description}")
    print(f"Avg Load Time: {total_time / len(shapes):.4f}s")
    print(f"Avg Size: {total_size / len(shapes):.2f} KB")
    print(f"Example Tensor Shape: {shapes[0]}")

# 1. Raw image (load + full transform)
benchmark_strategy("Raw JPEG + Full Transform", lambda path: (
    transform_full(Image.open(path).convert("RGB")),
    get_size(path)
))

# 2. Resized JPEG (manually saved)
def resized_jpeg_loader(path):
    resized_path = path.with_name("resized_" + path.name)
    if not resized_path.exists():
        img = Image.open(path).convert("RGB").resize((224, 224))
        img.save(resized_path, "JPEG", quality=95)
    return transform_full(Image.open(resized_path)), get_size(resized_path)

benchmark_strategy("Resized JPEG", resized_jpeg_loader)

# 3. Tensor only (no full transform)
benchmark_strategy("Tensor (No Normalize)", lambda path: (
    transform_partial(Image.open(path).convert("RGB")),
    transform_partial(Image.open(path).convert("RGB")).numpy().nbytes / 1024
))

# 4. Fully preprocessed Tensor (simulate cache hit)
benchmark_strategy("Tensor (Fully Transformed)", lambda path: (
    transform_full(Image.open(path).convert("RGB")),
    transform_full(Image.open(path).convert("RGB")).numpy().nbytes / 1024
))
