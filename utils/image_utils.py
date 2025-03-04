from PIL import Image
import numpy as np

def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size)
    return img

def save_image(image, path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(path)

def normalize_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    return image.astype(np.float32) / 255.0

def denormalize_image(image):
    return (image * 255).astype(np.uint8)