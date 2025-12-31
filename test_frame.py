# test_frame.py
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

frame_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)  # Test frame

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.fromarray(frame_rgb)
tensor = transform(image)
print(f"Succès! Tensor shape: {tensor.shape}")
