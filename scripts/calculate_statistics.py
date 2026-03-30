import os
from PIL import Image
import numpy as np
from tqdm import tqdm

root = "/storage/datasets/SUN/data/TestHardDataset/Unseen/Frame/"
cls_list = os.listdir(root)
means = np.zeros(3)
stds = np.zeros(3)  
pixel_count = 0   
for cls in tqdm(cls_list):
    case_path = os.path.join(root, cls)
    image_list = os.listdir(case_path)
    for image_path in image_list:
        img = Image.open(os.path.join(case_path, image_path)).convert('RGB')

        img_array = np.array(img) / 255.0 

        num_pixels = img_array.shape[0] * img_array.shape[1]

        pixel_count += num_pixels

        means += img_array.mean(axis=(0, 1)) * num_pixels

means /= pixel_count

for cls in tqdm(cls_list):
    case_path = os.path.join(root, cls)
    image_list = os.listdir(case_path)
    for image_path in image_list:
        img = Image.open(os.path.join(case_path, image_path)).convert('RGB')

        img_array = np.array(img) / 255.0 

        num_pixels = img_array.shape[0] * img_array.shape[1]

        stds += (((img_array - means) ** 2).sum(axis=(0, 1))) * num_pixels

stds = np.sqrt(stds / pixel_count)

print(means)
print(stds)

