import os
import json
from tqdm import tqdm

with open("DCI/data/densely_captioned_images/splits.json") as f:
    splits = json.load(f)
    
train_files = splits["train"][:2000]
paths = [f"DCI/data/densely_captioned_images/complete/{f}" for f in train_files]

for path in tqdm(paths):
    with open(path, "r") as file:
        data = json.load(file)
        filename = data["image"]
        os.system(f"cp DCI/data/densely_captioned_images/photos/{filename} images/")
