import os
import pandas as pd
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm

IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")

image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]

captions = []
for image_file in tqdm(image_files):
    image_path = os.path.join(IMAGE_DIR, image_file)
    raw_image = Image.open(image_path).convert('RGB')
    
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
    out = model.generate(**inputs)
    short_caption = processor.decode(out[0], skip_special_tokens=True).strip()
    
    caption = {
        "image": image_file,
        "caption": short_caption
    }
    captions.append(caption)

    captions_df = pd.DataFrame(captions)
    captions_df.to_json(os.path.join(OUTPUT_DIR, "blip2_short.jsonl"), orient="records", force_ascii=False, lines=True)
