import os
import pandas as pd
import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from tqdm import tqdm

MODE = "short"  # 'short' または 'long' を設定
IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"

processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]

captions = []
for image_file in tqdm(image_files):
    image_path = os.path.join(IMAGE_DIR, image_file)
    raw_image = Image.open(image_path).convert('RGB')
    
    if MODE == "short":
        prompt = "A short image description:"
        output_file = "instructblip_short.jsonl"
    else:
        prompt = "Describe the image in detail."
        output_file = "instructblip_long.jsonl"
    
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    caption_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    
    caption = {
        "image": image_file,
        "caption": caption_text
    }
    captions.append(caption)

    captions_df = pd.DataFrame(captions)
    captions_df.to_json(os.path.join(OUTPUT_DIR, output_file), orient="records", force_ascii=False, lines=True)
