import os
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm

MODE = "long"  # 'short' または 'long' を設定
IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"

model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]

captions = []
for image_file in tqdm(image_files):
    image_path = os.path.join(IMAGE_DIR, image_file)
    raw_image = Image.open(image_path).convert('RGB')
    
    if MODE == "short":
        prompt = "USER: <image>\nProvide a one-sentence caption for the provided image.\nASSISTANT:"
        output_file = "llava15_short.jsonl"
    else:
        prompt = "USER: <image>\nDescribe the image in detail.\nASSISTANT:"
        output_file = "llava15_long.jsonl"
    
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    caption_text = processor.decode(outputs[0], skip_special_tokens=True).strip()
    assistant_index = caption_text.find("\nASSISTANT: ") + len("\nASSISTANT: ")
    caption_text = caption_text[assistant_index:]
    
    caption = {
        "image": image_file,
        "caption": caption_text
    }
    captions.append(caption)

    captions_df = pd.DataFrame(captions)
    captions_df.to_json(os.path.join(OUTPUT_DIR, output_file), orient="records", force_ascii=False, lines=True)
