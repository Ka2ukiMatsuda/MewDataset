import os
import pandas as pd
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from tqdm import tqdm

MODE = "short"  # 'short' または 'long' を設定
IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"


model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]

captions = []
for image_file in tqdm(image_files):
    image_path = os.path.join(IMAGE_DIR, image_file)
    raw_image = Image.open(image_path).convert('RGB')
    
    if MODE == "short":
        prompt = "[INST] <image>\nProvide a one-sentence caption for the provided image. [/INST]"
        output_file = "llava-next_short.jsonl"
    else:
        prompt = "[INST] <image>\nDescribe the image in detail.\n[/INST]"
        output_file = "llava-next_long.jsonl"
    
    inputs = processor(prompt, raw_image, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    caption_text = processor.decode(outputs[0], skip_special_tokens=True).strip()
    assistant_index = caption_text.find("\nASSISTANT: ") + len("\nASSISTANT: ")
    caption_text = caption_text[assistant_index:]
    
    caption = {
        "image": image_file,
        "caption": caption_text
    }
    captions.append(caption)

captions_df = pd.DataFrame(captions)
captions_df['caption'] = captions_df['caption'].str.split('INST] ').str[1]
captions_df.to_json(os.path.join(OUTPUT_DIR, output_file), orient="records", force_ascii=False, lines=True)
