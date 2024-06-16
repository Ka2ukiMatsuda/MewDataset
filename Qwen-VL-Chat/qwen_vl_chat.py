import os
import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

torch.manual_seed(1234)

MODE = "long"  # 'short' または 'long' を設定
IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True, torch_dtype=torch.float16).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]

captions = []
for image_file in tqdm(image_files):
    image_path = os.path.join(IMAGE_DIR, image_file)
    raw_image = Image.open(image_path).convert('RGB')
    
    
    if MODE == "short":
        prompt = "Describe the image in English:"
        output_file = "qwen_vl_chat_short.jsonl"
    else:
        prompt = "Describe the image in detail in English."
        output_file = "qwen_vl_chat_long.jsonl"
    
    query = tokenizer.from_list_format([
        {'image': image_path},
        {'text': prompt},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    caption_text = response.strip()
    
    caption = {
        "image": image_file,
        "caption": caption_text
    }
    captions.append(caption)

    captions_df = pd.DataFrame(captions)
    captions_df.to_json(os.path.join(OUTPUT_DIR, output_file), orient="records", force_ascii=False, lines=True)
