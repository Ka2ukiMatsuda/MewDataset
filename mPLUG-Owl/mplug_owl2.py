import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import TextStreamer

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

MODE = "short"  # 'short' または 'long' を設定
IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"

model_path = 'MAGAer13/mplug-owl2-llama2-7b'
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device="cuda")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]

captions = []
for image_file in tqdm(image_files):
    image_path = os.path.join(IMAGE_DIR, image_file)
    raw_image = Image.open(image_path).convert('RGB')
    
    if MODE == "short":
        prompt = "USER: Provide a one-sentence caption for the provided image. ASSISTANT: "
        output_file = "mplug_owl2_short.jsonl"
    else:
        prompt = "Describe the image."
        output_file = "mplug_owl2_long.jsonl"
    
    max_edge = max(raw_image.size)
    raw_image = raw_image.resize((max_edge, max_edge))
    image_tensor = process_images([raw_image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    conv = conv_templates["mplug_owl2"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            # streamer=streamer,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id
        )
    caption_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    caption_text = caption_text.replace("<\/s>", "")
    
    caption = {
        "image": image_file,
        "caption": caption_text
    }
    captions.append(caption)

    captions_df = pd.DataFrame(captions)
    captions_df.to_json(os.path.join(OUTPUT_DIR, output_file), orient="records", force_ascii=False, lines=True)
