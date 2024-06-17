import os
import pandas as pd
import torch
from PIL import Image
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from tqdm import tqdm

MODE = "long"  # 'short' または 'long' を設定
IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"

pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'
model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
)
image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]

captions = []
for image_file in tqdm(image_files):
    image_path = os.path.join(IMAGE_DIR, image_file)
    raw_image = Image.open(image_path).convert('RGB')
    
    if MODE == "short":
        prompt = "USER: <image>Provide a one-sentence caption for the provided image. ASSISTANT: "
        output_file = "mplug_owl_short.jsonl"
    else:
        prompt = "<image>Describe the image."
        output_file = "mplug_owl_long.jsonl"
    
    images = [raw_image]
    prompts = [prompt]
    inputs = processor(text=prompts, images=images, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024
        )
    caption_text = tokenizer.decode(outputs.tolist()[0], skip_special_tokens=True).strip()
    
    caption = {
        "image": image_file,
        "caption": caption_text
    }
    captions.append(caption)

    captions_df = pd.DataFrame(captions)
    captions_df.to_json(os.path.join(OUTPUT_DIR, output_file), orient="records", force_ascii=False, lines=True)
