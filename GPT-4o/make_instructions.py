import os
from PIL import Image
from tqdm import tqdm
import json 
import base64
from openai import OpenAI
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

MODE = "long"  # 'short' または 'long' を設定
IMAGE_DIR = "images"
OUTPUT_DIR = "GPT-4o/outputs"

image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]

instructions = []
for image_file in tqdm(image_files):
    image_path = os.path.join(IMAGE_DIR, image_file)
    raw_image = Image.open(image_path).convert('RGB')
    
    with open(image_path, "rb") as image_file_obj:
        base64_image = base64.b64encode(image_file_obj.read()).decode('utf-8')
    
    if MODE == "short":
        prompt = "Provide a one-sentence caption for the provided image."
        output_file = "dci_short.jsonl"
    else:
        prompt = "Describe the image in detail."
        output_file = "dci_long.jsonl"
    

    instruction = {
        "custom_id": image_file,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                    "max_tokens": 1000
                }
            ]
        }
    }
    instructions.append(instruction)
    # break
    
instructions = sorted(instructions, key=lambda x: x['custom_id'])
print([instruction['custom_id'] for instruction in instructions[:5]])

print(len(instructions))

num_splits = 20
split_size = len(instructions) // num_splits
for i in tqdm(range(num_splits), desc="Generating instructions"):
    start_index = i * split_size
    end_index = (i + 1) * split_size if i != num_splits - 1 else len(instructions)
    with open(os.path.join(OUTPUT_DIR, f"{output_file}_{i+1}.jsonl"), 'w', encoding='utf-8') as f:
        for instruction in instructions[start_index:end_index]:
            f.write(json.dumps(instruction, ensure_ascii=False) + '\n')

for i in tqdm(range(num_splits), desc="Creating batch files"):
    batch_input_file = client.files.create(
        file=open(os.path.join(OUTPUT_DIR, f"{output_file}_{i+1}.jsonl"), "rb"),
            purpose="batch"
        )
    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"{output_file}_{i+1}.jsonl"
        }
    )
