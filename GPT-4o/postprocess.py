import os
import json
from glob import glob
import pandas as pd

MODE = "long"

batch_files = glob(f'GPT-4o/output_{MODE}/batch_*.jsonl')

merged_data = []
for batch_file in batch_files:
    with open(batch_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            custom_id = data['custom_id']
            response_content = data['response']['body']['choices'][0]['message']['content']
            merged_data.append({
                "image": custom_id,
                "caption": response_content
            })

merged_data = sorted(merged_data, key=lambda x: x['image'])

df = pd.DataFrame(merged_data)
output_file = f'GPT-4o/gpt4o_{MODE}.jsonl'
df.to_json(output_file, orient='records', lines=True, force_ascii=False)

image_dir = "images"
image_files = set(os.listdir(image_dir))
image_keys = set(df['image'])

difference = image_files - image_keys
print("difference:", difference)
