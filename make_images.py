import json
import os

# Flickr-30kのval, test splitの画像ファイルを使用
long_caps = []
with open("flickr30k_test_captions.jsonl", "r") as f:
    for line in f:
        long_caps.append(json.loads(line)["image_id"] + ".jpg") # len(long_caps) = 1023
with open("flickr30k_val_captions.jsonl", "r") as f:
    for line in f:
        long_caps.append(json.loads(line)["image_id"] + ".jpg") # len(long_caps) = 2032
print(long_caps[0])

# Flickr-ExpertとFlickr-CFに含まれる画像を除外
prohibited = set()
with open("CrowdFlowerAnnotations.txt", "r") as file:
    for line in file:
        parts = line.split('\t')
        prohibited.add(parts[0].split('_')[0] + ".jpg") # len(prohibited) = 1000
with open("ExpertAnnotations.txt", "r") as file:
    for line in file:
        parts = line.split('\t')
        prohibited.add(parts[0].split('_')[0] + ".jpg") # len(prohibited) = 1000
prohibited = list(prohibited)
print(prohibited[0])

# long_capsからprohibitedに含まれる画像を除外
for filename in long_caps:
    if filename in prohibited:
        long_caps.remove(filename)

# flickr30k/Imagesからimagesにコピー (合計1973枚)
for filename in long_caps:
    os.system(f"cp flickr30k/Images/{filename} images/")
