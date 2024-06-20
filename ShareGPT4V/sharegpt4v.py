from share4v.eval.run_share4v import *
from tqdm import tqdm
import os
import pandas as pd
import torch
from PIL import Image

# Place this file in ShareGPT4V/ directory
# Edit eval_model in ShareGPT4V/share4v/eval/run_share4v.py as follows
# def eval_model(args, model_name, tokenizer, model, image_processor):
#     # Model
#     disable_torch_init()

#     # model_name = get_model_name_from_path(args.model_path)
#     # tokenizer, model, image_processor, context_len = load_pretrained_model(
#     #     args.model_path, args.model_base, model_name)

#     qs = args.query
#     if model.config.mm_use_im_start_end:
#         qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
#             DEFAULT_IM_END_TOKEN + '\n' + qs
#     else:
#         qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

#     if 'llama-2' in model_name.lower():
#         conv_mode = "share4v_llama_2"
#     elif "v1" in model_name.lower():
#         conv_mode = "share4v_v1"
#     elif "mpt" in model_name.lower():
#         conv_mode = "mpt"
#     else:
#         conv_mode = "share4v_v0"

#     if args.conv_mode is not None and conv_mode != args.conv_mode:
#         print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
#             conv_mode, args.conv_mode, args.conv_mode))
#     else:
#         args.conv_mode = conv_mode

#     conv = conv_templates[args.conv_mode].copy()
#     conv.append_message(conv.roles[0], qs)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()

#     image = load_image(args.image_file)
#     image_tensor = image_processor.preprocess(image, return_tensors='pt')[
#         'pixel_values'].half().cuda()

#     input_ids = tokenizer_image_token(
#         prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

#     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#     keywords = [stop_str]
#     stopping_criteria = KeywordsStoppingCriteria(
#         keywords, tokenizer, input_ids)

#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=image_tensor,
#             do_sample=True,
#             temperature=0.2,
#             max_new_tokens=1024,
#             use_cache=True,
#             stopping_criteria=[stopping_criteria])

#     input_token_len = input_ids.shape[1]
#     n_diff_input_output = (
#         input_ids != output_ids[:, :input_token_len]).sum().item()
#     if n_diff_input_output > 0:
#         print(
#             f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
#     outputs = tokenizer.batch_decode(
#         output_ids[:, input_token_len:], skip_special_tokens=True)[0]
#     outputs = outputs.strip()
#     if outputs.endswith(stop_str):
#         outputs = outputs[:-len(stop_str)]
#     outputs = outputs.strip()
#     # print(outputs)
#     return outputs


MODE = "long"  # 'short' または 'long' を設定
IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"

model_path = "Lin-Chen/ShareGPT4V-7B"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]

captions = []
for image_file in tqdm(image_files):
    image_path = os.path.join(IMAGE_DIR, image_file)
    raw_image = Image.open(image_path).convert('RGB')
    
    if MODE == "short":
        prompt = "Provide a one-sentence caption for the provided image."
        output_file = "sharegpt4v_short.jsonl"
    else:
        prompt = "Describe the image in detail."
        output_file = "sharegpt4v_long.jsonl"
    
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": model_name,
        "query": prompt,
        "conv_mode": None,
        "image_file": image_path,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    
    caption = eval_model(args, model_name, tokenizer, model, image_processor)
    captions.append({"image": image_file, "caption": caption})

    captions_df = pd.DataFrame(captions)
    captions_df.to_json(os.path.join(OUTPUT_DIR, output_file), orient="records", force_ascii=False, lines=True)
