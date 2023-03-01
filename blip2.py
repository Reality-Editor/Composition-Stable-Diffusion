from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import glob,os,tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
model.to(device)

src_dir = "data/sofa_test"

img_files = glob.glob(os.path.join(src_dir, "*.jpg"))
# img_files.sort()
for img_file in tqdm.tqdm(img_files):
    prompt_path = img_file[:-4] + '.txt'
    token_path = img_file[:-4] + '.pt'
    if os.path.exists(token_path): continue

    image = Image.open(img_file)
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    with open(prompt_path, 'w') as f:
        f.write(generated_text)

    # # save image embedding
    # pixel_values = inputs['pixel_values']
    # batch_size = pixel_values.shape[0]
    # image_embeds = model.vision_model(pixel_values, return_dict=True).last_hidden_state
    # image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    # query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    # query_outputs = model.qformer(
    #     query_embeds=query_tokens,
    #     encoder_hidden_states=image_embeds,
    #     encoder_attention_mask=image_attention_mask,
    #     return_dict=True,
    # )
    # query_output = query_outputs.last_hidden_state
    # torch.save(query_output.detach().cpu(), token_path)
