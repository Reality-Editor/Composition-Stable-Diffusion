from PIL import Image, ImageDraw, ImageFilter
import requests
import numpy as np
import glob, os
import torch

from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler

model_name = "runwayml/stable-diffusion-inpainting"
log_path = "logs/sofa_caption/checkpoint-1000"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
pipe.unet.load_attn_procs(log_path)
pipe = pipe.to("cuda")

src_dir = "data/sofa_test"
os.makedirs(f'out/{log_path}', exist_ok=True)

img_paths = glob.glob(os.path.join(src_dir, '*.jpg'))
img_paths.sort()

for img_path in img_paths:
    init_image = Image.open(img_path).resize((512, 512))
    mask_image = Image.open(img_path[:-4] + '.png').resize((512, 512))
    mask_image = mask_image.filter(ImageFilter.MaxFilter(11))
    
    prompt = "a leather sofa in living room"
    image = pipe(prompt=prompt, image=init_image, 
                 mask_image=mask_image
                 ).images[0]

    cat_image = Image.new('RGB', (512 * 2, 512))

    masked_image = Image.composite(mask_image, init_image, mask_image)
    cat_image.paste(masked_image, (512*0, 0))
    cat_image.paste(image, (512*1, 0))

    cat_image.save(f'out/{log_path}/{os.path.basename(img_path)}')
