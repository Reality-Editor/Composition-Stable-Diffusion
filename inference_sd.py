from PIL import Image, ImageDraw, ImageFilter
import requests
import numpy as np
import glob, os
import torch
import argparse
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of preprocessing daa.")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to source directory.",
    )

    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="Path to output directory.",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to output directory.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to destinate directory.",
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(args.model_path)
    pipe = pipe.to(device)

    os.makedirs(f'{args.out_path}', exist_ok=True)

    if os.path.isdir(args.image_path):
        img_paths = glob.glob(os.path.join(args.image_path, '*.jpg'))
        img_paths.sort()
    else:
        img_paths = [args.image_path]

    # clipseg for image segmentation
    processor_clipseg = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model_clipseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model_clipseg.to(device)

    for img_path in img_paths:
        init_image = Image.open(img_path).convert("RGB")
        init_size = init_image.size
        init_image = init_image.resize((512, 512))

        inputs_clipseg = processor_clipseg(text=[args.instance_prompt], images=[init_image], padding="max_length", return_tensors="pt").to(device)
        outputs = model_clipseg(**inputs_clipseg)
        preds = outputs.logits.unsqueeze(0)[0].detach().cpu()
        mask_image = transforms.ToPILImage()(torch.sigmoid(preds)).convert("L").resize((512, 512))
        mask_image = mask_image.filter(ImageFilter.MaxFilter(21))

        prompt = f"a photo with sks {args.instance_prompt}"
        image = pipe(prompt=prompt, image=init_image, 
                    mask_image=mask_image
                    ).images[0]

        cat_image = Image.new('RGB', (512 * 2, 512))

        masked_image = Image.composite(mask_image, init_image, mask_image)
        cat_image.paste(init_image, (512*0, 0))
        cat_image.paste(image, (512*1, 0))

        cat_image.save(f"{args.out_path}/{os.path.basename(img_path)}")
