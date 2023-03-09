from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torchvision import transforms
import torch
import glob, os, tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of preprocessing daa.")
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="Path to source directory.",
    )

    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="target object to be composed.",
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # # blip2 for image caption
    # processor_blip2 = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # model_blip2 = Blip2ForConditionalGeneration.from_pretrained(
    #     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    # )
    # model_blip2.to(device)
    
    # clipseg for image segmentation
    processor_clipseg = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model_clipseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model_clipseg.to(device)

    img_files = glob.glob(os.path.join(args.instance_data_dir, "*.jpg"))

    for img_file in tqdm.tqdm(img_files):
        prompt_path = img_file[:-4] + '.txt'

        image = Image.open(img_file).convert("RGB")

        # # blip2
        # inputs_blip2 = processor_blip2(images=image, return_tensors="pt").to(device, torch.float16)
        # generated_ids = model_blip2.generate(**inputs_blip2)
        # generated_text = processor_blip2.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # with open(prompt_path, 'w') as f:
        #     f.write(generated_text)
        
        # clipseg
        inputs_clipseg = processor_clipseg(text=[args.instance_prompt], images=[image], padding="max_length", return_tensors="pt").to('cuda')
        outputs = model_clipseg(**inputs_clipseg)
        preds = outputs.logits.unsqueeze(0)[0].detach().cpu()
        mask = transforms.ToPILImage()(torch.sigmoid(preds)).convert("L")
        mask.save(img_file[:-4] + '.png')
