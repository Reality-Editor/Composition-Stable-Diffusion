# Image Composition via Stable Diffusion 

### We achieve image composition via Stable Diffusion Model. Application includes Clothes Try-On and Furniture arrangement.

![Example 1](docs/image_composition-1.png "Title")

![Example 2](docs/image_composition-2.png "Title")

### Installation
```bash
pip install -r requirements.txt

```

### 0. Prepare Images
Please provide at least one images in .jpg format and instance prompt.

### 1. Set Environment
```bash
export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_DIR="data/sofa"
export Test_DIR="data/sofa_test"
export MODEL_DIR="logs/sofa"
export OUT_DIR="out/sofa"
export INSTANCE_PROMPT="sofa"
```

### 2. Preprocess Images
Please provide at least one images in .jpg format and instance prompt. The preprocess.py script will generate captions and instance masks.

```bash
python preprocess.py --instance_dir $INSTANCE_DIR \
                     --instance_prompt $INSTANCE_PROMPT
```

### 3. Finetune with LoRA
We then finetune with LoRA to embed the instance images and prompt into stable diffusion model.

```bash
accelerate launch --num_processes 1 train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$MODEL_DIR \
  --instance_prompt=$INSTANCE_PROMPT \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --max_train_steps=500
```

### 4. Image Composition
Finally, you can provide new images to achieve image composition.

```bash
python inference.py --image_path $Test_DIR \
                    --model_path $MODEL_DIR \
                    --out_path $OUT_DIR \
                    --instance_prompt $INSTANCE_PROMPT
```

### Or else
using end-to-end run.sh.

```bash
python inference.py --image_path $Test_DIR \
                    --model_path $MODEL_DIR \
                    --out_path $OUT_DIR \
                    --instance_prompt $INSTANCE_PROMPT
```

### Author: [RealityEditor](https://realityeditor.com.cn)
