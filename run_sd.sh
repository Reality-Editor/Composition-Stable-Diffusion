export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_DIR="data/T-shirt"
export Test_DIR="data/T-shirt_test"
export OUT_DIR="out/T-shirt"
export INSTANCE_PROMPT="T-shirt"
export MODEL_DIR="logs/T-shirt"

# preprocess data
python preprocess.py --instance_data_dir $INSTANCE_DIR \
                     --instance_prompt $INSTANCE_PROMPT

# CUDA_VISIBLE_DEVICES=0
accelerate launch --num_processes 1 train_sd.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$MODEL_DIR \
  --instance_prompt=$INSTANCE_PROMPT \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --checkpointing_steps=5=1000


python inference_sd.py --image_path $Test_DIR \
                    --model_path $MODEL_DIR \
                    --out_path $OUT_DIR \
                    --instance_prompt $INSTANCE_PROMPT