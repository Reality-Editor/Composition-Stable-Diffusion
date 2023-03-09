export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_DIR="data/T-shirt"
export Test_DIR="data/T-shirt_test"
export MODEL_DIR="logs/T-shirt"
export OUT_DIR="out/T-shirt"
export INSTANCE_PROMPT="T-shirt"

# preprocess data
python preprocess.py --instance_dir $INSTANCE_DIR \
                     --instance_prompt $INSTANCE_PROMPT

# CUDA_VISIBLE_DEVICES=0
accelerate launch --num_processes 1 train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$MODEL_DIR \
  --instance_prompt=$INSTANCE_PROMPT \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --max_train_steps=1000 \

python inference.py --image_path $Test_DIR \
                    --model_path $MODEL_DIR \
                    --out_path $OUT_DIR \
                    --instance_prompt $INSTANCE_PROMPT
