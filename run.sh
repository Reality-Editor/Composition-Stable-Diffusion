export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_DIR="data/sofa"
export Test_DIR="data/sofa_test"
export OUT_DIR="out/sofa"
export INSTANCE_PROMPT="sofa"
export MODEL_DIR="logs/sofa"

# preprocess data
python preprocess.py --instance_data_dir $INSTANCE_DIR \
                     --instance_prompt $INSTANCE_PROMPT

# CUDA_VISIBLE_DEVICES=0
accelerate launch --num_processes 1 train.py \
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
  --max_train_steps=1000

python inference.py --image_path $Test_DIR \
                    --model_path $MODEL_DIR \
                    --out_path $OUT_DIR \
                    --instance_prompt $INSTANCE_PROMPT