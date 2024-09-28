#!/bin/bash

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CLASS_NAME="man"
export INSTANCE_NAME='elon_musk'

# Using if, elif, and else to check the value of MODEL_NAME
if [ "$MODEL_NAME" == "CompVis/stable-diffusion-v1-4" ]; then
    Model_IDENTITY=sdv1-4
    echo "Model is CompVis/stable-diffusion-v1-4"
elif [ "$MODEL_NAME" == "runwayml/stable-diffusion-v1-5" ]; then
    Model_IDENTITY=sdv1-5
    echo "Model is runwayml/stable-diffusion-v1-5"
elif [ "$MODEL_NAME" == "stabilityai/stable-diffusion-2" ]; then
    Model_IDENTITY=sdv2
    echo "Model is stabilityai/stable-diffusion-2"
elif [ "$MODEL_NAME" == "stabilityai/stable-diffusion-2-base" ]; then
    Model_IDENTITY=sdv2-base
    echo "Model is stabilityai/stable-diffusion-2-base"
elif [ "$MODEL_NAME" == "stabilityai/stable-diffusion-2-1" ]; then
    Model_IDENTITY=sdv2-1
    echo "Model is stabilityai/stable-diffusion-2-1"
elif [ "$MODEL_NAME" == "stabilityai/stable-diffusion-2-1-base" ]; then
    Model_IDENTITY=sdv2-1-base
    echo "Model is stabilityai/stable-diffusion-2-1-base"
else
    echo "Unknown model"
fi

export INSTANCE_DIR="dreambooth_data/$INSTANCE_NAME"
export CLASS_DIR="generated_images/$CLASS_NAME-images"

export OUTPUT_DIR="checkpoints/checkpoint_$INSTANCE_NAME-$Model_IDENTITY"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks $CLASS_NAME" \
  --class_prompt="a photo of $CLASS_NAME" \
  --resolution=512 \
  --train_batch_size=1 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --checkpointing_steps=200 \
  # --push_to_hub