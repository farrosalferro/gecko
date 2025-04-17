#!/bin/bash

MODEL=farrosalferro24/Gecko-Mantis-8B-siglip-llama3 # or you can use farrosalferro24/Gecko-Mantis-8B-clip-llama3
VISION_ENCODER=$(echo "$MODEL" | cut -d'-' -f4)

echo "Running evaluation for model: $MODEL"

python vstar_eval.py \
    --question_file ../benchmarks/vstar_bench/test_questions.jsonl \
    --answers_file ../answers/"${VISION_ENCODER}_vstar.jsonl" \
    --image_folder ../benchmarks/vstar_bench/ \
    --model_name_or_path $MODEL \
    --conv_format llama_3 \
    --temperature 0 \
    --topk 1 \
    --cropping_method dynamic \
    --vision_feature_select_strategy cls \
    --patch_picking_strategy last_layer \
    --keyword_criteria word \
    --positional_information explicit

python convert_vstar_for_eval.py \
    --test_file ../benchmarks/vstar_bench/test_questions.jsonl \
    --answers_file ../answers/"${VISION_ENCODER}_vstar.jsonl" \
    --output_file ../answers/"${VISION_ENCODER}_vstar_output.jsonl"