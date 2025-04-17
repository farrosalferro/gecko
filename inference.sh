#!/bin/bash

read -r -d '' QUESTION << EOM
What is the color of the little girl's shirt?
(A) yellow
(B) pink
(C) white
(D) black
Answer with the option's letter from the given choices directly.
EOM

MODEL=farrosalferro24/Gecko-Mantis-8B-siglip-llama3
IMAGES=examples/little_girl.jpg
VISUALIZE_TOP_PATCHES=True
VISION_FEATURE_SELECT_STRATEGY=cls
PATCH_PICKING_STRATEGY=last_layer
KEYWORD_CRITERIA=word
POSITIONAL_INFORMATION=explicit
CROPPING_METHOD=naive
TOPK=1

python inference.py \
    --question "$QUESTION" \
    --model "$MODEL" \
    --images "$IMAGES" \
    --visualize_top_patches "$VISUALIZE_TOP_PATCHES" \
    --vision_feature_select_strategy "$VISION_FEATURE_SELECT_STRATEGY" \
    --patch_picking_strategy "$PATCH_PICKING_STRATEGY" \
    --keyword_criteria "$KEYWORD_CRITERIA" \
    --positional_information "$POSITIONAL_INFORMATION" \
    --cropping_method "$CROPPING_METHOD" \
    --topk $TOPK
