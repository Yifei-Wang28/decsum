#!/bin/bash
set -x

OUTPUT_DIR=data/test_random/
MODEL_PATH=data/test_decsum/transformers/version_14-10-2024--21-43-45/checkpoints/epoch=2-val_loss=0.13.ckpt/
RES_DIR=${OUTPUT_DIR}models/sentence_select/

# DecSum
for m in Transformer # 
do
    for s in  window_1_DecSum_WD_sentbert #
    do
        for d in yelp # 
        do
            for k in 50trunc #10 # 5
            do
                (python -m models.sentence_select.rand \
                        --device 0 --feature_used notes \
                        --trained_model_path ${MODEL_PATH} \
                        --result_dir ${RES_DIR} \
                        --data_dir ${OUTPUT_DIR} \
                        --num_sentences $k \
                        --segment $s\_$k \
                        --model $m \
                        --data $d \
                        --num_review 50 \
                        --target_type reg )
            done
        done
    done
done
